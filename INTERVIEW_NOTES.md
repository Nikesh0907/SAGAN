# SAGAN (Spatial-aware Attention GAN) — End-to-End Interview Notes

This note is a step-by-step script you can use to explain the model in detail, including shapes and what each part is doing.

---

## 0. Big Picture

- **Task**: Unsupervised / semi-supervised anomaly detection on grayscale medical images (RSNA, VinCXR, LAG).
- **Key idea**: Learn a generator that reconstructs *normal* anatomy very well. At test time, use reconstruction error between input and output as the anomaly score.
- **Architecture**:
  - Generator `G`: patch-based U-Net style with residual blocks, binary memory/position codes, and attention gates.
  - Discriminator `D`: deep PatchGAN critic with WGAN-GP objective.
- **Training signals**:
  - WGAN-GP adversarial loss.
  - Identity loss on clean normals.
  - MSE loss forcing removal of synthetic anomalies.

If the interviewer asks, you can say: *“We train a GAN where the generator learns to turn possibly abnormal images into normal-looking ones, and anomalies show up as large reconstruction errors.”*

---

## 1. Data Pipeline and Preprocessing

Source: data_loader.py

### 1.1 Datasets and file structure

- Datasets supported: `rsna`, `vincxr`, `lag`.
- Root path in code: `/data/your dataset path`, then `RSNA/`, `LAG/`, `VinCXR/`.
- Each dataset folder contains:
  - `images/`: all images (grayscale, single-channel).
  - `data.json`: splits and labels.

`data.json` (conceptually):

- `train`:
  - `"0"`: labeled normal training images (file names).
  - `"unlabeled"["0"]`: unlabeled normal.
  - `"unlabeled"["1"]`: unlabeled abnormal.
- `test`:
  - `"0"`: test normals.
  - `"1"`: test abnormals.

### 1.2 Transforms and image ranges

- Transform pipeline (both train and test):
  - `Resize((image_size, image_size))` (default `image_size=64`).
  - `ToTensor()` → tensor in `[0, 1]`.
  - `Normalize((0.5,), (0.5,))` → tensor in `[-1, 1]`.
- So a **single image** has shape `(C, H, W) = (1, 64, 64)`.

### 1.3 Training dataset: dataset_train

Class: `dataset_train(main_path, img_size=64, transform=None, mode='train', ar=0.6)`.

Construction:

- `datasetB`: **labeled normals** (target domain).
  - `train_normal = data["train"]["0"]`.
  - All `train_normal` images are loaded into memory via `parallel_load` and stored in `self.datasetB`.
  - Let `N = len(train_normal)`; `self.num_images = N`.

- `datasetA`: **unlabeled mix** of normals and abnormals.
  - `unlabeled_normal_l = data["train"]["unlabeled"]["0"]`.
  - `unlabeled_abnormal_l = data["train"]["unlabeled"]["1"]`.
  - Compute counts using `ar` (default 0.6):
    - `abnormal_num = int(N * ar)` (≈ 60% of N).
    - `normal_num = N - abnormal_num`.
  - Build list: `train_unlabeled_l = unlabeled_abnormal_l[:abnormal_num] + unlabeled_normal_l[:normal_num]`.
  - Load these into `self.datasetA` using the same transform.

For each **index i**:

- `imgB = datasetB[i]` → clean labeled normal.
- `imgA = datasetA[i]` → unlabeled (could be normal or abnormal).
- Both are converted by `transform` into tensors with shape `(1, 64, 64)`.
- Then synthetic anomaly is generated from `imgB`:
  - `anomaly_img, mask = generate_anomaly(imgB, index, core_percent=0.6)`.
  - `anomaly_img`: image with a pasted patch region.
  - `mask`: binary tensor of anomaly region, shape `(64, 64)`, then unsqueezed to `(1, 64, 64)`.

Return from `__getitem__`:

- `imgA`: `(1, 64, 64)` — unlabeled.
- `imgB`: `(1, 64, 64)` — clean normal.
- `anomaly_img`: `(1, 64, 64)` — synthetic anomaly.
- `mask`: `(1, 64, 64)` — anomaly region.

### 1.4 Synthetic anomaly generation

Function: `generate_anomaly(self, image, index, core_percent=0.8)`.

- Start from `image` (a normal) of shape `(1, H, W)` (H=W=64).
- Compute a **center core region**:
  - `dims = (H, W)`.
  - `core = core_percent * dims` → e.g., `0.8 * 64 = 51.2` → ~51×51.
  - `offset = (1 - core_percent) * dims / 2` → ensures the patch lies in the central region.
- Choose **random patch size**:
  - `min_width ≈ 0.05 * W` → about 3 pixels.
  - `max_width ≈ 0.2 * W` → about 13 pixels.
  - Sample `patch_center` within the core, and `patch_width` between min and max.
  - Compute coordinates `[coor_min, coor_max]`, clipped to image bounds.
- Create a **mixing mask**:
  - `alpha = torch.rand(1)` between 0 and 1.
  - `mask` of shape `(H, W)` initialized to 0.
  - `mask[patch_region] = alpha`.
  - `mask_inv = 1 - mask`.
- Choose a **different source image** from `datasetB` (ensure `index` is not re-used) and apply transform.
- Blend images:
  - `image_synthesis = mask_inv * image + mask * anomaly_source`.
- Return:
  - `image_synthesis`: `(1, 64, 64)` synthetic anomaly.
  - `(mask > 0).long()`: `(64, 64)` binary mask.

This ensures anomalies are local and live inside the central core, to simulate lesion-like structures.

### 1.5 Test dataset: dataset_test

- Loads lists `data["test"]["0"]` (normals) and `data["test"]["1"]` (abnormals).
- All images go through the same transform to `(1, img_size, img_size)`.
- For each index:
  - `imgA`: `(1, 64, 64)`.
  - `img_id`: base filename.
  - `label`: 0 (normal) or 1 (abnormal).

---

## 2. Patch Extraction and Reconstruction

Source: solver.py, functions `make_window` and `window_reverse`.

### 2.1 Patch extraction: make_window

Signature: `make_window(x, window_size, stride=1, padding=0)`.

- Input shape: `x` is `(B, W, H, C)`.
- Internally, permutes to `(B, C, W, H)` and uses `F.unfold`.
- `window_size` is the kernel size for `unfold`.
- Output shape:
  - Windows: `(B * N_windows, window_size, window_size, C)`.
  - In our training setup:
    - Full image: `(B, 1, 64, 64)`.
    - We permute to `(B, 64, 64, 1)` then call `make_window(..., window_size=64//num_patch=32)`.
    - We get 4 windows per image (2×2) each of size `(1, 32, 32)`.
    - Final shape fed to G: `(B * 4, 1, 32, 32)`.

### 2.2 Patch reconstruction: window_reverse

Signature: `window_reverse(windows, window_size, H, W)`.

- Input: `windows` of shape `(B * N_windows, window_size, window_size, C)`.
- Internally:
  - Reshapes to `(B, H//window_size, W//window_size, window_size, window_size, C)`.
  - Permutes to reconstruct spatial layout.
- Output: `(B, H, W, C)`.
- In usage:
  - G outputs `(B * 4, 1, 32, 32)`.
  - Permute to `(B * 4, 32, 32, 1)`.
  - Call `window_reverse(..., window_size=32, H=64, W=64)`.
  - Get `(B, 64, 64, 1)`, then permute back to `(B, 1, 64, 64)`.

So throughout training and testing, **G operates on 32×32 patches**, but we reconstruct full 64×64 images for D and for scoring.

---

## 3. Generator Architecture and Dimensions

Source: model.py, class `Generator`.

Constructor (as used in Solver):

- `Generator(conv_dim=64, c_dim=0, repeat_num=6, num_memory=4)`.

### 3.1 Input

- From solver, after windowing, input to `G` is `x` with shape `(B * num_memory, 1, 32, 32)`:
  - `B` = original batch size.
  - `num_memory = 4` (2×2 windows).

### 3.2 Initial conv block (layer1)

- `Conv2d(1 + c_dim, 64, kernel_size=7, stride=1, padding=3)`.
  - Here `c_dim=0`, so it is `1 → 64` channels.
- `InstanceNorm2d(64)`, `ReLU`.
- Output `y`: `(B * 4, 64, 32, 32)`.

### 3.3 Encoder down-sampling with position conditioning

#### Enc_layer1

- `Conv2d(64 → 128, kernel_size=4, stride=2, padding=1)`.
- `InstanceNorm2d(128)`, `ReLU`.
- Output `y1` before conditioning: `(B * 4, 128, 16, 16)`.

Position conditioning (`add_condition` + `p_bn1`):

- Reshape `y1` to `(B, num_memory, C, H, W)` = `(B, 4, 128, 16, 16)`.
- Binary encoding vector:
  - `self.binary_encoding` size: `(4, num_bits)` with `num_bits = int(log2(num_memory)) + 1 = 3`.
  - Example codes (conceptually): 0→000, 1→001, 2→010, 3→011.
- Expand to match spatial dims: `(B, 4, 3, 16, 16)`.
- Concatenate along channels: `(B, 4, 128+3=131, 16, 16)`.
- Flatten back to `(B * 4, 131, 16, 16)`.
- `p_bn1` block: `Conv2d(131 → 128, kernel_size=1) → IN → ReLU`.
- Final `y1`: `(B * 4, 128, 16, 16)`.
- Store `y1` in skip list: `sc[0]`.

#### Enc_layer2

- `Conv2d(128 → 256, kernel_size=4, stride=2, padding=1) → IN → ReLU`.
- Output `y2` before conditioning: `(B * 4, 256, 8, 8)`.

Condition similarly:

- Reshape to `(B, 4, 256, 8, 8)`.
- Append 3-bit code: `(B, 4, 259, 8, 8)`.
- Flatten to `(B * 4, 259, 8, 8)`.
- `p_bn2`: `Conv2d(259 → 256, kernel_size=1) → IN → ReLU`.
- Final `y2`: `(B * 4, 256, 8, 8)`.
- Store `y2` in skip list: `sc[1]`.

#### Enc_layer3

- `Conv2d(256 → 512, kernel_size=4, stride=2, padding=1) → IN → ReLU`.
- Output `y3`: `(B * 4, 512, 4, 4)`.
- Store `y3` in skip list: `sc[2]`.

At this point, per patch:

- Level 0 (after stem): `(B * 4, 64, 32, 32)`.
- Level 1 (enc_layer1 + conditioning): `(B * 4, 128, 16, 16)`.
- Level 2 (enc_layer2 + conditioning): `(B * 4, 256, 8, 8)`.
- Bottleneck input (enc_layer3): `(B * 4, 512, 4, 4)`.

### 3.4 Bottleneck with position conditioning and residual blocks

- Take `y3` of shape `(B * 4, 512, 4, 4)`.
- Reshape to `(B, 4, 512, 4, 4)`.
- Append binary code (3 bits): channels become `512 + 3 = 515`.
- Flatten to `(B * 4, 515, 4, 4)`.

`self.bn2`:

- `Conv2d(515 → 512, kernel_size=1) → IN → ReLU`.
- Followed by `repeat_num = 6` residual blocks, each:
  - `Conv2d(512 → 512, k=3, s=1, p=1) → IN → ReLU`.
  - `Conv2d(512 → 512, k=3, s=1, p=1) → IN`.
  - Add skip connection.

- Output `embedding`: `(B * 4, 512, 4, 4)`.

### 3.5 Decoder with attention gates and multi-scale upsampling

We now decode from `embedding` back up to 32×32 per patch, using attention gates and skip connections.

#### Step 1: Top-level attention and upsample (4×4 → 8×8)

- `gate1` takes `gate=embedding` and `skip_connection=embedding` (self-attention at top level).
  - Both inputs: `(B * 4, 512, 4, 4)`.
  - Output `sc1`: `(B * 4, 512, 4, 4)`.
- Concatenate: `out1 = cat([embedding, sc1], dim=1)` → `(B * 4, 1024, 4, 4)`.
- `dec_bn1` (ResidualBlock1):
  - Internally does: `Conv2d(1024 → 512) → IN → ReLU`, then `Conv2d(512 → 512) → IN`, and adds to the first conv output.
  - Output: `(B * 4, 512, 4, 4)`.
- `dec_layer1`:
  - `ConvTranspose2d(512 → 256, kernel_size=4, stride=2, padding=1) → IN → ReLU`.
  - Spatial: 4×4 → 8×8.
  - Output `out1`: `(B * 4, 256, 8, 8)`.

#### Step 2: Mid-level attention (8×8) and upsample (8×8 → 16×16)

- `gate2` takes:
  - `gate=out1` with shape `(B * 4, 256, 8, 8)`.
  - `skip_connection = sc[-1-1] = sc[1]` (enc_layer2 output) with shape `(B * 4, 256, 8, 8)`.
  - Output `sc2`: `(B * 4, 256, 8, 8)`.
- Concatenate: `out2 = cat([out1, sc2], dim=1)` → `(B * 4, 512, 8, 8)`.
- `dec_bn2` (ResidualBlock1):
  - 512 → 256 channels; output: `(B * 4, 256, 8, 8)`.
- `dec_layer2`:
  - `ConvTranspose2d(256 → 128, kernel_size=4, stride=2, padding=1) → IN → ReLU`.
  - Spatial: 8×8 → 16×16.
  - Output `out2`: `(B * 4, 128, 16, 16)`.

#### Step 3: Lowest-level attention (16×16) and upsample (16×16 → 32×32)

- `gate3` takes:
  - `gate=out2` with shape `(B * 4, 128, 16, 16)`.
  - `skip_connection=sc[-1-2] = sc[0]` (enc_layer1 output) shape `(B * 4, 128, 16, 16)`.
  - Output `sc3`: `(B * 4, 128, 16, 16)`.
- Concatenate: `out3 = cat([out2, sc3], dim=1)` → `(B * 4, 256, 16, 16)`.
- `dec_bn3` (ResidualBlock1):
  - 256 → 128 channels; output: `(B * 4, 128, 16, 16)`.
- `dec_layer3`:
  - `ConvTranspose2d(128 → 64, kernel_size=4, stride=2, padding=1) → IN → ReLU`.
  - Spatial: 16×16 → 32×32.
  - Output `out3`: `(B * 4, 64, 32, 32)`.

#### Multi-scale upsampling branches

- `up_conv1` applied to early `out1` (at 8×8, 256 channels):
  - `ConvTranspose2d(256 → 128, kernel_size=4, stride=2, padding=1) → IN → ReLU` → 8×8 → 16×16.
  - `ConvTranspose2d(128 → 64, kernel_size=4, stride=2, padding=1) → IN → ReLU` → 16×16 → 32×32.
  - Output: `(B * 4, 64, 32, 32)`.

- `up_conv2` applied to `out2` (at 16×16, 128 channels):
  - `ConvTranspose2d(128 → 64, kernel_size=4, stride=2, padding=1) → IN → ReLU` → 16×16 → 32×32.
  - Output: `(B * 4, 64, 32, 32)`.

Now we have three 64-channel feature maps at 32×32:

- From `up_conv1`: `(B * 4, 64, 32, 32)`.
- From `up_conv2`: `(B * 4, 64, 32, 32)`.
- From `out3` after `dec_layer3`: `(B * 4, 64, 32, 32)`.

Concatenate along channels:

- `out = cat([up1, up2, out3], dim=1)` → `(B * 4, 192, 32, 32)`.

### 3.6 Final refinement and output

- `bn3` (ResidualBlock1) with `dim_in = 3 * curr_dim = 192`, `dim_out = curr_dim = 64`:
  - Reduces channels from 192 → 64 with residual structure.
  - Output: `(B * 4, 64, 32, 32)`.

- Final `conv`: `Conv2d(64 → 1, kernel_size=7, stride=1, padding=3)`.
  - Output `output`: `(B * 4, 1, 32, 32)`.

### 3.7 Training vs inference behavior (mask + residual)

- During training (`self.training == True`):
  - Build a spatial mask:
    - `mask` shape: `(B * 4, 1, 32, 32)`.
    - Values initialized to 0.95 then passed to `torch.bernoulli`, so ≈95% of locations are 1.
  - Mix: `output = mask * output + (1 - mask) * x`, where `x` is the input patch.
    - Interpretation: randomly keep some pixels as original to stabilize training and encourage robustness.
- Final activation:
  - `fake1 = tanh(output + x)`.
  - So the generator learns **residuals** around the input patch and uses `tanh` to keep values in `[-1, 1]`.

At test time (evaluation mode), the Bernoulli mask part is skipped (because `self.training` is false), but the residual/tanh structure remains.

---

## 4. Attention Gate (AG) Details

Source: `AttentionGate` in model.py.

- Inputs:
  - `gate`: decoder feature (e.g., `(B * 4, 512, 4, 4)` or `(B * 4, 256, 8, 8)`).
  - `skip_connection`: corresponding encoder feature.
- Internal ops:
  - `W_gate`: `Conv2d(F_g → n_coefficients, kernel_size=1)` + `InstanceNorm2d`.
  - `W_x`: `Conv2d(F_l → n_coefficients, kernel_size=1)` + `InstanceNorm2d`.
  - Add them: `psi = ReLU(W_gate(gate) + W_x(skip_connection))`.
  - `psi` passed through another 1×1 conv + IN + Sigmoid to get a **single-channel attention map**.
  - Multiply: `out = skip_connection * psi`.

Effect: The encoder skip features are gated spatially depending on decoder context, so G can focus on informative regions.

---

## 5. Discriminator Architecture and Dimensions

Source: model.py, class `Discriminator`.

Constructor (as used in Solver):

- `Discriminator(image_size=64, conv_dim=64, c_dim=0, repeat_num=6)`.

### 5.1 Layer-by-layer

- Input: image `(B, 1, 64, 64)`.
- First block:
  - `Conv2d(1 → 64, kernel_size=4, stride=2, padding=1)`.
  - `LeakyReLU(0.01)`.
  - Output: `(B, 64, 32, 32)`.

- Then a loop for `i` from 1 to 5 (total `repeat_num-1 = 5` blocks), each:
  - `Conv2d(curr_dim → 2 * curr_dim, kernel_size=4, stride=2, padding=1)`.
  - `LeakyReLU(0.01)`.
  - Doubles channels and halves spatial resolution.

So the sequence is:

- After i=1: `(B, 128, 16, 16)`.
- After i=2: `(B, 256, 8, 8)`.
- After i=3: `(B, 512, 4, 4)`.
- After i=4: `(B, 1024, 2, 2)`.
- After i=5: `(B, 2048, 1, 1)`.

- Final conv:
  - `conv1 = Conv2d(2048 → 1, kernel_size=3, stride=1, padding=1)`.
  - With 1×1 spatial size, the padding keeps it 1×1.
  - Output `out_src`: `(B, 1, 1, 1)`.

Forward returns:

- `h`: `(B, 2048, 1, 1)` (feature map).
- `out_src`: `(B, 1, 1, 1)` (critic score per image).

The discriminator acts as a **Wasserstein critic**, not a probability classifier.

---

## 6. Training Procedure and Losses

Source: solver.py, class `Solver`.

### 6.1 Training configuration

- Hyperparameters (from main.py):
  - `image_size = 64`.
  - `g_conv_dim = 64`, `d_conv_dim = 64`.
  - `g_repeat_num = 6`, `d_repeat_num = 6`.
  - `lambda_gp = 10`, `lambda_id = 1`.
  - `batch_size = 16`.
  - `g_lr = d_lr = 5e-5`.
  - `beta1 = 0.5`, `beta2 = 0.999`.
  - `n_critic = 2` (two D updates per G update).
  - `num_iters`, `num_iters_decay`, `log_step`, `model_save_step`, `lr_update_step`.

### 6.2 Data per iteration

At each iteration in `train()`:

1. Sample a batch from `dataset_train`:
   - `x_realA`: unlabeled images, shape `(B, 1, 64, 64)`.
   - `x_realB`: clean normals, shape `(B, 1, 64, 64)`.
   - `ano_B`: synthetic anomalies, shape `(B, 1, 64, 64)`.
   - `label_B`: anomaly masks (not directly used in loss here).

2. Patchify each of these using `make_window(..., window_size=64//num_patch=32)`:
   - `x_realA_p`: `(B * 4, 1, 32, 32)`.
   - `x_realB_p`: `(B * 4, 1, 32, 32)`.
   - `ano_B_p`: `(B * 4, 1, 32, 32)`.

### 6.3 Discriminator update (WGAN-GP)

1. **Real term**:
   - Forward: `_, out_src = D(x_realB)`.
   - `d_loss_real = - mean(out_src)`.

2. **Fake term**:
   - Get generator output from unlabeled images:
     - `x_fakeA1 = G(x_realA_p, bs)` → patches `(B * 4, 1, 32, 32)`.
     - Reassemble: `x_fakeA1` full image `(B, 1, 64, 64)` via `window_reverse`.
   - Forward: `_, out_src2 = D(x_fakeA1.detach())`.
   - `d_loss_fake = mean(out_src2)`.

3. **Gradient penalty**:
   - Sample `alpha` of shape `(B, 1, 1, 1)`.
   - Interpolate: `x_hat2 = alpha * x_realB + (1 - alpha) * x_fakeA1` with `requires_grad=True`.
   - Forward: `_, out_src2 = D(x_hat2)`.
   - Compute gradient: `dydx = ∂out_src2 / ∂x_hat2`.
   - Flatten per sample, compute L2 norm, and penalty:
     - `d_loss_gp = mean((||dydx||_2 - 1)^2)`.

4. **Total D loss**:
   - `d_loss = d_loss_real + d_loss_fake + lambda_gp * d_loss_gp`.
   - Backprop and update `self.d_optimizer.step()`.

### 6.4 Generator update

Every `n_critic` steps, update G using three losses.

1. **Identity loss (on clean normals)**:
   - Patches: `x_fakeB3 = G(x_realB_p, bs)`.
   - Reassemble to `(B, 1, 64, 64)`.
   - `g_loss_id = mean(|x_realB - x_fakeB3|)`.

2. **Anomaly removal MSE (on synthetic anomalies)**:
   - Patches: `x_fakeaB = G(ano_B_p, bs)`.
   - Reassemble to `(B, 1, 64, 64)`.
   - `g_loss_mse = mean((x_realB - x_fakeaB)^2)`.

3. **Adversarial loss (on unlabeled)**:
   - Patches: `x_fakeB1 = G(x_realA_p, bs)`.
   - Reassemble to `(B, 1, 64, 64)`.
   - Forward: `_, out_src2 = D(x_fakeB1)`.
   - `g_loss_fake = - mean(out_src2)`.

4. **Total G loss**:
   - `g_loss = g_loss_fake + lambda_id * g_loss_id + g_loss_mse`.
   - Backprop and update `self.g_optimizer.step()`.

Intuition you can state:

> Discriminator learns to distinguish real clean normals from generator outputs, while the generator is trained to (1) fool the discriminator on unlabeled images, (2) leave clean normals unchanged, and (3) map synthetic anomalies back to clean normals.

### 6.5 Checkpointing and validation

- Every `model_save_step` iterations:
  - Save `G` and `D` weights as `iter-G.ckpt`, `iter-D.ckpt`.
- Every `log_step` iterations:
  - Call `test_single_model(iter)` to compute AUC and AP.
  - Save best generator checkpoint as `best_G1.ckpt` when AUC and AP improve.
- During final phase, learning rates are decayed linearly using `update_lr`.

---

## 7. Evaluation and Anomaly Scoring

Source: methods `test_single_model` and `test` in solver.py.

### 7.1 Image-level anomaly score

For each test image:

1. Load `x_realA` (shape `(1, 1, 64, 64)`), label, and file name.
2. Build patches: `x_realA_p` using `make_window` → `(4, 1, 32, 32)`.
3. Run generator: `fake = G(x_realA_p)`.
4. Reassemble patches to full image: `fake` → `(1, 1, 64, 64)`.
5. Compute **difference map**:
   - `diff = |x_realA - fake|`.
   - `diff /= 2` (since images in `[-1, 1]`).
6. Convert `diff` to numpy and take spatial mean:
   - `meanp = mean(diff, axis=(1, 2, 3))` → scalar per image.

This scalar `meanp` is the anomaly score: higher means more abnormal.

### 7.2 Aggregation across slices and thresholding

- They may have multiple slices per patient / image ID; they accumulate and average `meanp` per `imgid`.
- Once they have arrays `gt` (ground truth labels) and `meanp` (scores), they:
  - Use `Find_Optimal_Cutoff(gt, meanp)` to pick a decision threshold (close to Youden index, where TPR ≈ 1 − FPR).
  - Compute:
    - ROC AUC via `roc_auc_score(gt, meanp)`.
    - Average precision (AP) via `average_precision_score(gt, meanp)`.

### 7.3 Optional visualizations

- (Commented code) builds anomaly maps and overlay heatmaps:
  - Upsample patch-wise differences to full size.
  - Normalize to [0, 255].
  - Use OpenCV colormaps to overlay on original images.

You can mention that the implementation supports Grad-CAM-style visualizations for interpretability, even if not enabled by default.

---

## 8. Key Hyperparameters and Their Roles

- `image_size = 64`: final resolution of images used in training and testing.
- `num_patch = 2`: number of patches per side; `num_memory = 4` total patches.
- `g_conv_dim = d_conv_dim = 64`: base channel size; encoder/decoder and D features scale from here.
- `g_repeat_num = 6`: number of residual blocks at bottleneck.
- `d_repeat_num = 6`: number of strided conv layers in D.
- `lambda_gp = 10`: strength of gradient penalty.
- `lambda_id = 1`: weight on identity loss.
- `batch_size = 16`: mini-batch size.
- `g_lr = d_lr = 5e-5`: learning rates for G and D.

If they ask how you tuned these, you can say the values follow common WGAN-GP settings and are balanced so that G and D improve at similar speeds.

---

## 9. Command-line Interface and Scripts

Source: main.py, train.sh, test.sh.

- Entry point: `python main.py`.
- Main arguments (defaults in code):
  - `--dataset {rsna,vincxr,lag}`.
  - `--mode {train,test}`.
  - `--image_size` (default 64).
  - `--g_conv_dim`, `--d_conv_dim`, `--g_repeat_num`, `--d_repeat_num`.
  - `--g_lr`, `--d_lr`, `--lambda_gp`, `--lambda_id`, `--n_critic`.
  - `--num_iters`, `--num_iters_decay`, `--test_iters`.
  - `--device` (GPU id).
- Example commands:

```bash
# Train
python main.py --dataset rsna --mode train --image_size 64 --device 0

# Test a specific checkpoint
python main.py --dataset rsna --mode test --test_iters 100000 --image_size 64 --device 0
```

`train.sh` and `test.sh` wrap these with preset parameters for convenience.

---

## 10. Common Interview Questions and Answers

**Q: Why process images as 2×2 windows instead of full 64×64 directly?**  
A: Windowing increases the effective batch size and allows the generator to attend to local structures while the binary memory code still tells it where each patch comes from, so reassembly preserves global consistency.

**Q: Why add a binary memory/position code?**  
A: Without position information, identical patches from different spatial locations are indistinguishable. The binary code explicitly encodes patch index so the network can learn different features per region (e.g., brain center vs. periphery) even though they share weights.

**Q: Why a residual output with tanh instead of directly predicting the full image?**  
A: Predicting residuals is easier when we expect the output to be close to input (for normal regions). The residual is usually small, so training is more stable and the network focuses on correcting anomalies.

**Q: Why both identity and anomaly MSE losses?**  
A: Identity loss ensures that on normal images the generator behaves like an autoencoder that preserves information. Anomaly MSE uses synthetic anomalies where we know the clean target; this explicitly teaches the generator to remove lesion-like patterns and reconstruct normal tissue.

**Q: Why WGAN-GP instead of vanilla GAN with BCE loss?**  
A: WGAN-GP provides a smoother and more stable training signal, especially important with high-resolution textures and limited labeled data. The gradient penalty enforces Lipschitz continuity, preventing mode collapse and exploding gradients.

**Q: How do you convert pixel-wise differences into a final decision?**  
A: We compute the average per-pixel absolute reconstruction error per image to get a scalar score. Then we select a threshold using ROC analysis (Youden index style) and measure AUC and AP. High scores indicate high anomaly likelihood.

---

## 11. One-minute Narrative (You Can Read Out Loud)

> We work with single-channel medical images, normalize them to [-1, 1], and split each image into a 2×2 grid of 32×32 patches. Each patch gets a small binary code that encodes its position, and all patches go through a shared generator: a U-Net style encoder–bottleneck–decoder with residual blocks and attention-gated skip connections. The generator predicts a residual correction over each patch and outputs via tanh, so normal regions are mostly preserved while anomalies are pushed toward normal patterns. A PatchGAN discriminator trained with WGAN-GP distinguishes real clean normal images from generator outputs. The generator is trained with three losses: an adversarial loss to fool the discriminator on unlabeled images, an identity loss that keeps clean normals unchanged, and an MSE loss that forces synthetic anomalies back to their original clean normals. At test time, we reconstruct each test image, compute the mean absolute difference between the input and reconstruction as an anomaly score, and set a decision threshold from the ROC curve. High reconstruction error means the input likely contains an anomaly.

This script plus the layer shapes above will let you explain the model confidently from input to final anomaly score.

---

## 12. Normalization and Scaling (with Formulas)

### 12.1 Input normalization (data_loader)

- Raw pixel values: typically in $[0, 255]$.
- `ToTensor()` scales to $[0, 1]$:
  - $x_{[0,1]} = \dfrac{x_{\text{raw}}}{255}$.
- `Normalize((0.5,), (0.5,))` maps to $[-1, 1]$:
  - $x_{[-1,1]} = \dfrac{x_{[0,1]} - 0.5}{0.5} = 2x_{[0,1]} - 1$.

You can say: *"All inputs and outputs of the generator and discriminator live in [-1, 1]."*

### 12.2 Instance normalization (InstanceNorm2d)

- Used everywhere in G and in the attention gates instead of BatchNorm.
- For each **sample** and **channel** $c$, with spatial positions $(i,j)$ over $H\times W$:
  - Mean: $\mu_c = \dfrac{1}{HW} \sum_{i,j} x_c(i,j)$.
  - Variance: $\sigma_c^2 = \dfrac{1}{HW} \sum_{i,j} (x_c(i,j) - \mu_c)^2$.
  - Normalized feature:
    $$\hat{x}_c(i,j) = \frac{x_c(i,j) - \mu_c}{\sqrt{\sigma_c^2 + \varepsilon}}$$
  - Affine re-scale and shift (learned $\gamma_c, \beta_c$):
    $$y_c(i,j) = \gamma_c\, \hat{x}_c(i,j) + \beta_c$$

Key point to say: *"InstanceNorm normalizes per image and per channel over spatial locations, which works better than BatchNorm in this GAN setting."*

### 12.3 Gradient penalty (WGAN-GP)

- For interpolated samples $\hat{x}$ between real and fake:
  - $\hat{x} = \alpha x_{\text{real}} + (1-\alpha) x_{\text{fake}}$, $\alpha \sim \mathcal{U}(0,1)$.
  - Compute gradient of critic $D$ w.r.t. $\hat{x}$: $g = \nabla_{\hat{x}} D(\hat{x})$.
  - Gradient penalty term:
    $$\mathcal{L}_{gp} = \mathbb{E}_{\hat{x}}\big[(\lVert g \rVert_2 - 1)^2\big]$$

This term is multiplied by $\lambda_{gp}=10$ in the D loss.

---

## 13. Loss Functions (Short Formulas)

Let $x_B$ = clean normal, $x_A$ = unlabeled, $x_{\text{ano}}$ = synthetic anomaly, $G(\cdot)$ = generator, $D(\cdot)$ = critic.

### 13.1 Discriminator (critic) loss

- WGAN-GP objective as used in code:
  - $$\mathcal{L}_D = -\mathbb{E}[D(x_B)] + \mathbb{E}[D(G(x_A))] + \lambda_{gp}\, \mathcal{L}_{gp}$$

### 13.2 Generator losses

1. **Adversarial loss** (unlabeled $x_A$):
   - $$\mathcal{L}_{adv} = -\mathbb{E}[D(G(x_A))]$$

2. **Identity loss** (normal $x_B$):
   - $L_1$ distance between input and reconstruction:
   - $$\mathcal{L}_{id} = \lVert x_B - G(x_B) \rVert_1 = \sum_{i,j} \left| x_B(i,j) - G(x_B)(i,j) \right|$$

3. **Anomaly-removal MSE** (synthetic anomaly $x_{\text{ano}}$):
   - Pixel-wise squared error to clean normal:
   - $$\mathcal{L}_{mse} = \lVert x_B - G(x_{\text{ano}}) \rVert_2^2 = \sum_{i,j} \big(x_B(i,j) - G(x_{\text{ano}})(i,j)\big)^2$$

4. **Total generator loss**:
   - $$\mathcal{L}_G = \mathcal{L}_{adv} + \lambda_{id}\, \mathcal{L}_{id} + \mathcal{L}_{mse}$$

### 13.3 Reconstruction and anomaly score

- Residual reconstruction (per patch):
  - Network outputs residual $r$ and adds input patch $x$:
  - $$\hat{x}_{\text{patch}} = \tanh(r + x)$$

- Difference map at test time (full image):
  - $$d(i,j) = \frac{1}{2}\,\big|x(i,j) - \hat{x}(i,j)\big|$$
  - The $\tfrac{1}{2}$ comes from mapping range $[-1,1]$ back toward $[0,1]$.

- Image-level anomaly score (mean absolute error):
  - $$s = \frac{1}{H W} \sum_{i,j} d(i,j)$$

You can say: *"Our anomaly score is simply the mean per-pixel absolute reconstruction error."*

---

## 14. Quick Layer Dimension Cheat Sheet

Use this as a fast reference for shapes (for one image with 1 channel, `image_size=64`, `num_patch=2`).

### 14.1 Generator (per patch)

- Input patch: $(1, 32, 32)$.
- After stem conv: $(64, 32, 32)$.
- Encoder 1: $(128, 16, 16)$.
- Encoder 2: $(256, 8, 8)$.
- Encoder 3 / bottleneck input: $(512, 4, 4)$.
- Bottleneck output (after 6 residual blocks): $(512, 4, 4)$.
- Decoder step 1: $(256, 8, 8)$.
- Decoder step 2: $(128, 16, 16)$.
- Decoder step 3: $(64, 32, 32)$.
- Multi-scale concat before `bn3`: $(192, 32, 32)$.
- After `bn3`: $(64, 32, 32)$.
- Final conv output patch: $(1, 32, 32)$.

### 14.2 Discriminator (full image)

- Input: $(1, 64, 64)$.
- After conv1: $(64, 32, 32)$.
- conv2: $(128, 16, 16)$.
- conv3: $(256, 8, 8)$.
- conv4: $(512, 4, 4)$.
- conv5: $(1024, 2, 2)$.
- conv6: $(2048, 1, 1)$.
- Final conv1: $(1, 1, 1)$ (critic score).

This section is what you can quickly glance at if they specifically ask about “what is the shape here?” or “which normalization and loss functions do you use mathematically?”.

---

## 15. Core Terminology Glossary (Interview Quick-Ref)

Below are the main ML / DL terms that appear in this paper and code, with short definitions and formulas where useful.

### 15.1 Tensor, batch, channels, spatial size

- **Tensor**: multi-dimensional array. Here images are 4D tensors with shape $(B, C, H, W)$:
  - $B$: batch size (number of images processed in parallel).
  - $C$: channels (here 1 for grayscale; intermediate layers have 64, 128, 256, etc.).
  - $H, W$: spatial height and width (64, 32, 16, ...).
- **Batch size**: number of samples processed in one forward/backward pass.
  - Here default `batch_size = 16` full images, which become $B \times 4$ patches (because of $2\times2$ windowing).
- **Channel**: number of feature maps. Example: $(B, 64, 32, 32)$ means 64 learned filters per spatial location.

### 15.2 Patch / window, stride, kernel size

- **Patch / window**: small spatial crop of the image used as input to the generator.
  - In this model: full image $64\times64$ is split into 4 patches of size $32\times32$.
- **Kernel size**: size of the convolution filter.
  - Example: `kernel_size=4` means a $4\times4$ filter; `kernel_size=7` means a $7\times7$ filter.
- **Stride**: step by which the kernel moves.
  - For downsampling, `stride=2` halves spatial resolution.
  - For transposed conv upsampling, `stride=2` doubles spatial resolution.

### 15.3 Convolution and transposed convolution

- **Conv2d**: linear spatial filtering with shared weights over the image.
  - For one input/output channel, output at position $(i,j)$:
    $$y(i,j) = \sum_{u,v} w(u,v)\,x(i+u, j+v) + b$$
- **ConvTranspose2d**: “inverse-like” operation used for learnable upsampling.
  - Increases spatial resolution by the stride (e.g., $4\times4 \to 8\times8$ when `stride=2`).

### 15.4 Activation functions

- **ReLU** (Rectified Linear Unit):
  $$\text{ReLU}(x) = \max(0, x)$$
- **LeakyReLU**: allows a small negative slope (here 0.01):
  $$\text{LeakyReLU}(x) = \begin{cases}x & x \ge 0 \\ 0.01x & x < 0\end{cases}$$
- **Tanh**: squashes to $[-1,1]$:
  $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$
  Used at generator output so images live in same range as normalized inputs.
- **Sigmoid**: squashes to $(0,1)$:
  $$\sigma(x) = \frac{1}{1+e^{-x}}$$
  Used in attention gates to produce masks between 0 and 1.

### 15.5 Normalization layers

- **InstanceNorm2d**: see Section 12.2 for full formula.
  - Normalizes each sample and channel across its spatial dimensions $(H,W)$.
- **Why InstanceNorm instead of BatchNorm?**
  - BatchNorm depends on batch statistics and can be unstable for GANs with small, non-i.i.d. batches.

### 15.6 Residual block

- **Residual connection**: learn a function $F(x)$ and output $x + F(x)$ instead of $F(x)$ alone:
  $$y = x + F(x)$$
- Benefit: easier optimization; the network can default to identity mapping when deeper layers are not needed.

### 15.7 Attention gate

- Learns a **spatial mask** to gate skip-connection features using decoder context.
- Given gate feature $g$ and encoder feature $x$:
  - Compress channels of both, add, apply ReLU and Sigmoid to get mask $\psi$.
  - Apply: $$\text{AG}(g,x) = x \odot \psi$$ where $\odot$ is element-wise multiplication.

### 15.8 Generator vs discriminator (GAN vocabulary)

- **Generator (G)**: maps input images/patches to reconstructed “normal-looking” outputs.
- **Discriminator / critic (D)**: outputs higher scores for real normal images than for generated ones.
- **GAN**: game where G tries to fool D, and D tries to distinguish real vs generated.

### 15.9 Loss-related terms

- **L1 loss**: mean absolute error between prediction and target:
  $$\text{L1}(x,\hat{x}) = \frac{1}{N}\sum_i |x_i - \hat{x}_i|$$
- **L2 loss / MSE**: mean squared error:
  $$\text{MSE}(x,\hat{x}) = \frac{1}{N}\sum_i (x_i - \hat{x}_i)^2$$
- **Adversarial loss**: encourages G to produce outputs that D scores as real.
- **Gradient penalty**: regularizer enforcing $\lVert \nabla_x D(x) \rVert_2 \approx 1$.

### 15.10 Evaluation metrics

- **TPR (True Positive Rate)**: $\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}$.
- **FPR (False Positive Rate)**: $\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$.
- **ROC curve**: TPR vs FPR for varying thresholds.
- **AUC (Area Under ROC Curve)**: scalar summary of ROC between 0 and 1.
- **Precision**: $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$.
- **Recall**: same as TPR.
- **AP (Average Precision)**: area under precision–recall curve.

### 15.11 Training dynamics terms

- **Epoch**: one full pass over the training dataset.
- **Iteration / step**: one optimizer update (one batch forward+backward).
- **Learning rate (LR)**: step size in gradient descent; here $5\times10^{-5}$.
- **LR decay**: schedule that gradually reduces LR to stabilize convergence.

Skim this glossary right before the interview to refresh terms like batch, channels, patches, residuals, attention, normalization, losses, and metrics so you can define them confidently if asked.
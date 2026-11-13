# RUOD enhancement

Underwater image enhancement of the RUOD dataset with
[AutoEnhancer](https://github.com/piggy2009/autoEnhancer) (Tang et al., *AutoEnhancer: Transformer on U-Net
Architecture Search for Underwater Image Enhancement*), used to train object detectors on enhanced images.

The enhanced images keep the original file names and image sizes, so the RUOD (and R-RUOD) COCO annotations
work unchanged: just point the detector's image folder at the enhanced output.

## Contents

| Path | What it is |
|---|---|
| `enhance_ruod.py` | Enhances every image of the RUOD `train/` and `test/` folders |
| `autoEnhancer/` | AutoEnhancer code (git submodule, commit `262dde4`) |
| `pixi.toml`, `pixi.lock` | Python environment (Python 3.10, PyTorch with CUDA 12.8) |
| `enhance_ruod.log` | Log of the full run on RUOD |
| `autoenhancer_examples*.png` | Original vs enhanced examples (train / test, and images with objects RUOD misses) |
| `examples_001397/` | Image 001397, original and enhanced, with its R-RUOD boxes drawn |
| `pseudo_v2_enhanced_vs_original_seed1.png/.pdf` | Detector results on enhanced vs original images |

## Steps

### 1. Clone with the submodule

```bash
git clone --recursive https://github.com/VenuriAmarasinghe/RUOD_enhancement.git
cd RUOD_enhancement
```

If you cloned without `--recursive`:

```bash
git submodule update --init
```

### 2. Download the AutoEnhancer model

The released model is not stored in this repo. Download `200000.pth` from the AutoEnhancer authors' Google
Drive folder (link in their README):
https://drive.google.com/drive/folders/10dJI5Pb5NMYhCg0ffZKlTExut-_PQsVf?usp=sharing

Put it exactly here (the folder name contains spaces and a colon):

```
autoEnhancer/ckpt/WaterEnhance_2022-06-07 08:37:34/200000.pth
```

```bash
mkdir -p "autoEnhancer/ckpt/WaterEnhance_2022-06-07 08:37:34"
mv /path/to/200000.pth "autoEnhancer/ckpt/WaterEnhance_2022-06-07 08:37:34/"
```

### 3. Install the environment

Install [pixi](https://pixi.sh) if you do not have it:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Then, in the repo folder:

```bash
pixi install
```

This needs a Linux machine with an NVIDIA GPU (the PyTorch wheels are for CUDA 12.8; edit the `index` URLs in
`pixi.toml` for another CUDA version).

### 4. Get the RUOD images

Download RUOD and find the image folder that holds `train/` and `test/` (called `RUOD_pic` in the release):

```
RUOD_pic/
  train/   9,800 images
  test/    4,200 images
```

### 5. Check on a few images first

```bash
pixi run python enhance_ruod.py --pic-root /path/to/RUOD_pic --out-root /path/to/RUOD_pic_enhanced_autoenhancer --limit 20
```

Look at the output images. If they look right, run step 6. Images that are already enhanced are skipped, so
the 20 test images do not need deleting.

### 6. Enhance the whole dataset

```bash
pixi run python enhance_ruod.py --pic-root /path/to/RUOD_pic --out-root /path/to/RUOD_pic_enhanced_autoenhancer
```

Always pass `--pic-root` and `--out-root`: the defaults in the script are paths on the author's machine.

Other options:

| Option | Default | Meaning |
|---|---|---|
| `--splits` | `train test` | Sub-folders of `--pic-root` to enhance |
| `--max-pixels` | `1920*1080` | Larger images are enhanced at a reduced size (see below) |
| `--quality` | `95` | JPEG quality of the output |
| `--limit` | `0` (all) | Only the first N images per split |

The run can be stopped and restarted at any time; finished images are skipped.

Output:

```
RUOD_pic_enhanced_autoenhancer/
  train/              enhanced train images, same names and sizes
  test/               enhanced test images
  reduced_size.txt    images not enhanced at full resolution, with their scale
```

### 7. Train / test a detector on the enhanced images

Use the RUOD (or R-RUOD) annotation files as they are and change only the image folder. In an MMDetection
config, for example:

```python
train_dataloader = dict(dataset=dict(data_prefix=dict(img='/path/to/RUOD_pic_enhanced_autoenhancer/train/')))
test_dataloader = dict(dataset=dict(data_prefix=dict(img='/path/to/RUOD_pic_enhanced_autoenhancer/test/')))
```

Test the model on the enhanced test images when it was trained on enhanced train images.

## How the enhancement works

- Same model, searched architecture (`SELECT = [2, 5, 5, 4, 0, 8, 8, 1, 2, 5, 5, 1]`) and inference steps as
  `autoEnhancer/infer_water.py`: RGB + LAB input, reflect padding to a multiple of 8, output clamped to [0, 1].
- Runs in fp16 (within ~73 dB PSNR of fp32, half the GPU memory).
- Images above `--max-pixels` are enhanced at a smaller size and the colour change is moved to the
  full-resolution image: `out = input + upsample(enhanced_small - input_small)`. This keeps the original size
  and detail and avoids colour differences between tiles. If an image still runs out of GPU memory it is
  retried at 0.75x the size. Every such image is listed in `reduced_size.txt`.
- `autoEnhancer`'s `base_SPOS.py` imports a `underwater_model.vit` module that is missing from their repo and
  not used; `enhance_ruod.py` stubs it, so the AutoEnhancer code does not need changing.

## Reference run

On one NVIDIA RTX 5080 (16 GB), all 14,000 RUOD images took about 74 minutes (train 47 min, test 28 min);
2,192 images were enhanced at a reduced size.

## Results

Faster R-CNN with pseudo labels (Pseudo v2), seed 1, best mAP@50 checkpoint (epoch 19), tested on R-RUOD test
labels. The enhanced model is trained and tested on enhanced images.

| Test set | Images | mAP@50 original | mAP@50 enhanced | mAP original | mAP enhanced |
|---|---|---|---|---|---|
| Full | 4,200 | 0.584 | 0.577 | 0.380 | 0.378 |
| ≤ 100 boxes per image | 4,134 | 0.608 | 0.601 | 0.396 | 0.395 |

Enhancement gave no gain in this setting (one seed).

## Credit

AutoEnhancer code and model: https://github.com/piggy2009/autoEnhancer (please cite their paper).
RUOD dataset: Fu et al., *Rethinking general underwater object detection: Datasets, challenges, and solutions*,
Neurocomputing 2023.

#!/usr/bin/env python3
"""
Enhance RUOD images with AutoEnhancer (Tang et al., "AutoEnhancer: Transformer on U-Net Architecture search
for Underwater Image Enhancement"; https://github.com/piggy2009/autoEnhancer), released 200000.pth model and
the same searched architecture / inference steps as autoEnhancer/infer_water.py (RGB + LAB input,
reflect padding to a multiple of 8, output clamped to [0, 1]).

Output keeps the file names and the image size, so the RUOD annotations can be used unchanged:
  <out-root>/<split>/<name>.jpg
Inference runs in fp16 (output within ~73 dB PSNR of fp32, half the memory, ~5 KB of GPU memory per pixel).
Images above --max-pixels (default 1920x1080) are enhanced at a reduced size that fits and the change is
transferred to the full-resolution image: out = input + upsample(enhanced_small - input_small). On 720x405
images this gives 35 dB PSNR to the full-resolution output at scale 0.7, better than overlapping tiles (33 dB,
visible colour differences between tiles, since the network's colour correction is image-wide).
If an image still runs out of GPU memory it is retried at 0.75x the size. Every image not enhanced at full
resolution is listed with its scale in
<out-root>/reduced_size.txt. Already enhanced images are skipped, so the script can be stopped and restarted.
"""
import argparse
import os
import sys
import types

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, 'autoEnhancer')
sys.path.insert(0, REPO)
# base_SPOS.py imports underwater_model.vit.ViT, which is not in the repo and not used: stub it
sys.modules['underwater_model.vit'] = types.SimpleNamespace(ViT=None)
from underwater_model.model_SPOS import Water  # noqa: E402

CKPT = os.path.join(REPO, 'ckpt', 'WaterEnhance_2022-06-07 08:37:34', '200000.pth')
SELECT = [2, 5, 5, 4, 0, 8, 8, 1, 2, 5, 5, 1]  # searched architecture used in infer_water.py
PIC = '/home/n12675148/Venu/QUT/datasets/RUOD/RUOD/RUOD_pic'
OUT = '/home/n12675148/Venu/QUT/datasets/RUOD/RUOD/RUOD_pic_enhanced_autoenhancer'
FACTOR = 8


def to_tensor(a, dev):
    return torch.from_numpy(a).permute(2, 0, 1).float().div(255).unsqueeze(0).to(dev)


@torch.no_grad()
def run(net, rgb, lab):
    """rgb, lab: 1x3xHxW in [0, 1]; returns 1x3xHxW in [0, 1] (as infer_water.py)."""
    h, w = rgb.shape[2:]
    padh, padw = (-h) % FACTOR, (-w) % FACTOR
    if padh or padw:
        rgb = F.pad(rgb, (0, padw, 0, padh), 'reflect')
        lab = F.pad(lab, (0, padw, 0, padh), 'reflect')
    out = net(rgb, lab, SELECT)[0][:, :, :h, :w]
    return out.clamp(0, 1)


def run_reduced(net, rgb, lab_np, scale):
    """Enhance at scale x the size, transfer the change to the full-size image."""
    h, w = rgb.shape[2:]
    hs, ws = max(FACTOR, round(h * scale)), max(FACTOR, round(w * scale))
    small = F.interpolate(rgb, (hs, ws), mode='area')
    lab_small = to_tensor(cv2.resize(lab_np, (ws, hs), interpolation=cv2.INTER_AREA), rgb.device)
    delta = run(net, small, lab_small) - small
    return (rgb + F.interpolate(delta, (h, w), mode='bilinear', align_corners=False)).clamp(0, 1)


def enhance(net, path, dev, max_pixels):
    """Returns the enhanced BGR image and the scale it was enhanced at (1 = full resolution)."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    r = to_tensor(rgb, dev)
    scale = min(1.0, (max_pixels / (rgb.shape[0] * rgb.shape[1])) ** 0.5)
    while True:
        try:
            with torch.autocast('cuda', dtype=torch.float16):
                out = run(net, r, to_tensor(lab, dev)) if scale == 1.0 else run_reduced(net, r, lab, scale)
            break
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            scale *= 0.75
    out = (out[0].float().permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR), scale


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--splits', nargs='+', default=['train', 'test'], help='folders under --pic-root')
    ap.add_argument('--pic-root', default=PIC)
    ap.add_argument('--out-root', default=OUT)
    ap.add_argument('--max-pixels', type=int, default=1920 * 1080,
                    help='larger images are enhanced at a reduced size (needs ~11 GB of free GPU memory at the default)')
    ap.add_argument('--quality', type=int, default=95, help='JPEG quality of the output')
    ap.add_argument('--limit', type=int, default=0, help='only the first N images per split (testing)')
    args = ap.parse_args()

    dev = torch.device('cuda')
    net = Water(dim=48)
    net.load_state_dict(torch.load(CKPT, map_location=dev))
    net.eval().to(dev)

    for split in args.splits:
        src, dst = os.path.join(args.pic_root, split), os.path.join(args.out_root, split)
        os.makedirs(dst, exist_ok=True)
        names = sorted(f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        if args.limit:
            names = names[:args.limit]
        todo = [n for n in names if not os.path.exists(os.path.join(dst, os.path.splitext(n)[0] + '.jpg'))]
        print(f'{split}: {len(names)} images, {len(names) - len(todo)} already done -> {dst}')
        for n in tqdm(todo, desc=split):
            out, scale = enhance(net, os.path.join(src, n), dev, args.max_pixels)
            if scale < 1.0:
                with open(os.path.join(args.out_root, 'reduced_size.txt'), 'a') as f:
                    f.write(f'{split}/{n}\tscale {scale:.3f}\n')
            tmp = os.path.join(dst, '.tmp_' + os.path.splitext(n)[0] + '.jpg')
            cv2.imwrite(tmp, out, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            os.replace(tmp, os.path.join(dst, os.path.splitext(n)[0] + '.jpg'))


if __name__ == '__main__':
    main()

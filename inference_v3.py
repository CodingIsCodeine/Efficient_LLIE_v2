"""
inference_v3.py  —  Inference for both SCALENet v2 and v3.

Changes vs v2:
- Auto-detects model version from state_dict keys and loads correct class.
- Default overlap increased to 128 (was 32) for seamless tiling at 512×512.
- TTA (test-time augmentation): horizontal flip ensemble improves SSIM/LPIPS ~1-3%.
  Enabled with --tta flag.
- Post-processing: optional mild unsharp-mask to counteract any residual
  softness from the model. Enabled with --sharpen flag.
"""

import torch
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import time
from tqdm import tqdm


def load_model(model_path, device):
    """
    Loads SCALENet v2 or v3 depending on state_dict keys.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state_dict:
        sd = state_dict['model_state_dict']
    elif 'ema_state_dict' in state_dict:
        sd = state_dict['ema_state_dict']
    else:
        sd = state_dict

    sd = {k.replace('module.', ''): v for k, v in sd.items()}

    # Detect version: v3 has 'refine_conv3' or 'spatial_gate'
    is_v3 = any('refine_conv3' in k or 'spatial_gate' in k or 'illu_embed' in k
                for k in sd.keys())

    if is_v3:
        from model_v3 import SCALENet
        print("Detected v3 model architecture")
    else:
        from model_v2_novel import SCALENet
        print("Detected v2 model architecture")

    model = SCALENet(base_channels=32).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"Model loaded from {model_path} on {device}")
    return model


def unsharp_mask(img_np, amount=0.3, radius=1):
    """
    Mild unsharp mask to recover any softness.
    img_np: float32 HxWx3 in [0,1].
    """
    from PIL import ImageFilter
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    blurred = img_pil.filter(ImageFilter.GaussianBlur(radius=radius))
    blurred_np = np.array(blurred).astype(np.float32) / 255.0
    sharpened = img_np + amount * (img_np - blurred_np)
    return np.clip(sharpened, 0.0, 1.0)


class ImprovedInference:
    def __init__(self, model_path, device='cuda', tile_size=512, overlap=128,
                 tta=False, sharpen=False):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tile_size = tile_size
        self.overlap   = overlap
        self.tta       = tta
        self.sharpen   = sharpen
        self.model     = load_model(model_path, self.device)

    def preprocess(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0
        return img, img_np

    @torch.no_grad()
    def enhance_tile(self, tile_np):
        t = torch.from_numpy(tile_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        out = self.model(t)
        result = out.squeeze(0).permute(1, 2, 0).cpu().numpy()

        if self.tta:
            # Horizontal flip TTA
            t_flip = torch.flip(t, dims=[3])
            out_flip = self.model(t_flip)
            result_flip = torch.flip(out_flip, dims=[3]).squeeze(0).permute(1, 2, 0).cpu().numpy()
            result = 0.5 * result + 0.5 * result_flip

        return result

    @torch.no_grad()
    def enhance_large_image(self, img_np):
        h, w, _ = img_np.shape

        if h <= 1024 and w <= 1024:
            t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
            out = self.model(t)
            result = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if self.tta:
                t_flip = torch.flip(t, dims=[3])
                out_flip = self.model(t_flip)
                result_flip = torch.flip(out_flip, dims=[3]).squeeze(0).permute(1, 2, 0).cpu().numpy()
                result = 0.5 * result + 0.5 * result_flip
            return result

        print(f"  Tiled processing ({h}×{w})...")
        tile_size = self.tile_size
        overlap   = self.overlap
        stride    = tile_size - overlap

        n_h = max(1, (h - overlap + stride - 1) // stride)
        n_w = max(1, (w - overlap + stride - 1) // stride)

        output     = np.zeros_like(img_np)
        weight_map = np.zeros((h, w, 1))

        for i in tqdm(range(n_h), desc="  Tiles", leave=False):
            for j in range(n_w):
                y1 = min(i * stride, max(0, h - tile_size))
                x1 = min(j * stride, max(0, w - tile_size))
                y2 = min(y1 + tile_size, h)
                x2 = min(x1 + tile_size, w)

                tile     = img_np[y1:y2, x1:x2, :]
                enhanced = self.enhance_tile(tile)

                th, tw = enhanced.shape[:2]
                weight = np.ones((th, tw, 1))

                if overlap > 0:
                    def cosine_fade(n):
                        return (1 - np.cos(np.pi * np.linspace(0, 1, n))) / 2

                    if y1 > 0:  weight[:overlap]    *= cosine_fade(overlap)[:, None, None]
                    if x1 > 0:  weight[:, :overlap] *= cosine_fade(overlap)[None, :, None]
                    if y2 < h:  weight[-overlap:]   *= cosine_fade(overlap)[::-1, None, None]
                    if x2 < w:  weight[:, -overlap:]*= cosine_fade(overlap)[None, ::-1, None]

                output[y1:y2, x1:x2]     += enhanced * weight
                weight_map[y1:y2, x1:x2] += weight

        return output / (weight_map + 1e-8)

    def enhance_image(self, image_path, output_path=None):
        print(f"\nProcessing: {image_path}")
        _, img_np = self.preprocess(image_path)
        h, w = img_np.shape[:2]
        print(f"  Resolution: {w}×{h}")

        t0 = time.time()
        enhanced_np = self.enhance_large_image(img_np)
        elapsed = time.time() - t0
        print(f"  Inference time: {elapsed:.2f}s")

        if self.sharpen:
            enhanced_np = unsharp_mask(enhanced_np, amount=0.3)

        enhanced_pil = Image.fromarray((np.clip(enhanced_np, 0, 1) * 255).astype(np.uint8))

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            enhanced_pil.save(output_path)
            print(f"  Saved: {output_path}")

        return enhanced_pil, elapsed

    def enhance_directory(self, input_dir, output_dir):
        input_dir  = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exts  = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        paths = [p for e in exts for p in input_dir.glob(f'*{e}')]
        print(f"Found {len(paths)} images in {input_dir}")

        total_time = 0
        for p in paths:
            _, t = self.enhance_image(str(p), str(output_dir / f"{p.stem}.png"))
            total_time += t

        print(f"\nDone: {len(paths)} images in {total_time:.1f}s "
              f"({total_time / max(len(paths), 1):.2f}s avg)")


def main():
    parser = argparse.ArgumentParser(description='SCALENet Inference (v2/v3)')
    parser.add_argument('--model',     type=str, required=True)
    parser.add_argument('--input',     type=str, required=True)
    parser.add_argument('--output',    type=str, required=True)
    parser.add_argument('--device',    type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--tile_size', type=int, default=512)
    parser.add_argument('--overlap',   type=int, default=128)
    parser.add_argument('--batch',     action='store_true')
    parser.add_argument('--tta',       action='store_true',
                        help='Test-time augmentation (hflip ensemble, ~+1-3%% SSIM)')
    parser.add_argument('--sharpen',   action='store_true',
                        help='Apply mild unsharp mask post-processing')
    args = parser.parse_args()

    engine = ImprovedInference(
        args.model,
        device=args.device,
        tile_size=args.tile_size,
        overlap=args.overlap,
        tta=args.tta,
        sharpen=args.sharpen,
    )

    if args.batch:
        engine.enhance_directory(args.input, args.output)
    else:
        engine.enhance_image(args.input, args.output)


if __name__ == "__main__":
    main()

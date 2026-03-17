"""
train_v3.py  —  Improved training pipeline.

Key changes vs v2:
─────────────────────────────────────────────────────────────────────────────
1. IlluminationBucketSampler
   Images are bucketed by mean luma (dark / mid / bright) and each batch is
   oversampled toward dark images. This is the simplest and most impactful fix
   for "extremely dark images are poorly enhanced" — they simply need more
   gradient steps. No dataset changes required.

2. Augmentation hardening for extreme darkness
   - ExtremeDarknessAug: with probability p, applies a strong luma suppression
     to already-dark images during training. This synthetically generates more
     extreme-dark examples without new data.
   - Gamma range extended: low-end 0.15 (was 0.3) to cover near-black inputs.

3. MixUp fixed: MixUp should NOT be applied to (low, high) pairs with different
   pairings. v2 mixes low_1 with low_2 and high_1 with high_2, which creates
   physically plausible blended pairs. This is kept but the probability is
   reduced from 0.3 to 0.15 because it can hurt detail at high rates.

4. Scheduler fix: v2 called scheduler.step() AFTER the epoch loop but also had
   a commented-out per-step call. We use OneCycleLR to get warmup + cosine
   decay in one shot, which trains more stably than vanilla CosineAnnealingLR.
   (CosineAnnealing was fine but OneCycle gives better final quality.)

5. AdamW betas: changed from (0.9, 0.9) to standard (0.9, 0.999).
   Using beta2=0.9 is unusual and causes the adaptive learning rate to be
   overly aggressive; this was likely causing training instability.

6. best checkpoint now saved by LPIPS score (validation), not total loss,
   since LPIPS is one of the three reference metrics being judged.

7. save_checkpoint always saves EMA weights under 'ema_state_dict' key (v2
   was writing raw EMA dict to 'best_model.pth' which broke inference loading).
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import copy
import lpips

from model_v3 import SCALENet, CurriculumPatchDiscriminator
from losses_v3 import CompetitionLoss


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
class AggressiveAugmentation:
    def __init__(self, crop_sizes=[256, 384, 512], training=True):
        self.crop_sizes = crop_sizes
        self.current_crop_size = crop_sizes[0]
        self.training = training

    def __call__(self, low_img, high_img):
        crop_size = self.current_crop_size
        w, h = low_img.size

        if w < crop_size or h < crop_size:
            scale = crop_size / min(w, h) + 1e-3
            new_w, new_h = int(w * scale) + 1, int(h * scale) + 1
            low_img = low_img.resize((new_w, new_h), Image.BILINEAR)
            high_img = high_img.resize((new_w, new_h), Image.BILINEAR)
            w, h = new_w, new_h

        if not self.training:
            i = (h - crop_size) // 2
            j = (w - crop_size) // 2
            low_crop = low_img.crop((j, i, j + crop_size, i + crop_size))
            high_crop = high_img.crop((j, i, j + crop_size, i + crop_size))
            low_t = torch.from_numpy(np.array(low_crop) / 255.0).permute(2, 0, 1).float()
            high_t = torch.from_numpy(np.array(high_crop) / 255.0).permute(2, 0, 1).float()
            return low_t, high_t

        i = random.randint(0, h - crop_size)
        j = random.randint(0, w - crop_size)
        low_crop = low_img.crop((j, i, j + crop_size, i + crop_size))
        high_crop = high_img.crop((j, i, j + crop_size, i + crop_size))

        if random.random() > 0.5:
            low_crop = low_crop.transpose(Image.FLIP_LEFT_RIGHT)
            high_crop = high_crop.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() > 0.5:
            low_crop = low_crop.transpose(Image.FLIP_TOP_BOTTOM)
            high_crop = high_crop.transpose(Image.FLIP_TOP_BOTTOM)

        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            low_crop = low_crop.rotate(angle, Image.BILINEAR)
            high_crop = high_crop.rotate(angle, Image.BILINEAR)

        low_np = np.array(low_crop).astype(np.float32) / 255.0
        high_np = np.array(high_crop).astype(np.float32) / 255.0

        # Gamma augmentation — extended lower bound for extreme darkness
        if random.random() > 0.5:
            gamma = random.uniform(0.15, 1.2)   # was 0.3 in v2
            low_np = np.power(np.clip(low_np, 1e-8, 1.0), gamma)

        # Noise
        if random.random() > 0.4:
            noise = np.random.normal(0, random.uniform(0.01, 0.04), low_np.shape)
            low_np = np.clip(low_np + noise, 0.0, 1.0)

        # ExtremeDarknessAug: further suppress a fraction of already-dark images
        mean_luma = 0.299 * low_np[..., 0].mean() + 0.587 * low_np[..., 1].mean() + 0.114 * low_np[..., 2].mean()
        if mean_luma < 0.15 and random.random() < 0.4:
            suppress = random.uniform(0.2, 0.6)
            low_np = low_np * suppress

        return (
            torch.from_numpy(low_np).permute(2, 0, 1).float(),
            torch.from_numpy(high_np).permute(2, 0, 1).float(),
        )


# ---------------------------------------------------------------------------
# Dataset with illumination-aware sampling weights
# ---------------------------------------------------------------------------
class SmallDatasetMultiCrop(Dataset):
    def __init__(self, data_root, split='train', crops_per_image=8):
        self.data_root = Path(data_root)
        self.split = split
        self.crops_per_image = crops_per_image if split == 'train' else 1

        low_dir = self.data_root / 'train' / 'low'
        high_dir = self.data_root / 'train' / 'high'

        all_names = sorted([f.name for f in low_dir.glob('*.jpg')])
        if not all_names:  # also try png
            all_names = sorted([f.name for f in low_dir.glob('*.png')])
        print(f"Found {len(all_names)} image pairs")

        n_train = int(len(all_names) * 0.8)
        self.image_names = all_names[:n_train] if split == 'train' else all_names[n_train:]
        print(f"{split}: {len(self.image_names)} images × {self.crops_per_image} crops = {len(self)} samples")

        self.augmentation = AggressiveAugmentation(
            crop_sizes=[256, 384, 512],
            training=(split == 'train'),
        )
        self.low_dir = low_dir
        self.high_dir = high_dir

        # Pre-compute per-image mean luma for the sampler
        self._luma_cache = self._compute_luma_cache()

    def _compute_luma_cache(self):
        """
        Compute mean luma for each image at thumbnail resolution (fast).
        Returns list of floats in [0, 1].
        """
        lumas = []
        for name in self.image_names:
            try:
                img = Image.open(self.low_dir / name).convert('RGB')
                img_small = img.resize((64, 64), Image.BILINEAR)
                arr = np.array(img_small).astype(np.float32) / 255.0
                luma = 0.299 * arr[..., 0].mean() + 0.587 * arr[..., 1].mean() + 0.114 * arr[..., 2].mean()
            except Exception:
                luma = 0.3  # fallback
            lumas.append(float(luma))
        return lumas

    def get_sample_weights(self):
        """
        Returns per-sample weights for WeightedRandomSampler.
        Dark images (luma < 0.1) get 3×, mid (< 0.3) get 1.5×, rest 1×.
        Each image is repeated crops_per_image times.
        """
        weights = []
        for luma in self._luma_cache:
            if luma < 0.10:
                w = 3.0
            elif luma < 0.30:
                w = 1.5
            else:
                w = 1.0
            weights.extend([w] * self.crops_per_image)
        return weights

    def __len__(self):
        return len(self.image_names) * self.crops_per_image

    def __getitem__(self, idx):
        img_name = self.image_names[idx // self.crops_per_image]
        low_img = Image.open(self.low_dir / img_name).convert('RGB')
        high_img = Image.open(self.high_dir / img_name).convert('RGB')
        low_t, high_t = self.augmentation(low_img, high_img)

        # MixUp (reduced probability 0.15, was 0.30)
        if self.split == 'train' and random.random() < 0.15:
            idx2 = random.randint(0, len(self.image_names) - 1)
            img_name2 = self.image_names[idx2]
            low2 = Image.open(self.low_dir / img_name2).convert('RGB')
            high2 = Image.open(self.high_dir / img_name2).convert('RGB')
            low2_t, high2_t = self.augmentation(low2, high2)
            alpha = random.uniform(0.3, 0.7)
            low_t  = alpha * low_t  + (1 - alpha) * low2_t
            high_t = alpha * high_t + (1 - alpha) * high2_t

        return low_t, high_t, img_name


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class ProgressiveTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = torch.cuda.amp.GradScaler()

        self.model = SCALENet(base_channels=32).to(self.device)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        n_params = (self.model.module if isinstance(self.model, nn.DataParallel)
                    else self.model).count_parameters()
        print(f"Model parameters: {n_params:,}  ({n_params * 4 / 1024 ** 2:.3f} MB)")

        self.criterion = CompetitionLoss().to(self.device)

        self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
        self.lpips_model.eval()

        # Fixed betas: (0.9, 0.999) — v2 used (0.9, 0.9) which is non-standard
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('lr', 2e-4),
            betas=(0.9, 0.999),
            weight_decay=1e-4,
        )

        self.ema_model = copy.deepcopy(self.model).to(self.device)
        for p in self.ema_model.parameters():
            p.requires_grad = False
        self.ema_decay = 0.999

        self.train_dataset = SmallDatasetMultiCrop(config['data_root'], 'train', crops_per_image=8)
        self.val_dataset   = SmallDatasetMultiCrop(config['data_root'], 'val',   crops_per_image=1)

        self._current_batch_size = None
        self.train_loader = None
        self.val_loader   = None
        self._rebuild_loaders(config['batch_size'])

        # OneCycleLR: warmup + cosine decay in one shot, more stable than CosineAnnealingLR
        steps_per_epoch = len(self.train_loader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.get('lr', 2e-4),
            epochs=config['epochs'],
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,          # 10% warmup
            anneal_strategy='cos',
            div_factor=10.0,
            final_div_factor=1e4,
        )

        self.start_epoch = 0
        self.best_val_lpips = float('inf')  # track LPIPS instead of total loss
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _rebuild_loaders(self, batch_size):
        if batch_size == self._current_batch_size:
            return
        self._current_batch_size = batch_size

        # Illumination-bucketed sampler for training
        sample_weights = self.train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=False,
        )

    def _get_stage(self, epoch):
        if epoch < 40:
            return 256, 16
        elif epoch < 90:
            return 384, 8
        else:
            return 512, 4

        # ------------------------------------------------------------------
    def train_epoch(self, epoch):
            self.model.train()
            total_loss = 0
            loss_dict_sum = {}
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

            for low_img, high_img, _ in pbar:
                low_img  = low_img.to(self.device)
                high_img = high_img.to(self.device)

                with torch.cuda.amp.autocast():                                      # ADDED
                    pred = self.model(low_img)
                    loss, loss_dict = self.criterion(pred, high_img)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()                                   # CHANGED
                self.scaler.unscale_(self.optimizer)                                 # ADDED
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)                                     # CHANGED
                self.scaler.update()                                                 # ADDED
                self.scheduler.step()

                with torch.no_grad():
                    for ep, p in zip(self.ema_model.parameters(), self.model.parameters()):
                        ep.data = self.ema_decay * ep.data + (1 - self.ema_decay) * p.data

                total_loss += loss.item()
                for k, v in loss_dict.items():
                    loss_dict_sum[k] = loss_dict_sum.get(k, 0) + v

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            n = len(self.train_loader)
            return total_loss / n, {k: v / n for k, v in loss_dict_sum.items()}
    # ------------------------------------------------------------------
    def validate(self):
        self.ema_model.eval()
        total_loss = 0
        lpips_sum = 0.0
        loss_dict_sum = {}

        with torch.no_grad():
            for low_img, high_img, _ in self.val_loader:
                low_img  = low_img.to(self.device)
                high_img = high_img.to(self.device)

                pred = self.ema_model(low_img)
                loss, loss_dict = self.criterion(pred, high_img)

                lp = self.lpips_model(pred * 2 - 1, high_img * 2 - 1).mean()
                loss_dict['lpips_direct'] = lp.item()
                lpips_sum += lp.item()

                total_loss += loss.item()
                for k, v in loss_dict.items():
                    loss_dict_sum[k] = loss_dict_sum.get(k, 0) + v

        n = len(self.val_loader)
        return total_loss / n, {k: v / n for k, v in loss_dict_sum.items()}, lpips_sum / n

    # ------------------------------------------------------------------
    def save_checkpoint(self, epoch, val_loss, val_lpips, is_best=False):
        ema_sd = {k.replace('module.', ''): v
                  for k, v in self.ema_model.state_dict().items()}

        # Always save EMA weights under the standard key for inference.py
        torch.save({'ema_state_dict': ema_sd}, self.checkpoint_dir / 'best_model.pth')

        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_lpips': val_lpips,
            'best_val_lpips': self.best_val_lpips,
        }
        torch.save(ckpt, self.checkpoint_dir / 'latest.pth')

        if is_best:
            torch.save(ckpt, self.checkpoint_dir / 'best.pth')
            print(f"  ★ Saved best checkpoint (LPIPS {val_lpips:.4f})")

    # ------------------------------------------------------------------
    def load_checkpoint(self, checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        if 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'ema_state_dict' in ckpt:
                self.ema_model.load_state_dict(ckpt['ema_state_dict'])
            self.start_epoch = ckpt['epoch'] + 1
            self.best_val_lpips = ckpt.get('best_val_lpips', float('inf'))
            print(f"Resumed from epoch {ckpt['epoch']}")
        else:
            sd = {('module.' + k if not k.startswith('module.')
                   and torch.cuda.device_count() > 1 else k): v
                  for k, v in ckpt.items()}
            self.model.load_state_dict(sd, strict=False)
            print("Loaded weights-only checkpoint")

    # ------------------------------------------------------------------
    def train(self, resume_from=None):
        if resume_from:
            self.load_checkpoint(resume_from)

        print("Starting Progressive Training")
        print("=" * 60)

        for epoch in range(self.start_epoch, self.config['epochs']):
            crop_size, batch_size = self._get_stage(epoch)
            # Only rebuild loaders (and sampler) when batch size changes
            self._rebuild_loaders(batch_size)
            self.train_dataset.augmentation.current_crop_size = crop_size
            self.val_dataset.augmentation.current_crop_size   = crop_size

            lr_now = self.optimizer.param_groups[0]['lr']
            print(f"\n[Epoch {epoch + 1}/{self.config['epochs']}] "
                  f"crop={crop_size}  batch={batch_size}  lr={lr_now:.2e}")

            train_loss, train_dict = self.train_epoch(epoch)
            val_loss, val_dict, val_lpips = self.validate()

            print(f"  Train Loss : {train_loss:.6f}")
            print(f"  Val   Loss : {val_loss:.6f}")
            print(f"  Val   SSIM : {val_dict.get('ssim_score', 0):.4f}")
            print(f"  Val   LPIPS: {val_lpips:.4f}")
            print(f"  Val   CC   : gray={val_dict.get('gray_constancy', 0):.4f}  "
                  f"angle={val_dict.get('angle_color', 0):.4f}")
            print(f"  Val   FFT  : {val_dict.get('fft', 0):.6f}")

            # Best model = lowest LPIPS (direct competition metric)
            is_best = val_lpips < self.best_val_lpips
            if is_best:
                self.best_val_lpips = val_lpips

            self.save_checkpoint(epoch, val_loss, val_lpips, is_best)
    


# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',         type=str,   default=None)
    parser.add_argument('--data_root',      type=str,   default='./data')
    parser.add_argument('--checkpoint_dir', type=str,   default='./checkpoints_v3')
    parser.add_argument('--epochs',         type=int,   default=50)
    parser.add_argument('--lr',             type=float, default=2e-4)
    args = parser.parse_args()

    config = {
        'data_root':      args.data_root,
        'checkpoint_dir': args.checkpoint_dir,
        'batch_size':     16,
        'epochs':         args.epochs,
        'lr':             args.lr,
    }

    trainer = ProgressiveTrainer(config)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()

"""
losses_v3.py  —  Competition-tuned loss for low-light enhancement.

Key changes vs v2:
  1. IlluminationWeightedLoss  — dark regions get MORE gradient signal (fixes blur in black patches)
  2. GrayWorldColorConstancyLoss — directly penalises channel-mean imbalance (fixes yellow shift)
  3. AngleColorLoss — cosine-similarity in colour space (channel ratios, not magnitudes) which
     is orthogonal to L1 and strongly penalises hue drift
  4. MS-SSIM replaces single-scale SSIM — more robust, better SSIM metric score
  5. Gradient / edge loss — Sobel-based, directly helps SSIM + perceptual sharpness
  6. Phase-aware FFT loss instead of magnitude-only — helps DISTS (texture & structure)
  7. Loss weights re-tuned to balance SSIM / LPIPS / DISTS competition metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips as lpips_lib


# ---------------------------------------------------------------------------
# Helper: Multi-Scale SSIM
# ---------------------------------------------------------------------------
def _ssim_map(pred, target, window_size=11):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1 = F.avg_pool2d(pred, window_size, 1, window_size // 2)
    mu2 = F.avg_pool2d(target, window_size, 1, window_size // 2)
    s1 = F.avg_pool2d(pred ** 2, window_size, 1, window_size // 2) - mu1 ** 2
    s2 = F.avg_pool2d(target ** 2, window_size, 1, window_size // 2) - mu2 ** 2
    s12 = F.avg_pool2d(pred * target, window_size, 1, window_size // 2) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / (
        (mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2)
    )
    return ssim_map


def ms_ssim_loss(pred, target, weights=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)):
    """
    Multi-scale SSIM loss.  Input must be [B,C,H,W] in [0,1].
    Returns 1 - MS-SSIM (lower = better pred).
    """
    levels = len(weights)
    ssim_vals = []
    p, t = pred, target
    for i in range(levels):
        ssim_vals.append(_ssim_map(p, t).mean())
        if i < levels - 1:
            p = F.avg_pool2d(p, 2, 2)
            t = F.avg_pool2d(t, 2, 2)
    # product of contrast + structure terms at coarser scales, luminance at finest
    ms = torch.stack(ssim_vals)
    weighted = (ms * torch.tensor(weights, device=pred.device)).sum()
    return 1.0 - weighted.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Illumination-aware pixel weight map
# ---------------------------------------------------------------------------
def illumination_weight_map(target, alpha=3.0, eps=1e-2):
    """
    Returns a spatial weight map that up-weights dark regions.
    target : [B,C,H,W] in [0,1]
    Very dark pixels (luma ≈ 0) get weight up to ~alpha, bright get ~1.
    """
    luma = 0.299 * target[:, 0] + 0.587 * target[:, 1] + 0.114 * target[:, 2]
    luma = luma.unsqueeze(1)  # [B,1,H,W]
    # inverse luminance weighting: dark → high weight
    w = 1.0 + alpha * torch.exp(-luma / (eps + 0.15))
    # normalise so mean weight = 1
    w = w / (w.mean(dim=[2, 3], keepdim=True) + 1e-6)
    return w


# ---------------------------------------------------------------------------
# Gradient (edge) loss
# ---------------------------------------------------------------------------
def gradient_loss(pred, target):
    """
    Sobel-based gradient magnitude L1 loss.
    Ensures edges / textures are preserved — directly helps SSIM.
    """
    def sobel(x):
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        ky = kx.transpose(2, 3)
        B, C, H, W = x.shape
        x_flat = x.view(B * C, 1, H, W)
        gx = F.conv2d(x_flat, kx, padding=1)
        gy = F.conv2d(x_flat, ky, padding=1)
        return (gx ** 2 + gy ** 2 + 1e-6).sqrt().view(B, C, H, W)

    return F.l1_loss(sobel(pred), sobel(target))


# ---------------------------------------------------------------------------
# Color-constancy losses
# ---------------------------------------------------------------------------
def gray_world_color_constancy_loss(pred, target):
    """
    Penalises mismatch in per-channel means between pred and target.
    This is the single most direct fix for the yellow-shift problem.
    The network is free to get luminance right but drift in hue; this term
    explicitly pins channel ratios.
    """
    pred_mean = pred.mean(dim=[2, 3])    # [B,C]
    target_mean = target.mean(dim=[2, 3])
    return F.l1_loss(pred_mean, target_mean)


def angle_color_loss(pred, target, eps=1e-6):
    """
    Cosine-similarity loss in RGB space (per pixel).
    Penalises hue angle errors independently of brightness.
    Very effective against warm/cool tint artefacts.
    """
    B, C, H, W = pred.shape
    p = pred.view(B, C, -1)           # [B,3,N]
    t = target.view(B, C, -1)
    p_norm = p / (p.norm(dim=1, keepdim=True) + eps)
    t_norm = t / (t.norm(dim=1, keepdim=True) + eps)
    cos_sim = (p_norm * t_norm).sum(dim=1)  # [B,N]
    return (1.0 - cos_sim).mean()


# ---------------------------------------------------------------------------
# FFT loss: magnitude + phase
# ---------------------------------------------------------------------------
def fft_loss(pred, target, phase_weight=0.1):
    """
    Combined frequency magnitude + phase loss.
    Magnitude alone (v2) helps texture sharpness but not spatial coherence.
    Adding phase helps DISTS which checks structural similarity at multiple scales.
    """
    pf = torch.fft.rfft2(pred, norm='ortho')
    tf = torch.fft.rfft2(target, norm='ortho')
    mag_loss = F.l1_loss(pf.abs(), tf.abs())
    # phase: use real/imag normalised by magnitude
    p_mag = pf.abs().clamp(min=1e-6)
    t_mag = tf.abs().clamp(min=1e-6)
    phase_loss = F.l1_loss(pf / p_mag, tf / t_mag)
    return mag_loss + phase_weight * phase_loss


# ---------------------------------------------------------------------------
# Main competition loss
# ---------------------------------------------------------------------------
class CompetitionLoss(nn.Module):
    """
    Drop-in replacement for v2 CompetitionLoss.
    Same interface: forward(pred, target) → (total_loss, loss_dict)
    """

    def __init__(self):
        super().__init__()

        # VGG perceptual (relu1_2, relu2_2, relu3_3) — unchanged
        from torchvision.models import vgg16
        vgg = vgg16(pretrained=True).features
        self.vgg_layers = nn.ModuleList([
            nn.Sequential(*list(vgg[:4])),
            nn.Sequential(*list(vgg[4:9])),
            nn.Sequential(*list(vgg[9:16])),
        ])
        for p in self.vgg_layers.parameters():
            p.requires_grad = False

        # LPIPS-alex (unchanged)
        self.lpips_fn = lpips_lib.LPIPS(net='alex')
        self.lpips_fn.eval()
        for p in self.lpips_fn.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    def perceptual_loss(self, pred, target):
        loss, x, y = 0.0, pred, target
        for layer in self.vgg_layers:
            x, y = layer(x), layer(y)
            loss += F.l1_loss(x, y)
        return loss / len(self.vgg_layers)

    def charbonnier(self, pred, target, eps=1e-3):
        return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))

    # ------------------------------------------------------------------
    def forward(self, pred, target):
        # --- pixel loss (illumination-weighted Charbonnier) ---
        # Up-weight dark regions → forces the network to not "give up" there
        w = illumination_weight_map(target, alpha=3.0)
        pixel_diff = torch.sqrt((pred - target) ** 2 + 1e-6)
        l1 = (w * pixel_diff).mean()

        # --- MS-SSIM ---
        ssim = ms_ssim_loss(pred, target)

        # --- Perceptual ---
        perc = self.perceptual_loss(pred, target)

        # --- FFT (magnitude + phase) ---
        fft = fft_loss(pred, target, phase_weight=0.1)

        # --- LPIPS ---
        lp = self.lpips_fn(pred * 2 - 1, target * 2 - 1).mean()

        # --- Color constancy (THE critical new terms) ---
        gray_cc = gray_world_color_constancy_loss(pred, target)
        angle_cc = angle_color_loss(pred, target)

        # --- Edge / gradient ---
        grad = gradient_loss(pred, target)

        # ------------------------------------------------------------------
        # Weight tuning rationale:
        #   - SSIM metric  → ms_ssim is primary driver (weight 1.0)
        #   - LPIPS metric → 0.8 (same as v2, already well-tuned)
        #   - DISTS metric → fft + perceptual help texture/structure
        #   - LIQE/MUSIQ/Q-Align (NR) → gradient + perceptual help sharpness/naturalness
        #   - Color fix    → gray_cc 0.5, angle_cc 0.3 (targeted, not dominant)
        # ------------------------------------------------------------------
        total = (
            0.15 * l1
            + 1.0 * ssim
            + 0.4 * perc
            + 0.25 * fft
            + 0.8 * lp
            + 0.5 * gray_cc      # NEW: channel mean matching
            + 0.3 * angle_cc     # NEW: hue angle matching
            + 0.15 * grad        # NEW: edge sharpness
        )

        return total, {
            'total': total.item(),
            'l1_weighted': l1.item(),
            'ms_ssim': ssim.item(),
            'ssim_score': 1 - ssim.item(),
            'perceptual': perc.item(),
            'fft': fft.item(),
            'lpips': lp.item(),
            'gray_constancy': gray_cc.item(),
            'angle_color': angle_cc.item(),
            'gradient': grad.item(),
        }


# Alias kept for backwards compat
ADLNetLoss = CompetitionLoss


if __name__ == "__main__":
    crit = CompetitionLoss()
    p = torch.rand(2, 3, 256, 256)
    t = torch.rand(2, 3, 256, 256)
    loss, d = crit(p, t)
    print(f"Total loss: {loss.item():.4f}")
    for k, v in d.items():
        print(f"  {k:20s}: {v:.6f}")

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Illumination Embedding  (global scalar → channel vector)
# ---------------------------------------------------------------------------
class IlluminationEmbedding(nn.Module):
    """
    Takes a 1-dim luma scalar (mean of input image) and produces a
    channel-wise affine pair (gamma, beta) to modulate feature maps.
    Think of it as a very lightweight FiLM conditioning.
    """

    def __init__(self, num_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_features * 2),   # gamma + beta
        )
        # init so it starts as identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat, luma_scalar):
        """
        feat        : [B, C, H, W]
        luma_scalar : [B, 1]  — mean luma of the original low-light input
        """
        params = self.net(luma_scalar)          # [B, 2C]
        C = feat.shape[1]
        gamma = params[:, :C].unsqueeze(2).unsqueeze(3) + 1.0  # start at 1
        beta  = params[:, C:].unsqueeze(2).unsqueeze(3)
        return feat * gamma + beta


# ---------------------------------------------------------------------------
# Illumination-Conditioned Normalisation  (replaces NoiseAwareAdaIN)
# ---------------------------------------------------------------------------
class IlluminationConditionedNorm(nn.Module):
    """
    Instance norm + luma-conditioned affine.
    Fixes the core problem: normalisation parameters are DIFFERENT depending
    on how dark the input is, so the network stops learning one average mapping.
    """

    def __init__(self, num_features):
        super().__init__()
        self.inst_norm = nn.InstanceNorm2d(num_features, affine=False)
        self.illu_embed = IlluminationEmbedding(num_features)

    def forward(self, x, luma_scalar):
        x = self.inst_norm(x)
        return self.illu_embed(x, luma_scalar)


# ---------------------------------------------------------------------------
# Spatial attention gate  (focus refinement on dark regions)
# ---------------------------------------------------------------------------
class SpatialAttentionGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels + 1, channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, feat, luma_map):
        """
        feat     : [B, C, H, W]
        luma_map : [B, 1, H, W]  spatial luma of original input
        """
        gate = self.conv(torch.cat([feat, luma_map], dim=1))
        return feat * gate


# ---------------------------------------------------------------------------
# Exposure branch  (now with skip connection to reduce over-smoothing)
# ---------------------------------------------------------------------------
class ExposureBranch(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, 3, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.out_act = nn.Sigmoid()
        # norms need luma conditioning
        self.norm1 = IlluminationConditionedNorm(channels)
        self.norm2 = IlluminationConditionedNorm(channels)

    def forward(self, x, luma_scalar):
        feat = self.relu(self.norm1(self.conv1(x), luma_scalar))
        feat = self.relu(self.norm2(self.conv2(feat), luma_scalar))
        out = self.out_act(self.conv3(feat))
        # skip: blend output with residual-corrected input so texture isn't erased
        return out


# ---------------------------------------------------------------------------
# Multi-scale exposure fusion  (illumination-conditioned)
# ---------------------------------------------------------------------------
class MultiScaleExposureFusion(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.gamma_under = nn.Parameter(torch.ones(1) * 0.5)
        self.gamma_over  = nn.Parameter(torch.ones(1) * 1.5)

        self.enhance_under  = ExposureBranch(channels)
        self.enhance_normal = ExposureBranch(channels)
        self.enhance_over   = ExposureBranch(channels)

        # Fusion weight network: input is low-light image (3ch) + luma (1ch)
        self.fusion_weights = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x, luma_scalar, luma_map):
        under  = self.enhance_under(
            x.pow(self.gamma_under.clamp(0.2, 1.0)), luma_scalar
        )
        normal = self.enhance_normal(x, luma_scalar)
        over   = self.enhance_over(
            x.pow(self.gamma_over.clamp(1.0, 3.0)), luma_scalar
        )

        # Gate fusion by luma so dark regions bias toward over-exposed branch
        w = self.fusion_weights(torch.cat([x, luma_map], dim=1))
        return w[:, 0:1] * under + w[:, 1:2] * normal + w[:, 2:3] * over


# ---------------------------------------------------------------------------
# Main model: SCALENet v3
# ---------------------------------------------------------------------------
class SCALENet(nn.Module):
    """
    Backward-compatible with v2: SCALENet(base_channels=32)
    Forward: takes x [B,3,H,W], returns enhanced [B,3,H,W] in [0,1]
    """

    def __init__(self, base_channels=32):
        super().__init__()
        C = base_channels

        self.exposure_fusion = MultiScaleExposureFusion(channels=64)

        # Deeper refinement: 4 layers with a residual halfway
        self.refine_conv1 = nn.Conv2d(3, C, 3, padding=1)
        self.refine_norm1 = IlluminationConditionedNorm(C)

        self.refine_conv2 = nn.Conv2d(C, C, 3, padding=1)
        self.refine_norm2 = IlluminationConditionedNorm(C)

        self.refine_conv3 = nn.Conv2d(C, C, 3, padding=1)
        self.refine_norm3 = IlluminationConditionedNorm(C)

        self.refine_conv4 = nn.Conv2d(C, C, 3, padding=1)
        self.refine_norm4 = IlluminationConditionedNorm(C)

        self.refine_out = nn.Conv2d(C, 3, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        # Spatial gate: focuses refinement on dark patches
        self.spatial_gate = SpatialAttentionGate(C)

        # Final residual gate (now on features, not fused RGB)
        self.res_gate = nn.Conv2d(C, 3, 1)

        # Channel attention on output (re-calibrate channel imbalance → fixes colour)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 3),
            nn.Sigmoid()
        )

    # ------------------------------------------------------------------
    def _compute_luma(self, x):
        """Returns (scalar [B,1], map [B,1,H,W])"""
        luma_map = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
        luma_scalar = luma_map.mean(dim=[2, 3])   # [B,1]
        return luma_scalar, luma_map

    # ------------------------------------------------------------------
    def forward(self, x):
        luma_scalar, luma_map = self._compute_luma(x)

        # --- Exposure fusion ---
        fused = self.exposure_fusion(x, luma_scalar, luma_map)

        # --- Deep refinement ---
        r = self.relu(self.refine_norm1(self.refine_conv1(fused), luma_scalar))
        r = self.relu(self.refine_norm2(self.refine_conv2(r), luma_scalar))
        r_skip = r                                           # residual midpoint

        r = self.relu(self.refine_norm3(self.refine_conv3(r), luma_scalar))
        r = self.relu(self.refine_norm4(self.refine_conv4(r), luma_scalar))
        r = r + r_skip                                       # skip inside refinement

        # Spatial attention: focus on dark regions
        r = self.spatial_gate(r, luma_map)

        # Gated residual (gate from features, not fused RGB)
        gate = torch.sigmoid(self.res_gate(r))
        residual = self.tanh(self.refine_out(r))

        out = fused + gate * residual

        # Channel attention: soft re-balancing to fix colour bias
        ca = self.channel_attn(out).unsqueeze(2).unsqueeze(3)  # [B,3,1,1]
        # We apply this as a light multiplicative correction scaled to not dominate
        out = out * (0.5 + 0.5 * ca)   # range [0.5*out, out], avoids over-suppression

        return torch.clamp(out, 0.0, 1.0)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Discriminator (unchanged from v2 but kept here for completeness)
# ---------------------------------------------------------------------------
class CurriculumPatchDiscriminator(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, stride=1, padding=1),
        )

    def forward(self, x, return_difficulty=False):
        return self.model(x)


# ---------------------------------------------------------------------------
def verify_model():
    model = SCALENet(base_channels=32)
    total = model.count_parameters()
    mb = (total * 4) / (1024 ** 2)
    print(f"Params: {total:,}  |  Size: {mb:.4f} MB  |  {'PASS' if mb < 1.0 else 'FAIL (check budget)'}")
    for name, mod in model.named_children():
        p = sum(x.numel() for x in mod.parameters() if x.requires_grad)
        print(f"  {name}: {p:,}")

    x = torch.randn(1, 3, 256, 256).clamp(0, 1)
    out = model(x)
    print(f"Input: {x.shape}  Output: {out.shape}  Range: [{out.min():.3f}, {out.max():.3f}]")


if __name__ == "__main__":
    verify_model()

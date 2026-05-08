import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ============================================================
# 1. Squash Function (Sabour et al., NeurIPS 2017)
# ============================================================
def squash(x, dim=-1, eps=1e-8):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * x / torch.sqrt(squared_norm + eps)

# ============================================================
# 2. Pretrained Backbone
# ============================================================
# class PretrainedBackbone(nn.Module):
#     def __init__(self, backbone_type='densenet121', in_channels=1):
#         super().__init__()
#         self.in_channels = in_channels

#         # if backbone_type == 'efficientnet_b0':
#         #     net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
#         #     self.features = net.features
#         #     self.out_channels = 1280
#         if backbone_type == 'densenet121':
#             net = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
#             self.features = net.features
#             self.out_channels = 1024
#         else:
#             raise ValueError("backbone_type phải là 'efficientnet_b0' hoặc 'densenet121'")
class PretrainedBackbone(nn.Module):
    def __init__(self, backbone_type='densenet121', in_channels=1, freeze_until='denseblock3'):
        super().__init__()
        self.in_channels = in_channels
        net = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.features = net.features
        self.out_channels = 1024

        # Freeze toàn bộ trước, chỉ unfreeze từ denseblock4 trở đi
        for name, param in self.features.named_parameters():
            if freeze_until in name or 'norm5' in name:
                break
            param.requires_grad = False
    def forward(self, x):
        # Ảnh y tế (xám) 1 kênh cần nhân bản lên 3 kênh để khớp Pre-trained ImageNet
        if self.in_channels == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.features(x)
        return x

# ============================================================
# 3. CNN Refinement Block
# ============================================================
class CNNRefinement(nn.Module):
    def __init__(self, in_channels, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Dropout2d(dropout),          # ← thêm

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Dropout2d(dropout),          # ← thêm

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)

# ============================================================
# 4. Primary Capsules
# ============================================================
class PrimaryCaps(nn.Module):
    def __init__(self, in_channels, num_capsules=8, dim_capsule=8):
        super().__init__()
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule

        self.conv = nn.Conv2d(
            in_channels,
            num_capsules * dim_capsule,
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, x):
        x = self.conv(x)
        batch_size = x.size(0)
        
        # Bảo toàn tính toàn vẹn không gian (spatial integrity)
        x = x.view(batch_size, self.num_capsules, self.dim_capsule, -1)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(batch_size, -1, self.dim_capsule)
        
        return squash(x)

# ============================================================
# 5. Fuzzy Membership (Fuzzy C-Means)
# ============================================================
def fuzzy_membership(distances, m=2.0, eps=1e-8):
    power = 2.0 / (m - 1.0)
    d = distances + eps
    ratio = (d.unsqueeze(-1) / d.unsqueeze(-2)) ** power
    membership = 1.0 / ratio.sum(dim=-1)
    return membership

# ============================================================
# 6. Advanced Routing Capsule (ĐÃ TỐI ƯU HIỆU NĂNG)
# ============================================================
class AdvancedRoutingCaps(nn.Module):
    def __init__(self, num_caps_in, dim_caps_in, num_caps_out, dim_caps_out, num_routing=3, num_heads=4, fuzzy_m=2.0, residual_alpha=0.7):
        super().__init__()
        self.num_caps_in  = num_caps_in
        self.dim_caps_in  = dim_caps_in
        self.num_caps_out = num_caps_out
        self.dim_caps_out = dim_caps_out
        self.num_routing  = num_routing
        self.fuzzy_m      = fuzzy_m
        self.residual_alpha = residual_alpha

        self.W = nn.Parameter(0.01 * torch.randn(1, num_caps_in, num_caps_out, dim_caps_in, dim_caps_out))

        # BỎ LayerNorm vì nó triệt tiêu ý nghĩa độ dài (xác suất) của Capsule
        self.mha = nn.MultiheadAttention(embed_dim=dim_caps_out, num_heads=num_heads, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        B = x.size(0)

        # [BƯỚC 1] Prediction vectors
        x_expanded = x.unsqueeze(2).unsqueeze(-1)
        W = self.W.expand(B, -1, -1, -1, -1)
        u_hat = torch.matmul(W.transpose(-1, -2), x_expanded).squeeze(-1)

        # [BƯỚC 2] Attention + Residual + Dropout (làm 1 lần duy nhất)
        att_input = u_hat.reshape(B, self.num_caps_in * self.num_caps_out, self.dim_caps_out)
        att_out, _ = self.mha(att_input, att_input, att_input)
        att_out = att_out.reshape(B, self.num_caps_in, self.num_caps_out, self.dim_caps_out)
        att_out = self.dropout(u_hat + att_out)  # ✅ Residual + Dropout, chỉ 1 lần

        # [BƯỚC 3] Khởi tạo v_j từ att_out đã hoàn chỉnh
        c_ij_init = torch.ones(B, self.num_caps_in, self.num_caps_out, device=x.device) / self.num_caps_out
        s_j_init = torch.sum(c_ij_init.unsqueeze(-1) * att_out, dim=1)  # ✅ nhất quán
        v_j = squash(s_j_init)

        # [BƯỚC 4] Fuzzy Routing
        for _ in range(self.num_routing):
            diff = att_out - v_j.unsqueeze(1)
            distances = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)
            c_ij = fuzzy_membership(distances, m=self.fuzzy_m)
            s_j = torch.sum(c_ij.unsqueeze(-1) * att_out, dim=1)
            v_new = squash(s_j)
            v_j = self.residual_alpha * v_new + (1.0 - self.residual_alpha) * v_j

        return v_j

# ============================================================
# 7. Super Hybrid Model (Hoàn chỉnh)
# ============================================================
class SuperHybridModel(nn.Module):
    def __init__(self, backbone_type='densenet121', in_channels=1, num_classes=2, img_size=128):
        super().__init__()

        self.backbone = PretrainedBackbone(backbone_type=backbone_type, in_channels=in_channels)
        self.refinement = CNNRefinement(self.backbone.out_channels)
        self.primary_caps = PrimaryCaps(in_channels=128, num_capsules=8, dim_capsule=8)

        # Tự động tính số lượng Capsule đầu vào (num_caps_in)
        with torch.no_grad():
            dummy = torch.randn(1, in_channels, img_size, img_size)
            feat = self.backbone(dummy)
            feat = self.refinement(feat)
            prim = self.primary_caps(feat)
            self.num_caps_in = prim.size(1)

        self.digit_caps = AdvancedRoutingCaps(
            num_caps_in=self.num_caps_in,
            dim_caps_in=8,
            num_caps_out=num_classes,
            dim_caps_out=32,
            num_routing=3,
            num_heads=4,
            fuzzy_m=2.0,
            residual_alpha=0.7
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.refinement(x)
        x = self.primary_caps(x)
        x = self.digit_caps(x)

        # Chiều dài Capsule chính là xác suất của class đó
        probs = torch.norm(x, dim=-1)
        return probs

# ============================================================
# Test Model
# ============================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Sử dụng thiết bị:", device)

    # Khởi tạo mô hình
    model = SuperHybridModel(
        backbone_type='densenet121', # Có thể đổi thành 'efficientnet_b0'
        in_channels=3,
        num_classes=2,
        img_size=224
    ).to(device)

    # Test với dummy data
    dummy = torch.randn(2, 3, 224, 224).to(device)
    output = model(dummy)

    print("=> Output shape:", output.shape)
    print("=> Output values (Classes Probabilities):\n", output)
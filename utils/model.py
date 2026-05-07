import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models


# ===============================
# Squash (stable version)
# ===============================
def squash(x, dim=-1):
    norm = torch.norm(x, dim=dim, keepdim=True)
    scale = norm / (1 + norm)
    return scale * (x / (norm + 1e-8))


# ===============================
# 1. ResNet50 Encoder
# ===============================
class ResNet50Encoder(nn.Module):
    def __init__(self, freeze=True, fine_tune_last=True):
        super().__init__()
        resnet = models.resnet50(weights='DEFAULT')
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        for name, p in self.features.named_parameters():
            if freeze:
                if fine_tune_last and "layer4" in name:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            else:
                p.requires_grad = True

    def forward(self, x):
        return self.features(x)


# ===============================
# 2. CNN Feature Extractor (stronger signal)
# ===============================
class CNN2DFeatureExtractor(nn.Module):
    def __init__(self, embed_dim, num_filters=192, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(embed_dim, num_filters, k, padding=(k - 1) // 2)
            for k in kernel_sizes
        ])
        self.bn = nn.BatchNorm2d(num_filters * len(kernel_sizes))
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        out = torch.cat(conv_outs, dim=1)
        out = self.bn(out)
        return self.dropout(out)


# ===============================
# 3. Primary Capsules
# ===============================
class PrimaryCapsules2D(nn.Module):
    def __init__(self, in_channels, num_caps=16, caps_dim=8):
        super().__init__()
        self.num_caps = num_caps
        self.caps_dim = caps_dim
        self.conv = nn.Conv2d(in_channels, num_caps * caps_dim, kernel_size=1)

    def forward(self, x):
        out = self.conv(x)
        B, _, H, W = out.size()

        out = out.view(B, self.num_caps, self.caps_dim, H, W)
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        out = out.view(B, -1, self.caps_dim)

        return squash(out)


# ===============================
# 4. Fuzzy Multi-Head Routing (gradient-fixed)
# ===============================
class FuzzyMHARouting(nn.Module):
    def __init__(self, num_primary_caps, primary_dim, num_classes, out_dim, num_heads=4):
        super().__init__()

        self.num_classes = num_classes
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.W = nn.Parameter(
            torch.randn(1, num_primary_caps, num_classes, out_dim, primary_dim) * 0.01
        )

        self.query = nn.Parameter(
            torch.randn(1, 1, num_classes, num_heads, self.head_dim)
        )

        self.key_proj = nn.Linear(out_dim, out_dim)
        self.val_proj = nn.Linear(out_dim, out_dim)

        self.mu = nn.Parameter(torch.full((num_heads,), 0.5))
        self.sigma = nn.Parameter(torch.full((num_heads,), 0.2))

    def forward(self, x, routing_iters=3):
        B, N, _ = x.size()

        x_expanded = x.unsqueeze(2).unsqueeze(4)
        W_t = self.W.expand(B, -1, -1, -1, -1)
        u_hat = torch.matmul(W_t, x_expanded).squeeze(-1)

        b = torch.zeros(B, N, self.num_classes, self.num_heads, device=x.device) * 0.01

        for _ in range(routing_iters):
            c = F.softmax(b, dim=1)

            K = self.key_proj(u_hat).view(B, N, self.num_classes, self.num_heads, self.head_dim)
            V = self.val_proj(u_hat).view(B, N, self.num_classes, self.num_heads, self.head_dim)
            Q = self.query.expand(B, N, -1, -1, -1)

            # Attention
            attn_scores = (Q * K).sum(dim=-1) / math.sqrt(self.head_dim)
            attn_scores = attn_scores / 0.7   # temperature

            # ❌ BỎ sigmoid → giữ gradient mạnh
            attn_scores_norm = attn_scores

            # Fuzzy ổn định
            sigma = torch.clamp(self.sigma, 0.1, 0.5)
            fuzzy = torch.exp(-((attn_scores_norm - self.mu)**2) / (2 * sigma**2 + 1e-8))

            weights = c * fuzzy
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

            s = (weights.unsqueeze(-1) * V).sum(dim=1)
            v = squash(s.view(B, self.num_classes, self.out_dim))

            # Agreement
            v_heads = v.view(B, self.num_classes, self.num_heads, self.head_dim)
            agreement = (V * v_heads.unsqueeze(1)).sum(dim=-1)

            # 🔥 soften detach (giữ gradient nhẹ)
            b = b + 0.5 * agreement.detach()

        return v


# ===============================
# 5. Full Model
# ===============================
class SuperHybridModel(nn.Module):
    def __init__(self, num_classes=2, img_size=224):
        super().__init__()

        self.image_encoder = ResNet50Encoder(freeze=True, fine_tune_last=True)

        self.cnn = CNN2DFeatureExtractor(embed_dim=2048, num_filters=192)
        cnn_out_channels = 192 * 3

        self.primary_caps = PrimaryCapsules2D(
            in_channels=cnn_out_channels,
            num_caps=16,
            caps_dim=8
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            feat = self.image_encoder(dummy)
            _, _, H, W = feat.shape

        num_primary_caps = 16 * (H * W)

        self.class_caps = FuzzyMHARouting(
            num_primary_caps=num_primary_caps,
            primary_dim=8,
            num_classes=num_classes,
            out_dim=16,
            num_heads=4
        )

    def forward(self, x):
        x = self.image_encoder(x)
        x = self.cnn(x)
        x = self.primary_caps(x)
        x = self.class_caps(x)

        # 🔥 scale logits để tăng gradient
        logits = torch.sqrt((x ** 2).sum(dim=-1) + 1e-8)
        return logits


# ===============================
# DEBUG GRADIENT
# ===============================import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def squash(x, dim=-1):
    '''
    Hàm kích hoạt Squash: Nén độ dài vector vào khoảng [0, 1].
    Sử dụng cho Capsule Networks để đại diện cho xác suất tồn tại của thực thể.
    '''
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm) / torch.sqrt(squared_norm + 1e-8)
    return scale * x

class MedDINOEncoder(nn.Module):
    '''
    Wrapper cho MedDINO (ViT-based).
    Chuyển đổi các patch tokens từ dạng chuỗi (Sequence) về dạng lưới 2D (Feature Map).
    '''
    def __init__(self, embed_dim=768, freeze_backbone=True):
        super().__init__()
        # Giả định sử dụng kiến trúc DINOv2 (ViT-B/14) thường thấy ở MedDINOv3
        try:
            self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        except:
            # Fallback nếu không tải được qua hub
            from torchvision.models import vit_b_16
            self.backbone = vit_b_16()
            
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.dino_dim = 768 
        self.proj = nn.Conv2d(self.dino_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        if isinstance(features, dict):
            patch_tokens = features['x_norm_patchtokens']
        else:
            patch_tokens = features[:, 1:, :] 
            
        B, N, D = patch_tokens.shape
        grid_size = int(math.sqrt(N))
        
        # Reshape về 2D: (B, D, grid_size, grid_size)
        feat_2d = patch_tokens.transpose(1, 2).contiguous().view(B, D, grid_size, grid_size)
        
        out = self.proj(feat_2d)
        return out

class CNN2DFeatureExtractor(nn.Module):
    '''
    CNN 2D dùng để trích xuất đặc trưng cục bộ từ feature map của MedDINO.
    '''
    def __init__(self, embed_dim, num_filters, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=embed_dim, out_channels=num_filters, 
                      kernel_size=k, padding=(k-1)//2)
            for k in kernel_sizes
        ])

    def forward(self, x):
        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        out = torch.cat(conv_outs, dim=1)
        return out

class PrimaryCapsules2D(nn.Module):
    '''
    Lớp Primary Capsule 2D: Nhận feature maps và đóng gói thành các vectors.
    '''
    def __init__(self, in_channels, num_caps, caps_dim):
        super().__init__()
        self.num_caps = num_caps
        self.caps_dim = caps_dim
        self.conv = nn.Conv2d(in_channels, num_caps * caps_dim, kernel_size=1)

    def forward(self, x):
        out = self.conv(x)
        B, _, H, W = out.size()
        
        out = out.view(B, self.num_caps, self.caps_dim, H, W)
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        out = out.view(B, -1, self.caps_dim)
        
        return squash(out)

class FuzzyMHARouting(nn.Module):
    '''
    Hybrid Routing: Multi-head Attention + Fuzzy Logic Gaussian.
    '''
    def __init__(self, num_primary_caps, primary_dim, num_classes, out_dim, num_heads=4):
        super().__init__()
        self.num_classes = num_classes
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.W = nn.Parameter(torch.randn(1, num_primary_caps, num_classes, out_dim, primary_dim) * 0.01)

        self.query = nn.Parameter(torch.randn(1, 1, num_classes, num_heads, self.head_dim))
        self.key_proj = nn.Linear(out_dim, out_dim)
        self.val_proj = nn.Linear(out_dim, out_dim)

        self.mu = nn.Parameter(torch.full((num_heads,), 0.5))
        self.sigma = nn.Parameter(torch.full((num_heads,), 0.2))

    def forward(self, x):
        batch_size = x.size(0)
        num_primary_caps = x.size(1)

        x_expanded = x.unsqueeze(2).unsqueeze(4) 
        W_t = self.W.expand(batch_size, -1, -1, -1, -1)
        u_hat = torch.matmul(W_t, x_expanded).squeeze(-1) 

        K = self.key_proj(u_hat).view(batch_size, num_primary_caps, self.num_classes, self.num_heads, self.head_dim)
        V = self.val_proj(u_hat).view(batch_size, num_primary_caps, self.num_classes, self.num_heads, self.head_dim)
        Q = self.query.expand(batch_size, num_primary_caps, -1, -1, -1)

        attn_scores = (Q * K).sum(dim=-1) / math.sqrt(self.head_dim) 

        fuzzy_membership = torch.exp(-((attn_scores - self.mu)**2) / (2 * self.sigma**2 + 1e-8))
        
        fuzzy_attn_scores = attn_scores * fuzzy_membership
        routing_weights = F.softmax(fuzzy_attn_scores, dim=1) 

        out_caps = (routing_weights.unsqueeze(-1) * V).sum(dim=1) 
        out_caps = out_caps.view(batch_size, self.num_classes, self.out_dim)
        
        return squash(out_caps)

class SuperHybridModel(nn.Module):
    '''
    Kiến trúc toàn vẹn: MedDINO -> CNN 2D -> Primary Caps -> Fuzzy MHA Routing.
    '''
    def __init__(self, embed_dim=768, num_classes=2, img_size=224, patch_size=14):
        super().__init__()
        
        self.image_encoder = MedDINOEncoder(embed_dim=embed_dim)
        
        self.cnn = CNN2DFeatureExtractor(embed_dim=embed_dim, num_filters=64, kernel_sizes=[3, 5, 7])
        cnn_out_channels = 64 * 3
        
        self.primary_caps = PrimaryCapsules2D(in_channels=cnn_out_channels, num_caps=32, caps_dim=8)
        
        grid_size = img_size // patch_size
        num_primary_caps = 32 * (grid_size * grid_size)
        
        self.class_caps = FuzzyMHARouting(
            num_primary_caps=num_primary_caps,
            primary_dim=8,
            num_classes=num_classes,
            out_dim=16,
            num_heads=4
        )

    def forward(self, x):
        x = self.image_encoder(x)          
        cnn_feats = self.cnn(x)            
        p_caps = self.primary_caps(cnn_feats) 
        c_caps = self.class_caps(p_caps)      
        
        logits = (c_caps ** 2).sum(dim=-1) ** 0.5
        return logits

if __name__ == "__main__":
    model = SuperHybridModel(num_classes=2, img_size=224, patch_size=14)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape (Logits): {output.shape}")
def check_grad_flow(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: {param.grad.abs().mean().item():.6f}")


# ===============================
# TEST
# ===============================
if __name__ == "__main__":
    model = SuperHybridModel(num_classes=2, img_size=224)

    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)

    print("Input :", dummy_input.shape)
    print("Output:", output.shape)
    print(output)
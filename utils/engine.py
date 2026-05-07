import os
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import torch.nn.functional as F
# Bật dòng này lên nếu vẫn bị NaN để PyTorch chỉ đích danh dòng code gây lỗi (chỉ dùng khi debug)
# torch.autograd.set_detect_anomaly(True)
import numpy as np

# ===================== Margin Loss =====================
class MarginLoss(nn.Module):
    def __init__(self, m_plus=0.9, m_minus=0.1, lambda_=0.5):
        super().__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_ = lambda_

    def forward(self, logits, targets):
        present = torch.clamp(self.m_plus - logits, min=0.) ** 2
        absent = torch.clamp(logits - self.m_minus, min=0.) ** 2

        loss = targets * present + self.lambda_ * (1 - targets) * absent
        return loss.sum(dim=1).mean()
# ===================== Early Stopping =====================
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0, path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        # Khởi tạo bằng vô cực dương thay vì None để logic so sánh toán học chuẩn xác ngay từ đầu
        self.best_loss = np.inf 
        self.early_stop = False
        self.path = path # Đường dẫn để lưu model tốt nhất

    def __call__(self, val_loss, model):
        # Nếu val_loss giảm sâu hơn mức best_loss tối thiểu (min_delta)
        if val_loss < self.best_loss - self.min_delta:
            self.save_checkpoint(val_loss, model)
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"⚠️ EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        """Lưu lại mô hình khi validation loss giảm."""
        if self.best_loss == np.inf:
            print(f"✅ Khởi tạo val_loss: {val_loss:.6f}. Saving model...")
        else:
            print(f"✅ val_loss giảm ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...")
            
        torch.save(model.state_dict(), self.path)
# ===================== Training & Validation Loops =====================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    # Khởi tạo thanh tiến trình tqdm
    pbar = tqdm(loader, desc="🚀 Training")
    
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        probs = model(imgs)
        # Convert labels to one-hot encoding for MarginLoss
        targets_onehot = F.one_hot(labels, num_classes=probs.size(1)).float()
        loss = criterion(probs, targets_onehot)
        
        loss.backward()
        
        # Clip gradient: Cực kỳ quan trọng với CapsNet để chặn Exploding Gradient
        try:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, error_if_nonfinite=False)
        except RuntimeError:
            # Nếu xảy ra lỗi CUDA, skip gradient clipping cho batch này
            pass
        
        optimizer.step()
        
        # Lưu kết quả
        running_loss += loss.item()
        all_preds.extend(probs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Hiển thị loss trực tiếp trên thanh chạy
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    avg_loss = running_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, acc

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    pbar = tqdm(loader, desc="🔍 Validating")
    
    with torch.no_grad():
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            probs = model(imgs)
            # Convert labels to one-hot encoding for MarginLoss
            targets_onehot = F.one_hot(labels, num_classes=probs.size(1)).float()
            loss = criterion(probs, targets_onehot)
            
            running_loss += loss.item()
            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
    metrics = {
        'loss': running_loss / len(loader),
        'acc': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0)
    }
    
    return metrics

# ===================== Utilities =====================
def save_learning_curves(train_losses, val_losses, train_accs, val_accs, filepath="learning_curves.png"):
    """
    Hàm vẽ và lưu biểu đồ Loss và Accuracy qua các Epochs.
    """
    # Tạo thư mục nếu đường dẫn có chứa thư mục con (VD: 'results/curves.png')
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- Đồ thị Loss ---
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', marker='o')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', marker='s')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # --- Đồ thị Accuracy ---
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc', marker='o')
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc', marker='s')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300) # dpi=300 cho ảnh nét, dễ copy vào paper
    plt.close()
    
    print(f"📈 Đã lưu biểu đồ huấn luyện tại: {filepath}")


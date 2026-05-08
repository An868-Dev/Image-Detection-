import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import từ các module của bạn
from data.dataset import load_records_kits, MedicalImageDataset, get_transforms
from utils.model import SuperHybridModel
from utils.engine import MarginLoss


def validate_model(model_path, data_root=".", batch_size=32, img_size=224, split_type="val"):
    """
    Hàm validation model trên tập val hoặc train của KiTS19 dataset
    
    Args:
        model_path: đường dẫn đến file .pth
        data_root: đường dẫn tới thư mục cha của 'dataset'
        batch_size: kích thước batch
        img_size: kích thước ảnh đầu vào
        split_type: "train" hoặc "val"
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Sử dụng device: {device}")
    
    # ==========================================
    # 1. CHUẨN BỊ DỮ LIỆU
    # ==========================================
    print(f"\n📦 Chuẩn bị dữ liệu từ dataset folder...")
    _, val_transform = get_transforms(img_size)
    
    # Load records từ folder dataset (giống như train)
    dataset_path = os.path.join(data_root, "dataset")
    records = load_records_kits(dataset_path, split_type=split_type)
    
    if not records:
        print(f"❌ Lỗi: Không tìm thấy records cho split: {split_type}")
        print(f"   Kiểm tra lại đường dẫn: {dataset_path}")
        return None
    
    val_dataset = MedicalImageDataset(records=records, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Lấy thông tin về labels (từ records)
    labels_set = set([r["label"] for r in records])
    NUM_CLASSES = max(labels_set) + 1 if labels_set else 2
    
    print(f"✅ Đã load {len(records)} records từ split '{split_type}'")
    print(f"📊 Số lượng classes: {NUM_CLASSES}")
    print(f"   Labels: {sorted(labels_set)}")
    print(f"📈 Số lượng samples: {len(val_dataset)}")
    
    # ==========================================
    # 2. TẢI MODEL
    # ==========================================
    print(f"\n🔄 Tải model từ: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ Lỗi: File {model_path} không tồn tại!")
        return None
    
    model = SuperHybridModel(in_channels=3, num_classes=NUM_CLASSES, img_size=img_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Tải model thành công!")
    
    # ==========================================
    # 3. CHẠY VALIDATION
    # ==========================================
    print(f"\n🧪 Bắt đầu kiểm tra trên tập {split_type}...")
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    criterion = MarginLoss()
    running_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Validating ({split_type})")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Forward pass
            probs = model(imgs)
            
            # Tính loss
            targets_onehot = F.one_hot(labels, num_classes=NUM_CLASSES).float()
            loss = criterion(probs, targets_onehot)
            running_loss += loss.item()
            
            # Lưu predictions và labels
            preds = probs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # ==========================================
    # 4. TÍNH TOÁN METRICS
    # ==========================================
    print(f"\n{'='*60}")
    print(f"{'KẾT QUẢ ĐÁNH GIÁ - VALIDATION RESULTS':^60}")
    print(f"{'='*60}\n")
    
    avg_loss = running_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print(f"📊 Loss              : {avg_loss:.6f}")
    print(f"🎯 Accuracy         : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📈 Macro F1-Score   : {f1_macro:.4f}")
    print(f"📈 Weighted F1-Score: {f1_weighted:.4f}")
    print(f"🔍 Precision (Macro): {precision:.4f}")
    print(f"📍 Recall (Macro)   : {recall:.4f}")
    
    # ==========================================
    # 5. CLASSIFICATION REPORT
    # ==========================================
    print(f"\n{'─'*60}")
    print("📋 CLASSIFICATION REPORT:")
    print(f"{'─'*60}\n")
    
    class_names = [f"Label {i}" for i in sorted(labels_set)]
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0))
    
    # ==========================================
    # 6. CONFUSION MATRIX
    # ==========================================
    print(f"\n{'─'*60}")
    print("🔄 CONFUSION MATRIX:")
    print(f"{'─'*60}\n")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # ==========================================
    # 7. VẼ CONFUSION MATRIX
    # ==========================================
    print(f"\n📊 Vẽ Confusion Matrix...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {split_type.upper()} Set', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{split_type}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu: confusion_matrix_{split_type}.png")
    
    # ==========================================
    # 8. VẼ CONFIDENCE DISTRIBUTION
    # ==========================================
    print(f"\n📊 Vẽ biểu đồ độ tin cậy (Confidence)...")
    all_probs = np.array(all_probs)
    max_probs = np.max(all_probs, axis=1)
    
    plt.figure(figsize=(10, 5))
    plt.hist(max_probs, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    plt.xlabel('Confidence (Max Probability)', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title(f'Distribution of Model Confidence on {split_type.upper()} Set', fontsize=14, fontweight='bold')
    plt.axvline(np.mean(max_probs), color='red', linestyle='--', label=f'Mean: {np.mean(max_probs):.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'confidence_distribution_{split_type}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu: confidence_distribution_{split_type}.png")
    
    # ==========================================
    # 9. TÓM TẮT KẾT QUẢ
    # ==========================================
    results = {
        'model_path': model_path,
        'split_type': split_type,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
        'loss': avg_loss,
        'num_samples': len(val_dataset),
        'num_classes': NUM_CLASSES,
        'class_names': class_names
    }
    
    print(f"\n{'='*60}")
    print(f"✨ VALIDATION HOÀN THÀNH ✨")
    print(f"{'='*60}")
    print(f"\n📁 Model Path: {model_path}")
    print(f"📂 Split: {split_type.upper()}")
    print(f"✅ Số lượng samples: {len(val_dataset)}")
    print(f"🎯 Độ chính xác: {accuracy*100:.2f}%")
    print(f"📈 F1-Score (Macro): {f1_macro:.4f}\n")
    
    return results


if __name__ == "__main__":
    import sys
    
    # ==========================================
    # CẤU HÌNH VALIDATION
    # ==========================================
    MODEL_PATH = "best_super_hybrid.pth"  # Hoặc "best_model.pth" nếu sử dụng Early Stopping
    DATA_ROOT = "."  # Thư mục cha của 'dataset'
    BATCH_SIZE = 32
    IMG_SIZE = 224
    
    # VALIDATION 1: Kiểm tra trên tập VAL
    print("\n" + "="*60)
    print("VALIDATION 1: KIỂM TRA TRÊN TẬP VAL")
    print("="*60)
    results_val = validate_model(MODEL_PATH, DATA_ROOT, BATCH_SIZE, IMG_SIZE, split_type="val")
    
    # VALIDATION 2: (Tùy chọn) Kiểm tra trên tập TRAIN để xem overfitting
    print("\n" + "="*60)
    print("VALIDATION 2: KIỂM TRA TRÊN TẬP TRAIN (Để xem Overfitting)")
    print("="*60)
    results_train = validate_model(MODEL_PATH, DATA_ROOT, BATCH_SIZE, IMG_SIZE, split_type="train")
    
    # So sánh kết quả
    if results_val and results_train:
        print("\n" + "="*60)
        print("SO SÁNH KẾT QUẢ TRAIN vs VAL")
        print("="*60)
        print(f"\n{'Metric':<20} {'Train':<15} {'Val':<15} {'Diff':<15}")
        print("─"*60)
        print(f"{'Accuracy':<20} {results_train['accuracy']:.4f} {results_val['accuracy']:.4f} {results_train['accuracy']-results_val['accuracy']:.4f}")
        print(f"{'Loss':<20} {results_train['loss']:.4f} {results_val['loss']:.4f} {results_train['loss']-results_val['loss']:.4f}")
        print(f"{'F1-Macro':<20} {results_train['f1_macro']:.4f} {results_val['f1_macro']:.4f} {results_train['f1_macro']-results_val['f1_macro']:.4f}")
        print(f"{'Precision':<20} {results_train['precision']:.4f} {results_val['precision']:.4f} {results_train['precision']-results_val['precision']:.4f}")
        print(f"{'Recall':<20} {results_train['recall']:.4f} {results_val['recall']:.4f} {results_train['recall']-results_val['recall']:.4f}")

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
from data.dataset import MedicalImageDataset, get_transforms
from utils.model import SuperHybridModel
from utils.engine import MarginLoss


def test_model(model_path, test_dir, batch_size=32, img_size=224):
    """
    Hàm test model trên tập test dataset
    
    Args:
        model_path: đường dẫn đến file .pth
        test_dir: đường dẫn tới thư mục test dataset
        batch_size: kích thước batch
        img_size: kích thước ảnh đầu vào
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Sử dụng device: {device}")
    
    # ==========================================
    # 1. CHUẨN BỊ DỮ LIỆU
    # ==========================================
    print(f"\n📦 Chuẩn bị dữ liệu test từ: {test_dir}")
    _, _, test_transform = get_transforms(img_size)
    test_dataset = MedicalImageDataset(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    NUM_CLASSES = len(test_dataset.classes)
    class_names = test_dataset.classes
    print(f"📊 Số lượng classes: {NUM_CLASSES}")
    print(f"📋 Tên classes: {class_names}")
    print(f"📈 Số lượng samples test: {len(test_dataset)}")
    
    # ==========================================
    # 2. TẢI MODEL
    # ==========================================
    print(f"\n🔄 Tải model từ: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ Lỗi: File {model_path} không tồn tại!")
        return
    
    model = SuperHybridModel(num_classes=NUM_CLASSES, img_size=img_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Tải model thành công!")
    
    # ==========================================
    # 3. CHẠY TEST
    # ==========================================
    print(f"\n🧪 Bắt đầu kiểm tra trên tập test...")
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    criterion = MarginLoss()
    running_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
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
    print(f"{'KẾT QUẢ ĐÁNH GIÁ - TEST RESULTS':^60}")
    print(f"{'='*60}\n")
    
    avg_loss = running_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
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
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
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
    plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Đã lưu: confusion_matrix.png")
    
    # ==========================================
    # 8. VẼ CONFIDENCE DISTRIBUTION
    # ==========================================
    print(f"\n📊 Vẽ biểu đồ độ tin cậy (Confidence)...")
    all_probs = np.array(all_probs)
    max_probs = np.max(all_probs, axis=1)
    
    plt.figure(figsize=(10, 5))
    plt.hist(max_probs, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Confidence (Max Probability)', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Distribution of Model Confidence on Test Set', fontsize=14, fontweight='bold')
    plt.axvline(np.mean(max_probs), color='red', linestyle='--', label=f'Mean: {np.mean(max_probs):.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Đã lưu: confidence_distribution.png")
    
    # ==========================================
    # 9. TÓMLẠO KẾT QUẢ
    # ==========================================
    results = {
        'model_path': model_path,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
        'loss': avg_loss,
        'num_test_samples': len(test_dataset),
        'num_classes': NUM_CLASSES,
        'class_names': class_names
    }
    
    print(f"\n{'='*60}")
    print(f"✨ TEST HÀN ✨")
    print(f"{'='*60}")
    print(f"\n📁 Model Path: {model_path}")
    print(f"✅ Số lượng test samples: {len(test_dataset)}")
    print(f"🎯 Độ chính xác cuối cùng: {accuracy*100:.2f}%\n")
    
    return results


def test_single_image(model_path, image_path, img_size=224, num_classes=2):
    """
    Test model trên một ảnh đơn lẻ
    
    Args:
        model_path: đường dẫn đến file .pth
        image_path: đường dẫn đến ảnh cần test
        img_size: kích thước ảnh đầu vào
        num_classes: số lượng classes
    """
    from PIL import Image
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Sử dụng device: {device}")
    
    # Load model
    print(f"\n🔄 Tải model từ: {model_path}")
    model = SuperHybridModel(num_classes=num_classes, img_size=img_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Tải model thành công!")
    
    # Load ảnh
    print(f"\n📷 Load ảnh từ: {image_path}")
    if not os.path.exists(image_path):
        print(f"❌ Lỗi: File ảnh {image_path} không tồn tại!")
        return
    
    _, _, test_transform = get_transforms(img_size)
    img = Image.open(image_path).convert('RGB')
    img_tensor = test_transform(img).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = logits.argmax(dim=1).item()
        confidence = probs.max().item()
    
    print(f"\n🎯 KẾT QUẢ DỰ ĐOÁN:")
    print(f"   Lớp dự đoán: Class {pred_class}")
    print(f"   Độ tin cậy: {confidence*100:.2f}%")
    print(f"\n📊 Xác suất cho từng lớp:")
    for i, prob in enumerate(probs[0].cpu().numpy()):
        print(f"   Class {i}: {prob*100:.2f}%")


if __name__ == "__main__":
    import sys
    
    # ==========================================
    # CẤU HÌNH TEST
    # ==========================================
    MODEL_PATH = "best_super_hybrid.pth"  # Hoặc "best_model.pth" nếu sử dụng Early Stopping
    TEST_DIR = "dataset/test"
    BATCH_SIZE = 32
    IMG_SIZE = 224
    NUM_CLASSES = 2  # Sửa lại nếu cần
    
    # TEST 1: Kiểm tra trên toàn bộ tập test
    print("\n" + "="*60)
    print("TEST 1: KIỂM TRA TRÊN TOÀN BỘ TẬP TEST")
    print("="*60)
    results = test_model(MODEL_PATH, TEST_DIR, BATCH_SIZE, IMG_SIZE)
    
    # TEST 2: (Tùy chọn) Kiểm tra trên một ảnh đơn lẻ
    # Bỏ comment để kiểm tra
    # print("\n" + "="*60)
    # print("TEST 2: KIỂM TRA TRÊN MỘT ẢNH ĐƠN LẺ")
    # print("="*60)
    # test_single_image(MODEL_PATH, "path/to/your/image.jpg", IMG_SIZE, NUM_CLASSES)

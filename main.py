import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# Import từ các module của bạn
from data.dataset import MedicalImageDataset, get_transforms
from utils.model import SuperHybridModel
from utils.engine import MarginLoss, train_epoch, validate_epoch, save_learning_curves, EarlyStopping
def main():
    # ==========================================
    # 1. CẤU HÌNH (HYPERPARAMETERS)
    # ==========================================
    EPOCHS = 8
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    IMG_SIZE = 224
    
    # Đường dẫn thư mục (Dựa trên cấu trúc chuẩn của bạn)
    DATA_ROOT = "dataset" # Sửa lại cho đúng thư mục chứa train/val/test
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    VAL_DIR = os.path.join(DATA_ROOT, "val")
    TEST_DIR = os.path.join(DATA_ROOT, "test")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Bắt đầu huấn luyện trên thiết bị: {device}")
    
    # ==========================================
    # 2. CHUẨN BỊ DỮ LIỆU
    # ==========================================
    train_transform, val_transform, test_transform = get_transforms(IMG_SIZE)
    
    train_dataset = MedicalImageDataset(TRAIN_DIR, transform=train_transform)
    val_dataset = MedicalImageDataset(VAL_DIR, transform=val_transform)
    test_dataset = MedicalImageDataset(TEST_DIR, transform=test_transform)
    
    NUM_CLASSES = len(train_dataset.classes)
    print(f"📦 Số lượng classes tìm thấy: {NUM_CLASSES} -> {train_dataset.classes}")
    
    # Tối ưu num_workers (nếu chạy trên Windows máy cá nhân có thể để 0 hoặc 2)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # ==========================================
    # 3. KHỞI TẠO MÔ HÌNH & TỐI ƯU HÓA
    # ==========================================
    model = SuperHybridModel(num_classes=NUM_CLASSES, img_size=IMG_SIZE).to(device)
    criterion = MarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Công cụ hỗ trợ
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    history = {'t_loss': [], 'v_loss': [], 't_acc': [], 'v_acc': []}
    best_f1 = 0.0
    
    # ==========================================
    # 4. VÒNG LẶP HUẤN LUYỆN
    # ==========================================
    for epoch in range(EPOCHS):
        print(f"\n{'='*30}\nEpoch {epoch+1}/{EPOCHS}\n{'='*30}")
        
        # Huấn luyện và Đánh giá
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Giải nén metrics
        v_loss = v_metrics['loss']
        v_acc = v_metrics['acc']
        v_f1_macro = v_metrics['f1_macro']
        
        # Lưu lịch sử để vẽ biểu đồ
        history['t_loss'].append(t_loss)
        history['v_loss'].append(v_loss)
        history['t_acc'].append(t_acc)
        history['v_acc'].append(v_acc)
        
        print(f"📊 Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
        print(f"🎯 Val Acc: {v_acc:.4f} | Val Macro-F1: {v_f1_macro:.4f} | Recall: {v_metrics['recall']:.4f}")
        
        # Vẽ biểu đồ trực tiếp sau mỗi epoch
        save_learning_curves(history['t_loss'], history['v_loss'], history['t_acc'], history['v_acc'])
        
        # Lưu mô hình tốt nhất (dựa trên Macro-F1 vì y tế quan trọng chỉ số này)
        if v_f1_macro > best_f1:
            best_f1 = v_f1_macro
            torch.save(model.state_dict(), "best_super_hybrid.pth")
            print("💾 >>> Đã lưu model tốt nhất mới!")
            
        # Kiểm tra Early Stopping
        early_stopping(v_loss, model)
        if early_stopping.early_stop:
            print("🛑 Early stopping được kích hoạt. Quá trình huấn luyện dừng lại để tránh Overfitting.")
            break

    # ==========================================
    # 5. ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST
    # ==========================================
    print(f"\n{'*'*40}\nKIỂM THỬ TRÊN TẬP TEST ĐỘC LẬP\n{'*'*40}")
    # Load lại weights tốt nhất
    model.load_state_dict(torch.load("best_model.pth"))
    
    test_metrics = validate_epoch(model, test_loader, criterion, device)
    print("\n[KẾT QUẢ FINAL - BÁO CÁO KHOA HỌC]")
    print(f"Test Accuracy    : {test_metrics['acc']:.4f}")
    print(f"Test Macro-F1    : {test_metrics['f1_macro']:.4f}")
    print(f"Test Precision   : {test_metrics['precision']:.4f}")
    print(f"Test Recall      : {test_metrics['recall']:.4f}")
    print(f"Test Weighted-F1 : {test_metrics['f1_weighted']:.4f}")

if __name__ == "__main__":
    main()
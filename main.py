import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# Import từ file data.py vừa tạo
from data.dataset import MedicalImageDataset, get_transforms, load_records_kits

# Import từ thư mục utils của bạn
from utils.model import SuperHybridModel
from utils.engine import MarginLoss, train_epoch, validate_epoch, save_learning_curves, EarlyStopping

def main():
    # 1. CẤU HÌNH
    EPOCHS = 8
    BATCH_SIZE = 16 
    LEARNING_RATE = 3e-4 
    IMG_SIZE = 224
    DATA_ROOT = r"f:\Paper20tr\dataset" # Đường dẫn "vàng" của bạn
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Thiết bị hiện tại: {device}")
    
    # 2. CHUẨN BỊ DATA
    print("🔍 Đang quét và phân chia dữ liệu KiTS19...")
    train_records = load_records_kits(DATA_ROOT, split_type="train")
    val_records   = load_records_kits(DATA_ROOT, split_type="val")
    
    if not train_records:
        print("❌ Lỗi: Không tìm thấy dữ liệu. Dừng chương trình.")
        return
        
    train_trans, val_trans = get_transforms(IMG_SIZE)
    
    train_ds = MedicalImageDataset(records=train_records, transform=train_trans)
    val_ds   = MedicalImageDataset(records=val_records, transform=val_trans)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # KiTS19 label ác tính là 3 -> Cần 4 classes (0, 1, 2, 3)
    NUM_CLASSES = 4 
    
    # 3. KHỞI TẠO MÔ HÌNH
    model = SuperHybridModel(
        backbone_type='densenet121', 
        in_channels=3, 
        num_classes=NUM_CLASSES, 
        img_size=IMG_SIZE
    ).to(device)
    
    criterion = MarginLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3) 
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    
    history = {'t_loss': [], 'v_loss': [], 't_acc': [], 'v_acc': []}
    best_f1 = 0.0
    
    # 4. TRAINING LOOP
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_metrics = validate_epoch(model, val_loader, criterion, device)
        
        v_loss, v_acc, v_f1_macro = v_metrics['loss'], v_metrics['acc'], v_metrics['f1_macro']
        
        history['t_loss'].append(t_loss)
        history['v_loss'].append(v_loss)
        history['t_acc'].append(t_acc)
        history['v_acc'].append(v_acc)
        
        print(f"📊 Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
        print(f"🎯 Val Acc: {v_acc:.4f} | F1: {v_f1_macro:.4f}")
        
        save_learning_curves(history['t_loss'], history['v_loss'], history['t_acc'], history['v_acc'])
        
        if v_f1_macro > best_f1:
            best_f1 = v_f1_macro
            torch.save(model.state_dict(), "best_super_hybrid.pth")
            print("💾 >>> Saved Best Model!")
            
        early_stopping(v_loss, model)
        if early_stopping.early_stop:
            print("🛑 Early stopping activated.")
            break

    print("\n✅ Huấn luyện hoàn tất!")

if __name__ == "__main__":
    main()
import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Đọc danh sách class (NORMAL, PNEUMONIA) từ folder
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                # Kiểm tra định dạng ảnh phổ biến trong y tế
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Đọc ảnh bằng OpenCV (BGR) -> Chuyển sang RGB
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, torch.tensor(label, dtype=torch.long)

def get_transforms(img_size=224):
    # Augmentation cho tập Train: Thêm xoay nhẹ và lật dọc (vì đôi khi ảnh X-quang bị ngược)
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    # Val và Test giữ nguyên kích thước để đảm bảo tính khách quan
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    test_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform, test_transform

if __name__ == "__main__":
    # 1. Khởi tạo transform (lấy cả 3 bộ cho chắc chắn)
    # Lưu ý: img_size=128 như bạn đã định nghĩa
    train_trans, val_trans, test_trans = get_transforms(img_size=224)

    # 2. Định nghĩa đường dẫn gốc (Dựa trên cấu trúc trong image_3e85f8.png)
    # Nếu file script này nằm cùng cấp với các thư mục test, train, val thì dùng "."
    data_root = "dataset" 

    try:
        # 3. Khởi tạo Dataset
        train_ds = MedicalImageDataset(root_dir=os.path.join(data_root, "train"), transform=train_trans)
        val_ds   = MedicalImageDataset(root_dir=os.path.join(data_root, "val"),   transform=val_trans)
        test_ds  = MedicalImageDataset(root_dir=os.path.join(data_root, "test"),  transform=test_trans)

        print("--- THÔNG TIN DATASET ---")
        print(f"Tìm thấy các lớp: {train_ds.classes}")
        print(f"Mapping Class -> Index: {train_ds.class_to_idx}")
        print(f"Số lượng ảnh Train: {len(train_ds)}")
        print(f"Số lượng ảnh Val:   {len(val_ds)}")
        print(f"Số lượng ảnh Test:  {len(test_ds)}")

        # 4. Kiểm tra DataLoader với 1 batch
        loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        images, labels = next(iter(loader))

        print("\n--- KIỂM TRA BATCH ĐẦU TIÊN ---")
        print(f"Kích thước Tensor ảnh (Batch, C, H, W): {images.shape}")
        print(f"Kích thước Tensor nhãn: {labels.shape}")
        print(f"Nhãn trong batch này: {labels}")
        
        # Kiểm tra giá trị pixel (sau Normalize thường nằm trong khoảng [-3, 3])
        print(f"Giá trị Pixel lớn nhất: {images.max():.4f}")
        print(f"Giá trị Pixel nhỏ nhất: {images.min():.4f}")

        print("\n=> DATASET ĐÃ SẴN SÀNG ĐỂ TRAIN!")

    except FileNotFoundError as e:
        print(f"\n[LỖI]: Không tìm thấy thư mục dữ liệu. Kiểm tra lại đường dẫn.")
        print(f"Chi tiết: {e}")
    except Exception as e:
        print(f"\n[LỖI HỆ THỐNG]: {e}")
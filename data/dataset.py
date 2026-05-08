import os
import json
import random
import cv2
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==========================================
# HÀM TẢI DỮ LIỆU KITS19 (DUMMY MODE)
# ==========================================
def load_records_kits(root_dir, split_type="train"):
    random.seed(42)
    records = []
    
    root_dir = os.path.abspath(str(root_dir))
    json_path = os.path.join(root_dir, "kits.json")
    
    if not os.path.exists(json_path):
        print(f"[CẢNH BÁO] Không tìm thấy kits.json tại: {json_path}")
        return records

    with open(json_path, "r", encoding="utf-8") as json_file:
        metadata = json.load(json_file)

    # Nhãn: 0 (Lành tính) và 3 (Ác tính)
    label_map = {item.get("case_id"): 3 if item.get("malignant", False) else 0 
                 for item in metadata if item.get("case_id")}
    
    all_cases = [d for d in os.listdir(root_dir) 
                 if d.startswith("case_") and os.path.isdir(os.path.join(root_dir, d))]
    all_cases.sort()
    random.shuffle(all_cases)
    
    split_idx = int(0.8 * len(all_cases))
    selected_cases = all_cases[:split_idx] if split_type == "train" else all_cases[split_idx:]

    for case_name in selected_cases:
        label = label_map.get(case_name)
        if label is not None:
            case_dir = os.path.join(root_dir, case_name)
            image_path = os.path.join(case_dir, "imaging.nii.gz")
            mask_path = os.path.join(case_dir, "segmentation.nii.gz")
            
            if os.path.exists(mask_path):
                records.append({
                    "image_path": image_path,
                    "mask_path": mask_path, 
                    "label": label,
                    "has_image": os.path.exists(image_path)
                })
    return records

# ==========================================
# CLASS DATASET CHÍNH
# ==========================================
class MedicalImageDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform
        self.dummy_warned = False 

    def __len__(self):
        return len(self.records)

    def extract_best_slice(self, img_path, mask_path, has_image):
        if has_image:
            img_vol = nib.load(img_path).get_fdata()
            best_slice_idx = img_vol.shape[0] // 2 
            mask_vol = nib.load(mask_path).get_fdata()
            
            if mask_vol.ndim == 3:
                tumor_pixels = np.sum(mask_vol > 0, axis=(1, 2))
                if np.any(tumor_pixels > 0):
                    best_slice_idx = int(np.argmax(tumor_pixels))
            
            img_slice = img_vol[best_slice_idx, :, :]
            img_slice = cv2.normalize(img_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            if not self.dummy_warned:
                print("[!] Bật chế độ Dummy: Dùng mask thay thế cho file imaging bị thiếu.")
                self.dummy_warned = True
            mask_vol = nib.load(mask_path).get_fdata()
            img_slice = (mask_vol[mask_vol.shape[0]//2, :, :] * 255).astype(np.uint8)

        return cv2.cvtColor(img_slice, cv2.COLOR_GRAY2RGB)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = self.extract_best_slice(record["image_path"], record["mask_path"], record["has_image"])
        if self.transform:
            image = self.transform(image=image)['image']
        return image, torch.tensor(record["label"], dtype=torch.long)

# ==========================================
# HÀM TRANSFORMS
# ==========================================
def get_transforms(img_size=224):
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return train_transform, val_transform
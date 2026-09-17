import json
import random
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict
from collections import defaultdict
from PIL import Image
import torchvision.transforms.functional as TF

class SARPreprocess:
    def __init__(self, mean=1.33, std=3.73, clip_lo=-13.8, clip_hi=4.97, use_clahe=True):
        self.mean = float(mean)
        self.std = float(std)
        self.clip_lo = float(clip_lo)
        self.clip_hi = float(clip_hi)
        self.use_clahe = use_clahe

    def __call__(self, img_pil):
        img_pil = img_pil.convert("L")
        
        if self.use_clahe:
            img_np = np.array(img_pil, dtype=np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_clahe = clahe.apply(img_np)
            img_pil = Image.fromarray(img_clahe)

        x = TF.to_tensor(img_pil)
        if x.max() <= 1.0: 
            x = x * 255.0
            
        x = torch.log(x + 1e-6)
        x = torch.clamp(x, self.clip_lo, self.clip_hi)
        x = (x - self.mean) / (self.std + 1e-6)
            
        return x

class FUSARDataset(torch.utils.data.Dataset):
    """Dataset FUSAR (Fase 2) - Fornisce SOLO Classificazione con bilanciamento aumentato"""
    def __init__(self, fusar_json: str, image_dir: str, augment: bool = False):
        with open(fusar_json, 'r') as f:
            self.coco_data = json.load(f)
        self.image_dir = Path(image_dir)
        self.augment = augment
        
        self.preprocess = SARPreprocess(mean=2.6584, std=0.7534, use_clahe=True)
        
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations = {ann['image_id']: ann for ann in self.coco_data['annotations']}
        categories = sorted(list({ann["category_name"] for ann in self.coco_data['annotations']}))
        self.class_to_idx = {c: i for i, c in enumerate(categories)}
        self.valid_ids = list(self.annotations.keys())

        class_counts = defaultdict(int)
        for ann in self.annotations.values():
            class_counts[ann['category_name']] += 1
        max_count = max(class_counts.values()) if class_counts else 1
        
        self.aug_probs = {}
        for cat in categories:
            freq = class_counts[cat] / max_count
            self.aug_probs[self.class_to_idx[cat]] = max(1.0 - freq, 0.2)
    
    def __len__(self) -> int: 
        return len(self.valid_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        img_id = self.valid_ids[idx]
        img_meta = self.images[img_id]
        ann = self.annotations[img_id]
        
        img_path = self.image_dir / img_meta['file_name']
        img_pil = Image.open(img_path).convert("L")
        
        cls = self.class_to_idx[ann["category_name"]]
        img_tensor = self.preprocess(img_pil)
        
        if self.augment:
            if random.random() > 0.5: 
                img_tensor = TF.hflip(img_tensor)
            if random.random() > 0.5: 
                img_tensor = TF.vflip(img_tensor)

            if random.random() < self.aug_probs[cls]:
                k_rot = random.choice([1, 2, 3]) 
                img_tensor = torch.rot90(img_tensor, k=k_rot, dims=[1, 2])

        return {
            'image': img_tensor,
            'image_id': torch.tensor([img_id]),
            'category_id': torch.tensor([cls], dtype=torch.long)
        }

class HRSIDDataset(torch.utils.data.Dataset):
    """Dataset HRSID (Fase 1) - Fornisce Bounding Boxes E Dimensioni"""
    def __init__(self, hrsid_json: str, image_dir: str, augment: bool = False):
        with open(hrsid_json, 'r') as f:
            self.coco_data = json.load(f)
        self.image_dir = Path(image_dir)
        self.augment = augment
        
        self.preprocess = SARPreprocess(mean=3.5796, std=0.7026, use_clahe=True)
        
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations_by_image = defaultdict(list)
        
        self.max_L = 400.0 
        self.max_W = 100.0

        for ann in self.coco_data['annotations']:
            self.annotations_by_image[ann['image_id']].append(ann)
        self.valid_ids = [img_id for img_id in self.images.keys() if img_id in self.annotations_by_image]
    
    def __len__(self) -> int: 
        return len(self.valid_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        img_id = self.valid_ids[idx]
        img_meta = self.images[img_id]
        anns = self.annotations_by_image[img_id]
        
        img_path = self.image_dir / img_meta['file_name']
        img_pil = Image.open(img_path).convert("L")
        W, H = img_pil.size
        
        img_tensor = self.preprocess(img_pil)
        
        boxes, dimensions = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            l_val = ann.get('length', 0.0) / self.max_L
            w_val = ann.get('width', 0.0) / self.max_W
            dimensions.append([l_val, w_val])
            
        boxes = torch.tensor(boxes, dtype=torch.float32)
        dimensions = torch.tensor(dimensions, dtype=torch.float32)
        
        if self.augment and boxes.numel() > 0:
            if random.random() > 0.5:
                img_tensor = TF.hflip(img_tensor)
                boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
            if random.random() > 0.5:
                img_tensor = TF.vflip(img_tensor)
                boxes[:, [1, 3]] = H - boxes[:, [3, 1]]

        return {
            'image': img_tensor,
            'image_id': torch.tensor([img_id]),
            'boxes': boxes,
            'dimensions': dimensions,
            'labels': torch.ones((len(boxes),), dtype=torch.int64)
        }

import json
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

def compute_single_dataset_stats(name, json_path, img_dir, clip_lo=-13.8, clip_hi=4.97):
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    num_pixels = 0

    with open(json_path, 'r') as f:
        coco_data = json.load(f)
        
    img_dir_path = Path(img_dir)
    print(f"\nCalcolando le statistiche per {name}...")
    
    # Prendi solo le immagini per evitare KeyError
    images_list = coco_data.get('images', [])
    if not images_list:
        print(f"Nessuna immagine trovata nel JSON di {name}!")
        return None, None

    for img_meta in tqdm(images_list):
        img_path = img_dir_path / img_meta['file_name']
        if not img_path.exists():
            continue
            
        img_pil = Image.open(img_path).convert("L")
        
        # 1. CLAHE
        img_np = np.array(img_pil, dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_clahe = clahe.apply(img_np)
        img_pil = Image.fromarray(img_clahe)
        
        # 2. To Tensor e Scalatura
        x = TF.to_tensor(img_pil)
        if x.max() <= 1.0: 
            x = x * 255.0
            
        # 3. Trasformazioni SAR
        x = torch.log(x + 1e-6)
        x = torch.clamp(x, clip_lo, clip_hi)
        
        # Accumulo per Mean e Std
        # x.sum() restituisce la somma di tutti i pixel
        pixel_sum += float(x.sum().item())
        pixel_sq_sum += float((x ** 2).sum().item())
        num_pixels += x.numel()

    if num_pixels == 0:
        print(f"Nessuna immagine valida trovata nella cartella per {name}.")
        return None, None

    # Calcolo finale
    mean = pixel_sum / num_pixels
    variance = (pixel_sq_sum / num_pixels) - (mean ** 2)
    std = float(np.sqrt(max(variance, 0.0)))
    
    return mean, std

if __name__ == "__main__":
    datasets = [
        {
            "name": "HRSID",
            "json_path": "/root/dark-vessel-paltrinieri/HRSID_pipeline/HRSID_JPG/HRSID_JPG/annotations/train2017.json",
            "img_dir": "/root/dark-vessel-paltrinieri/HRSID_pipeline/HRSID_JPG/HRSID_JPG/JPEGImages"
        },
        {
            "name": "FUSAR",
            "json_path": "/root/dark-vessel-paltrinieri/ICIP_vessel_detection_regression/datasets/FUSAR/train/annotations.json",
            "img_dir": "/root/dark-vessel-paltrinieri/ICIP_vessel_detection_regression/datasets/FUSAR/train/images"
        }
    ]
    
    results = {}
    for ds in datasets:
        m, s = compute_single_dataset_stats(ds["name"], ds["json_path"], ds["img_dir"])
        results[ds["name"]] = {"mean": m, "std": s}
        
    print("\n" + "="*30)
    print(" RISULTATI FINALI")
    print("="*30)
    for name, stats in results.items():
        if stats["mean"] is not None:
            print(f"{name}:")
            print(f"  Mean = {stats['mean']:.4f}")
            print(f"  Std  = {stats['std']:.4f}\n")
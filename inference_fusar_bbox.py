# inference_fusar_bbox.py
import torch
import json
import argparse
import math
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

from architecture.model import HybridVesselModel
from preprocessing import SARPreprocess

def apply_clahe(pil_img, clip_limit=2.0, tile_grid_size=(8,8)):
    """Applica CLAHE per migliorare il contrasto locale senza bruciare l'immagine."""
    img_np = np.array(pil_img, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_clahe = clahe.apply(img_np)
    return Image.fromarray(img_clahe)

def filter_by_centroid(boxes, scores, img_h, img_w, max_dist_ratio=0.35):
    """Scarta le BBox il cui centroide è troppo distante dal centro dell'immagine originale."""
    cx, cy = img_w / 2.0, img_h / 2.0
    max_dist = max(img_w, img_h) * max_dist_ratio
    
    valid_boxes = []
    valid_scores = []
    
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        bcx, bcy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dist = math.sqrt((bcx - cx)**2 + (bcy - cy)**2)
        
        if dist <= max_dist:
            valid_boxes.append(boxes[i])
            valid_scores.append(scores[i])
            
    if not valid_boxes:
        return np.empty((0, 4)), np.array([])
        
    return np.array(valid_boxes), np.array(valid_scores)

def boxes_intersect(b1, b2, margin=20):
    if b1[2] + margin < b2[0] or b1[0] - margin > b2[2]: return False
    if b1[3] + margin < b2[1] or b1[1] - margin > b2[3]: return False
    return True

def merge_overlapping_boxes_and_get_central(boxes, scores, img_h, img_w, score_threshold=0.1, margin=20):
    """Fonde le box connesse e seleziona quella più vicina al centro."""
    valid_boxes = [b.tolist() for b, s in zip(boxes, scores) if s >= score_threshold]
    if not valid_boxes: return None

    groups = []
    for box in valid_boxes:
        overlapping_groups_indices = []
        for i, group in enumerate(groups):
            if any(boxes_intersect(box, gb, margin) for gb in group):
                overlapping_groups_indices.append(i)
                
        if not overlapping_groups_indices:
            groups.append([box])
        else:
            first_idx = overlapping_groups_indices[0]
            groups[first_idx].append(box)
            for i in reversed(overlapping_groups_indices[1:]):
                groups[first_idx].extend(groups.pop(i))

    cx, cy = img_w / 2.0, img_h / 2.0
    best_merged_box = None
    min_dist = float('inf')

    for group in groups:
        min_x1 = min(b[0] for b in group)
        min_y1 = min(b[1] for b in group)
        max_x2 = max(b[2] for b in group)
        max_y2 = max(b[3] for b in group)
        
        bcx, bcy = (min_x1 + max_x2) / 2.0, (min_y1 + max_y2) / 2.0
        dist = math.sqrt((bcx - cx)**2 + (bcy - cy)**2)
        
        if dist < min_dist:
            min_dist = dist
            best_merged_box = [min_x1, min_y1, max_x2, max_y2]

    return best_merged_box

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_hrsid.pt')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--num_images', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--threshold', type=float, default=0.05, help='Soglia di confidenza accettazione')
    parser.add_argument('--margin', type=int, default=20, help='Tolleranza fusione in pixel originali')
    args = parser.parse_args()

    out_dir = Path('results/mini-inference-multiscale-test')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.config, 'r') as f:
        config = json.load(f)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = HybridVesselModel(num_classes=config['model']['num_classes'], pretrained_backbone=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    preprocess = SARPreprocess()
    json_path = config['paths'][f'fusar_{args.split}_json']
    img_dir = Path(config['paths'][f'fusar_{args.split}_images'])
    
    with open(json_path, 'r') as f:
        fusar_data = json.load(f)
    
    # Definiamo le scale per la Multi-Scale Inference
    scales = [1.0, 0.5, 0.25]
    
    processed = 0
    with torch.no_grad():
        for img_meta in fusar_data['images']:
            if processed >= args.num_images: break
                
            img_path = img_dir / img_meta['file_name']
            if not img_path.exists(): continue
                
            # 1. Caricamento immagine originale
            img_pil_gray = Image.open(img_path).convert("L")
            W, H = img_pil_gray.size
            
            # Immagine di output (pura originale per disegno finale)
            orig_pil = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(orig_pil)
            
            # 2. Applichiamo il CLAHE per uniformare il contrasto (pre-processing per la rete)
            img_clahe = apply_clahe(img_pil_gray)
            
            all_boxes = []
            all_scores = []
            
            # 3. Multi-Scale Inference Loop
            for scale in scales:
                # Resize dell'immagine preprocessata col CLAHE
                new_W, new_H = int(W * scale), int(H * scale)
                img_scaled = img_clahe.resize((new_W, new_H), Image.Resampling.BILINEAR)
                
                # Inferenza sulla scala corrente
                img_tensor = preprocess(img_scaled).to(device)
                detections = model.detector([img_tensor])[0]
                
                boxes_scaled = detections['boxes'].cpu().numpy()
                scores_scaled = detections['scores'].cpu().numpy()
                
                # Riportiamo le coordinate alla scala ORIGINALE
                if len(boxes_scaled) > 0:
                    boxes_original_scale = boxes_scaled / scale
                    
                    # Accumuliamo i risultati di tutte le scale
                    all_boxes.extend(boxes_original_scale.tolist())
                    all_scores.extend(scores_scaled.tolist())
            
            # 4. Post-processing sull'insieme di TUTTE le predizioni trovate a diverse scale
            if len(all_boxes) > 0:
                all_boxes_np = np.array(all_boxes)
                all_scores_np = np.array(all_scores)
                
                # Eliminiamo detection palesemente fuori centro
                filtered_boxes, filtered_scores = filter_by_centroid(all_boxes_np, all_scores_np, H, W)
                
                if len(filtered_boxes) > 0:
                    # Disegniamo tutte le box "parziali" trovate ai vari livelli (in verde sottile)
                    for i in range(len(filtered_boxes)):
                        if filtered_scores[i] >= args.threshold:
                            x1, y1, x2, y2 = filtered_boxes[i]
                            draw.rectangle([x1, y1, x2, y2], outline="green", width=1)
                    
                    # 5. Merge intelligente (Componenti Connesse + Selezione del più centrale)
                    merged_box = merge_overlapping_boxes_and_get_central(
                        filtered_boxes, filtered_scores, H, W, 
                        score_threshold=args.threshold, 
                        margin=args.margin
                    )
                    
                    # Disegniamo la macro-box finale (in rosso spesso)
                    if merged_box is not None:
                        mx1, my1, mx2, my2 = merged_box
                        draw.rectangle([mx1, my1, mx2, my2], outline="red", width=3)
                        draw.text((mx1, max(0, my1 - 15)), "MERGED", fill="red")

            save_path = out_dir / f"infer_{img_meta['file_name']}"
            orig_pil.save(save_path)
            processed += 1

    print(f"Finito! Controlla la cartella {out_dir}")

if __name__ == "__main__":
    main()
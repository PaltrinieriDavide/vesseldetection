# test.py
import torch
import json
import argparse
import logging
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from torchvision.ops import box_iou

from architecture.model import HybridVesselModel
from data import create_dataloaders

def setup_logging(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh = logging.FileHandler(log_file)
    sh = logging.StreamHandler()
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger

def get_resolution_from_filename(filename):
    """Estrae la risoluzione spaziale (m/px) in base alle regole fornite per HRSID."""
    match = re.search(r'P(\d{4})', filename)
    if match:
        num = int(match.group(1))
        if num in[124, 125, 130, 131]:
            return 0.5
        elif num in[123, 128]:
            return 1.0
        elif (1 <= num <= 122) or num in [126, 127, 129] or (132 <= num <= 136):
            return 3.0
    return 3.0  # Fallback di sicurezza

def build_ground_truth_map(config):
    """
    Legge i JSON e mappa ogni Image_ID alle dimensioni reali (metri e pixel) NON NORMALIZZATE.
    """
    gt_map = {}
    json_paths =[
        config.get('paths', {}).get('hrsid_test_json', ''),
        config.get('paths', {}).get('fusar_test_json', '')
    ]

    for jp in json_paths:
        if not jp: continue
        p = Path(jp)
        if p.exists():
            with open(p, 'r') as f:
                data = json.load(f)

            # Dizionario di supporto per trovare il nome file
            img_dict = {img['id']: img.get('file_name', '') for img in data.get('images', [])}

            for ann in data.get('annotations',[]):
                img_id = ann['image_id']
                if img_id not in gt_map:
                    gt_map[img_id] = {
                        'filename': img_dict.get(img_id, ''),
                        'anns':[]
                    }

                # Salviamo i VERI valori in metri e pixel del JSON
                gt_map[img_id]['anns'].append({
                    'l_m': float(ann.get('length', 0)),
                    'w_m': float(ann.get('width', 0)),
                    'l_p': float(ann.get('length_pixel', 0)),
                    'w_p': float(ann.get('width_pixel', 0))
                })
    return gt_map

class TestMetricsTracker:
    def __init__(self, gt_map):
        self.gt_map = gt_map
        self.ious, self.mape_widths, self.mape_lengths = [], [],[]

        self.mae_widths_pixel, self.mae_lengths_pixel = [],[]

        self.num_samples, self.num_detections = 0, 0
        self.tp, self.fp, self.fn = 0, 0, 0
        self.correct_cls, self.total_cls = 0, 0

    def update(self, preds, targets, task):
        self.num_samples += len(targets)

        # === 1. Classificazione ===
        if task in['cls', 'multi']:
            valid_preds, valid_targets = [],[]
            idx = 0
            for t in targets:
                n_boxes = len(t['boxes']) if 'boxes' in t else 1
                if 'category_id' in t:
                    valid_preds.append(preds['logits'][idx:idx+n_boxes])
                    valid_targets.append(t['category_id'].repeat(n_boxes))
                idx += n_boxes
            if valid_preds:
                p_cls = torch.argmax(torch.cat(valid_preds), dim=1).cpu().numpy()
                t_cls = torch.cat(valid_targets).cpu().numpy()
                self.correct_cls += (p_cls == t_cls).sum()
                self.total_cls += len(t_cls)

        # === 2. Detection & Regressione ===
        if task in ['det_reg', 'multi'] and 'detections' in preds:
            idx = 0
            for p_det, t in zip(preds['detections'], targets):
                n_raw_preds = len(p_det['boxes'])

                img_id = t.get('image_id', None)
                if isinstance(img_id, torch.Tensor): img_id = img_id.item()

                if 'boxes' in t:
                    gt_boxes = t['boxes']
                    keep_idx = p_det['scores'] > 0.5
                    pred_boxes = p_det['boxes'][keep_idx]

                    if 'dimensions' in t:
                        raw_dims = preds['dimensions'][idx : idx + n_raw_preds]
                        pred_dims = raw_dims[keep_idx]

                    n_valid_preds = len(pred_boxes)
                    self.num_detections += n_valid_preds

                    if len(gt_boxes) == 0:
                        self.fp += n_valid_preds
                    elif n_valid_preds == 0:
                        self.fn += len(gt_boxes)
                    else:
                        self.ious.append(box_iou(pred_boxes, gt_boxes).max().item())
                        iou_mat = box_iou(pred_boxes, gt_boxes)
                        matched_gt = set()

                        for pred_idx in range(n_valid_preds):
                            best_iou, best_gt = iou_mat[pred_idx].max(dim=0)

                            if best_iou > 0.5 and best_gt.item() not in matched_gt:
                                self.tp += 1
                                matched_gt.add(best_gt.item())

                                if 'dimensions' in t:
                                    pr_l, pr_w = pred_dims[pred_idx].detach().cpu().numpy()
                                    gt_l, gt_w = t['dimensions'][best_gt.item()].cpu().numpy()

                                    # Errore relativo percentuale (invariante alla scala, rimane sempre uguale)
                                    self.mape_lengths.append(np.abs(gt_l - pr_l) / (gt_l + 1e-6) * 100)
                                    self.mape_widths.append(np.abs(gt_w - pr_w) / (gt_w + 1e-6) * 100)

                                    # --- Calcolo MAE Esatto in Pixel aggirando la Normalizzazione ---
                                    best_json_ann = None
                                    if img_id is not None and img_id in self.gt_map:
                                        anns = self.gt_map[img_id]['anns']
                                        target_idx = best_gt.item()

                                        # Ricerca la label originale esatta che corrisponde a questa nave
                                        if target_idx < len(anns):
                                            best_json_ann = anns[target_idx]

                                            # Sicurezza: il rapporto geometrico è sensato?
                                            true_ratio = best_json_ann['l_m'] / max(best_json_ann['w_m'], 1e-6)
                                            pred_ratio = float(gt_l) / max(float(gt_w), 1e-6)

                                            if abs(true_ratio - pred_ratio) > 0.5:
                                                # Fallback: trova la nave col rapporto L/W più simile nel JSON
                                                min_diff = float('inf')
                                                for ann in anns:
                                                    tr = ann['l_m'] / max(ann['w_m'], 1e-6)
                                                    diff = abs(tr - pred_ratio)
                                                    if diff < min_diff:
                                                        min_diff = diff
                                                        best_json_ann = ann

                                    if best_json_ann and best_json_ann['l_m'] > 0:
                                        true_l_m = best_json_ann['l_m']
                                        true_w_m = best_json_ann['w_m']

                                        # Estrae la risoluzione in m/px
                                        filename = self.gt_map[img_id]['filename']
                                        res = get_resolution_from_filename(filename)

                                        # Estrae i veri pixel (se non ci sono nel JSON li calcola)
                                        true_l_p = best_json_ann['l_p'] if best_json_ann['l_p'] > 0 else true_l_m / res
                                        true_w_p = best_json_ann['w_p'] if best_json_ann['w_p'] > 0 else true_w_m / res

                                        # 1. Ricaviamo i VERI METRI PREDETTI disfacendo la normalizzazione del dataloader
                                        scale_l = true_l_m / max(float(gt_l), 1e-6)
                                        scale_w = true_w_m / max(float(gt_w), 1e-6)

                                        pr_l_meters = pr_l * scale_l
                                        pr_w_meters = pr_w * scale_w

                                        # 2. Convertiamo i veri metri predetti in pixel
                                        pr_l_px = pr_l_meters / res
                                        pr_w_px = pr_w_meters / res

                                        # 3. Confronto assoluto con la label originaria (Pixel)
                                        self.mae_lengths_pixel.append(np.abs(true_l_p - pr_l_px))
                                        self.mae_widths_pixel.append(np.abs(true_w_p - pr_w_px))

                            else:
                                self.fp += 1

                        self.fn += len(gt_boxes) - len(matched_gt)

                idx += n_raw_preds

    def get_metrics(self):
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics = {
            'accuracy': self.correct_cls / self.total_cls if self.total_cls > 0 else 0.0,
            'mape_length': np.mean(self.mape_lengths) if self.mape_lengths else 0.0,
            'mape_width': np.mean(self.mape_widths) if self.mape_widths else 0.0,
            'iou': np.mean(self.ious) if self.ious else 0.0,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'detection_rate': self.num_detections / max(self.num_samples, 1)
        }

        if self.mae_lengths_pixel:
            metrics['mae_length_pixel'] = np.mean(self.mae_lengths_pixel)
            metrics['mae_width_pixel'] = np.mean(self.mae_widths_pixel)

        return metrics

def test_epoch(model, dataloader, device, task, gt_map):
    model.eval()
    metrics = TestMetricsTracker(gt_map)
    with torch.no_grad():
        for images, targets in dataloader:

            for i in range(len(images)):
                single_img = [images[i].to(device)]
                single_tgt =[{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in targets[i].items()}]

                with torch.autocast(device_type='cuda'):
                    preds, _ = model(single_img, phase=task)

                metrics.update(preds, single_tgt, task)

    return metrics.get_metrics()

def log_det_metrics(logger, metrics):
    logger.info(f"  Precision:      {metrics['precision']:.4f}")
    logger.info(f"  Recall:         {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:       {metrics['f1']:.4f}")
    logger.info(f"  IoU (BBOX):     {metrics['iou']:.4f}")

    logger.info(f"  MAPE Length (m):{metrics['mape_length']:.2f}%")
    logger.info(f"  MAPE Width (m): {metrics['mape_width']:.2f}%")

    if 'mae_length_pixel' in metrics:
        # Stampiamo lo stesso MAPE per mostrare che l'errore % resta invariato tra m e px
        logger.info(f"  MAPE Length px: {metrics['mape_length']:.2f}%")
        logger.info(f"  MAPE Width px:  {metrics['mape_width']:.2f}%")

        # L'errore ASSOLUTO reale in pixel
        logger.info(f"  MAE Length px:  {metrics['mae_length_pixel']:.2f} px")
        logger.info(f"  MAE Width px:   {metrics['mae_width_pixel']:.2f} px")

    logger.info(f"  Det Rate:       {metrics['detection_rate']:.2f} boxes/img")

def log_cls_metrics(logger, metrics):
    logger.info(f"  Accuracy (cls): {metrics['accuracy']:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    with open(args.config, 'r') as f: config = json.load(f)

    logger = setup_logging(config['paths']['logs_dir'])
    device = args.device if torch.cuda.is_available() else 'cpu'

    # 1. Costruiamo la mappa Esatta leggendo i JSON prima di far partire i test
    gt_map = build_ground_truth_map(config)

    model = HybridVesselModel(num_classes=config['model']['num_classes'], pretrained_backbone=False).to(device)
    dataloaders = create_dataloaders(config)
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])

    checkpoints_map = {
        'Fase 1 (HRSID Model)': checkpoint_dir / 'best_hrsid.pt',
        'Fase 2 (FUSAR Model)': checkpoint_dir / 'best_fusar.pt',
        'Fase 3 (Combined Model)': checkpoint_dir / 'final_hybrid_model.pt'
    }

    for phase_name, ckpt_path in checkpoints_map.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"  Testing {phase_name}...")
        logger.info(f"{'='*60}")

        if not ckpt_path.exists():
            logger.error(f"  [!] Checkpoint not found: {ckpt_path.name}. Skipping.")
            continue

        logger.info(f"  --> Loading weights: {ckpt_path.name}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

        if 'HRSID' in phase_name:
            metrics = test_epoch(model, dataloaders['test_hrsid'], device, 'det_reg', gt_map)
            log_det_metrics(logger, metrics)

        elif 'FUSAR' in phase_name:
            metrics = test_epoch(model, dataloaders['test_fusar'], device, 'cls', gt_map)
            log_cls_metrics(logger, metrics)

        elif 'Combined' in phase_name:
            logger.info("\n  --- Evaluating on HRSID Test Set (Detection Task) ---")
            metrics_hrsid = test_epoch(model, dataloaders['test_hrsid'], device, 'det_reg', gt_map)
            log_det_metrics(logger, metrics_hrsid)

            logger.info("\n  --- Evaluating on FUSAR Test Set (Classification Task) ---")
            metrics_fusar = test_epoch(model, dataloaders['test_fusar'], device, 'cls', gt_map)
            log_cls_metrics(logger, metrics_fusar)

if __name__ == '__main__':
    main()
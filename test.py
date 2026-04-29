# test.py
import torch
import json
import argparse
import logging
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

class TestMetricsTracker:
    def __init__(self):
        self.ious, self.mape_widths, self.mape_lengths = [], [],[]
        self.num_samples, self.num_detections = 0, 0
        self.tp, self.fp, self.fn = 0, 0, 0
        self.correct_cls, self.total_cls = 0, 0

    def update(self, preds, targets, task):
        self.num_samples += len(targets)
        
        # === 1. Classificazione ===
        if task in ['cls', 'multi']:
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
                                    
                                    self.mape_lengths.append(np.abs(gt_l - pr_l) / (gt_l + 1e-6) * 100)
                                    self.mape_widths.append(np.abs(gt_w - pr_w) / (gt_w + 1e-6) * 100)
                            else:
                                self.fp += 1
                                
                        self.fn += len(gt_boxes) - len(matched_gt)
                        
                idx += n_raw_preds

    def get_metrics(self):
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': self.correct_cls / self.total_cls if self.total_cls > 0 else 0.0,
            'mape_length': np.mean(self.mape_lengths) if self.mape_lengths else 0.0,
            'mape_width': np.mean(self.mape_widths) if self.mape_widths else 0.0,
            'iou': np.mean(self.ious) if self.ious else 0.0,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'detection_rate': self.num_detections / max(self.num_samples, 1)
        }

def test_epoch(model, dataloader, device, task):
    model.eval()
    metrics = TestMetricsTracker()
    with torch.no_grad():
        for images, targets in dataloader:
            
            # Scorriamo UN sample alla volta per eliminare l'artefatto da padding 
            # e replicare ESATTAMENTE la mini-inferenza del training.
            for i in range(len(images)):
                single_img = [images[i].to(device)]
                single_tgt =[{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in targets[i].items()}]
                
                with torch.autocast(device_type='cuda'):
                    # Chiamiamo il modello OMETTENDO i targets per simulare
                    # l'inferenza pura basata sulle detection reali
                    preds, _ = model(single_img, phase=task)
                    
                metrics.update(preds, single_tgt, task)
                
    return metrics.get_metrics()

def log_det_metrics(logger, metrics):
    logger.info(f"  Precision:      {metrics['precision']:.4f}")
    logger.info(f"  Recall:         {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:       {metrics['f1']:.4f}")
    logger.info(f"  IoU (BBOX):     {metrics['iou']:.4f}")
    logger.info(f"  MAPE Length:    {metrics['mape_length']:.2f}%")
    logger.info(f"  MAPE Width:     {metrics['mape_width']:.2f}%")
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
            metrics = test_epoch(model, dataloaders['test_hrsid'], device, 'det_reg')
            log_det_metrics(logger, metrics)
            
        elif 'FUSAR' in phase_name:
            metrics = test_epoch(model, dataloaders['test_fusar'], device, 'cls')
            log_cls_metrics(logger, metrics)
            
        elif 'Combined' in phase_name:
            # Per il Combined, usiamo i pesi finali ma li passiamo chirurgicamente 
            # sui due dataset con i loro task specifici (Inferenza Reale).
            logger.info("\n  --- Evaluating on HRSID Test Set (Detection Task) ---")
            metrics_hrsid = test_epoch(model, dataloaders['test_hrsid'], device, 'det_reg')
            log_det_metrics(logger, metrics_hrsid)
            
            logger.info("\n  --- Evaluating on FUSAR Test Set (Classification Task) ---")
            metrics_fusar = test_epoch(model, dataloaders['test_fusar'], device, 'cls')
            log_cls_metrics(logger, metrics_fusar)

if __name__ == '__main__':
    main()
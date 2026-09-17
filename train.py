import torch
import torch.optim as optim
import json
import logging
import re
from pathlib import Path
from datetime import datetime
import argparse
import numpy as np
from collections import defaultdict
from torchvision.ops import box_iou
from torch.utils.tensorboard import SummaryWriter

from architecture.model import HybridVesselModel
from data import create_dataloaders

ACCUMULATION_STEPS = 32
GRAD_CLIP = 1.0

def get_resolution_from_filename(filename):
    match = re.search(r'P(\d{4})', filename)
    if match:
        num = int(match.group(1))
        if num in [124, 125, 130, 131]:
            return 0.5
        elif num in [123, 128]:
            return 1.0
        elif (1 <= num <= 122) or num in [126, 127, 129] or (132 <= num <= 136):
            return 3.0
    return 3.0

def build_ground_truth_map(config):
    gt_map = {}
    json_paths = [
        config.get('paths', {}).get('hrsid_val_json', ''),
        config.get('paths', {}).get('fusar_val_json', '')
    ]
    for jp in json_paths:
        if not jp: continue
        p = Path(jp)
        if p.exists():
            with open(p, 'r') as f:
                data = json.load(f)
            img_dict = {img['id']: img.get('file_name', '') for img in data.get('images', [])}
            for ann in data.get('annotations', []):
                img_id = ann['image_id']
                if img_id not in gt_map:
                    gt_map[img_id] = {'filename': img_dict.get(img_id, ''), 'anns': []}
                gt_map[img_id]['anns'].append({
                    'l_m': float(ann.get('length', 0)),
                    'w_m': float(ann.get('width', 0)),
                    'l_p': float(ann.get('length_pixel', 0)),
                    'w_p': float(ann.get('width_pixel', 0))
                })
    return gt_map

def setup_logging(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.propagate = False

    return logger

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else -float('inf')
        self.early_stop = False

    def __call__(self, current_score):
        if self.mode == 'min':
            is_better = current_score < self.best_score - self.min_delta
        else:
            is_better = current_score > self.best_score + self.min_delta

        if is_better:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
class MetricsTracker:
    def __init__(self, phase_name='det_reg', gt_map=None):
        self.phase_name = phase_name
        self.gt_map = gt_map or {}
        self.reset()

    def reset(self):
        self.losses = defaultdict(list)
        self.accuracies, self.mape_widths, self.mape_lengths, self.ious = [], [], [], []
        self.mae_widths_pixel, self.mae_lengths_pixel = [], []
        
        self.correct_per_class = defaultdict(int)
        self.total_per_class = defaultdict(int)

        self.num_samples = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, losses_dict, out_dict, targets, task='det_reg'):
        for k, v in losses_dict.items():
            if isinstance(v, torch.Tensor): self.losses[k].append(v.item())

        self.num_samples += len(targets)
        if not out_dict: return

        
        if task in ['det_reg', 'multi'] and 'detections' in out_dict:
            idx = 0
            for i, (p, t) in enumerate(zip(out_dict['detections'], targets)):
                
                b_forward = out_dict['boxes'][i]
                n_forward_boxes = b_forward.shape[0] if b_forward.numel() > 0 else 0

                if 'boxes' in t:
                    gt_boxes = t['boxes']
                    keep_idx = p['scores'] > 0.5
                    pred_boxes = p['boxes'][keep_idx]

                    
                    
                    can_do_pixel = False
                    if 'dimensions' in t and 'dimensions' in out_dict and n_forward_boxes == len(p['boxes']):
                        raw_dims = out_dict['dimensions'][idx : idx + n_forward_boxes]
                        pred_dims = raw_dims[keep_idx]
                        can_do_pixel = True

                    if len(gt_boxes) == 0:
                        self.fp += len(pred_boxes)
                        idx += n_forward_boxes
                        continue
                    if len(pred_boxes) == 0:
                        self.fn += len(gt_boxes)
                        idx += n_forward_boxes
                        continue

                    if len(p['boxes']) > 0 and len(gt_boxes) > 0:
                        self.ious.append(box_iou(p['boxes'], gt_boxes).max().item())

                    ious_mat = box_iou(pred_boxes, gt_boxes)
                    matched_gt = set()

                    for pred_idx in range(len(pred_boxes)):
                        best_iou, best_gt = ious_mat[pred_idx].max(dim=0)
                        if best_iou > 0.5 and best_gt.item() not in matched_gt:
                            self.tp += 1
                            matched_gt.add(best_gt.item())
                            
                            
                            if can_do_pixel and self.gt_map:
                                img_id = t.get('image_id', None)
                                if isinstance(img_id, torch.Tensor): img_id = img_id.item()
                                
                                pr_l, pr_w = pred_dims[pred_idx].detach().cpu().numpy()
                                gt_l, gt_w = t['dimensions'][best_gt.item()].cpu().numpy()
                                
                                best_json_ann = None
                                if img_id is not None and img_id in self.gt_map:
                                    anns = self.gt_map[img_id]['anns']
                                    target_idx = best_gt.item()
                                    if target_idx < len(anns):
                                        best_json_ann = anns[target_idx]
                                        true_ratio = best_json_ann['l_m'] / max(best_json_ann['w_m'], 1e-6)
                                        pred_ratio = float(gt_l) / max(float(gt_w), 1e-6)
                                        if abs(true_ratio - pred_ratio) > 0.5:
                                            min_diff = float('inf')
                                            for ann in anns:
                                                tr = ann['l_m'] / max(ann['w_m'], 1e-6)
                                                diff = abs(tr - pred_ratio)
                                                if diff < min_diff:
                                                    min_diff, best_json_ann = diff, ann
                                                    
                                if best_json_ann and best_json_ann['l_m'] > 0:
                                    true_l_m = best_json_ann['l_m']
                                    true_w_m = best_json_ann['w_m']
                                    filename = self.gt_map[img_id]['filename']
                                    res = get_resolution_from_filename(filename)
                                    
                                    true_l_p = best_json_ann['l_p'] if best_json_ann['l_p'] > 0 else true_l_m / res
                                    true_w_p = best_json_ann['w_p'] if best_json_ann['w_p'] > 0 else true_w_m / res
                                    
                                    scale_l = true_l_m / max(float(gt_l), 1e-6)
                                    scale_w = true_w_m / max(float(gt_w), 1e-6)
                                    
                                    pr_l_meters = pr_l * scale_l
                                    pr_w_meters = pr_w * scale_w
                                    
                                    pr_l_px = pr_l_meters / res
                                    pr_w_px = pr_w_meters / res
                                    
                                    self.mae_lengths_pixel.append(np.abs(true_l_p - pr_l_px))
                                    self.mae_widths_pixel.append(np.abs(true_w_p - pr_w_px))
                            

                        else:
                            self.fp += 1

                    self.fn += len(gt_boxes) - len(matched_gt)
                idx += n_forward_boxes

        
        
        if task in ['cls', 'det_reg', 'multi']:
            valid_cls_preds, valid_cls_targs = [], []
            valid_reg_preds, valid_reg_targs = [], []
            idx = 0

            for i, t in enumerate(targets):
                b = out_dict['boxes'][i]
                num_preds = b.shape[0] if b.numel() > 0 else 0

                if num_preds > 0:
                    if 'category_id' in t: 
                        valid_cls_preds.append(out_dict['logits'][idx:idx+num_preds])
                        valid_cls_targs.append(t['category_id'].repeat(num_preds))
                    if 'dimensions' in t: 
                        valid_reg_preds.append(out_dict['dimensions'][idx:idx+num_preds])
                        valid_reg_targs.append(t['dimensions'])

                idx += num_preds

            if valid_cls_preds:
                preds_cls = torch.argmax(torch.cat(valid_cls_preds), dim=1).cpu().numpy()
                true_cls = torch.cat(valid_cls_targs).cpu().numpy()
                self.accuracies.append((preds_cls == true_cls).mean())
                
                
                for p_c, t_c in zip(preds_cls, true_cls):
                    self.correct_per_class[t_c] += (p_c == t_c)
                    self.total_per_class[t_c] += 1

            if valid_reg_preds:
                pred_dims = torch.cat(valid_reg_preds).detach().cpu().numpy()
                true_dims = torch.cat(valid_reg_targs).cpu().numpy()
                mape_l = np.mean(np.abs(true_dims[:, 0] - pred_dims[:, 0]) / (true_dims[:, 0] + 1e-6)) * 100
                mape_w = np.mean(np.abs(true_dims[:, 1] - pred_dims[:, 1]) / (true_dims[:, 1] + 1e-6)) * 100
                self.mape_lengths.append(min(mape_l, 1000.0))
                self.mape_widths.append(min(mape_w, 1000.0))
                
                
    def get_avg_metrics(self):
        metrics = {k: np.mean(v) if v else 0.0 for k, v in self.losses.items()}
        metrics['loss_total'] = sum(metrics.values())
        metrics['accuracy'] = np.mean(self.accuracies) if self.accuracies else 0.0
        metrics['mape_length'] = np.mean(self.mape_lengths) if self.mape_lengths else 0.0
        metrics['mape_width'] = np.mean(self.mape_widths) if self.mape_widths else 0.0
        metrics['iou'] = np.mean(self.ious) if self.ious else 0.0

        if self.mae_lengths_pixel:
            metrics['mae_length_pixel'] = np.mean(self.mae_lengths_pixel)
            metrics['mae_width_pixel'] = np.mean(self.mae_widths_pixel)

        
        for cls_idx in self.total_per_class.keys():
            if self.total_per_class[cls_idx] > 0:
                metrics[f'acc_cls_{cls_idx}'] = self.correct_per_class[cls_idx] / self.total_per_class[cls_idx]

        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1

        return metrics

def train_epoch(model, dataloader, optimizer, scaler, device, log_interval, epoch, logger, task, gt_map=None):
    model.train()
    metrics = MetricsTracker(phase_name=task, gt_map=gt_map)
    optimizer.zero_grad()

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = [im.to(device) for im in images]
        targets_gpu = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        with torch.autocast(device_type='cuda'):
            out, losses = model(images, targets_gpu, phase=task)
            if len(losses) == 0: continue
            loss = sum(l for l in losses.values()) / ACCUMULATION_STEPS

        scaler.scale(loss).backward()
        if ((batch_idx + 1) % ACCUMULATION_STEPS == 0) or (batch_idx + 1 == len(dataloader)):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if task in ['det_reg', 'multi']:
            model.eval()
            with torch.no_grad(): det = model.detector(images)
            out['detections'] = det
            model.train()

        metrics.update(losses, out, targets_gpu, task)

        if batch_idx % log_interval == 0:
            m = metrics.get_avg_metrics()
            logger.info(f"Training epoch {epoch} [{batch_idx}/{len(dataloader)}] | Loss: {m['loss_total']:.4f}")

    return metrics.get_avg_metrics()

@torch.no_grad()
def val_epoch(model, dataloader, device, epoch, logger, task, gt_map=None):
    model.train()
    loss_tracker = MetricsTracker(phase_name=task, gt_map=gt_map)
    for images, targets in dataloader:
        images = [im.to(device) for im in images]
        targets_gpu = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.autocast(device_type='cuda'):
            _, losses = model(images, targets_gpu, phase=task)
            if losses: loss_tracker.update(losses, {}, targets_gpu, task)

    model.eval()
    metric_tracker = MetricsTracker(phase_name=task, gt_map=gt_map)
    for images, targets in dataloader:
        images = [im.to(device) for im in images]
        targets_gpu = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.autocast(device_type='cuda'):
            preds, _ = model(images, targets_gpu, phase=task)
            metric_tracker.update({}, preds, targets_gpu, task)

    res = loss_tracker.get_avg_metrics()
    metric_res = metric_tracker.get_avg_metrics()
    
    res.update({k: v for k, v in metric_res.items() if k in ['iou', 'accuracy', 'mape_length', 'mape_width', 'precision', 'recall', 'f1', 'mae_length_pixel', 'mae_width_pixel'] or k.startswith('acc_cls_')})
    return res

def setup_phase(model, phase_config, logger, dataset):
    logger.info(f"\n{'='*80}\nTRAINING PHASE: {phase_config['name']} ({dataset.upper()})\n{'='*80}")
    if dataset == 'fusar':
        model.unfreeze_backbone_last_layers()
        logger.info("Backbone: partially unfrozen (layer 4 and FPN only).")
    elif phase_config.get('freeze_backbone', False):
        for p in model.backbone.parameters(): p.requires_grad = False
        logger.info("Backbone: fully frozen.")
    else:
        for p in model.backbone.parameters(): p.requires_grad = True
        logger.info("Backbone: fully unfrozen.")

    if phase_config.get('freeze_detection', False) or dataset == 'fusar':
        model.freeze_detection()
        logger.info("Detector: frozen.")
    else:
        model.unfreeze_detection()
        logger.info("Detector: unfrozen.")

    if dataset == 'hrsid':
        model.unfreeze_reg_head()
        model.freeze_cls_head()
        logger.info("Attribute heads: regression unfrozen; classification frozen.")
    elif dataset == 'fusar':
        model.freeze_reg_head()
        model.unfreeze_cls_head()
        logger.info("Attribute heads: regression frozen; classification unfrozen.")
    elif dataset == 'combined':
        if phase_config.get('freeze_classification', False):
            model.freeze_cls_head()
            logger.info("Attribute heads: classification frozen.")
        else:
            model.unfreeze_cls_head()
            logger.info("Attribute heads: classification unfrozen.")
            
        if phase_config.get('freeze_regression', False):
            model.freeze_reg_head()
            logger.info("Attribute heads: regression frozen.")
        else:
            model.unfreeze_reg_head()
            logger.info("Attribute heads: regression unfrozen.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    with open(args.config, 'r') as f: config = json.load(f)

    logger = setup_logging(config['paths']['logs_dir'])
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    gt_map = build_ground_truth_map(config)

    tb_dir = Path(config['paths']['logs_dir']) / f"tensorboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    logger.info(f"TensorBoard logs will be written to: {tb_dir}")

    model = HybridVesselModel(num_classes=config['model']['num_classes'], pretrained_backbone=config['model']['pretrained_backbone']).to(device)
    dataloaders = create_dataloaders(config)
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler()
    patience = config.get('common_training', {}).get('early_stopping_patience', 10)

    global_epoch_counter = 0

    for phase_name, phase_config in config['training_phases'].items():
        dataset = phase_config['dataset']
        task = {'hrsid': 'det_reg', 'fusar': 'cls', 'combined': 'multi'}[dataset]
        setup_phase(model, phase_config, logger, dataset)

        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=phase_config['learning_rate'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase_config['num_epochs'], eta_min=1e-6)
        early_stopping = EarlyStopping(patience=patience, min_delta=0.001)

        for epoch in range(1, phase_config['num_epochs'] + 1):
            global_epoch_counter += 1

            train_m = train_epoch(model, dataloaders[f'train_{dataset}'], optimizer, scaler, device, config['common_training']['log_interval'], epoch, logger, task, gt_map)

            if epoch % config['common_training']['val_interval'] == 0:
                val_m = val_epoch(model, dataloaders[f'val_{dataset}'], device, epoch, logger, task, gt_map)

                writer.add_scalars(f'Loss/{phase_name}', {
                    'Train': train_m['loss_total'],
                    'Val': val_m['loss_total']
                }, global_epoch_counter)

                if task in ['det_reg', 'multi']:
                    writer.add_scalar(f'Metrics/{phase_name}/IoU', val_m.get('iou', 0), global_epoch_counter)
                    writer.add_scalar(f'Metrics/{phase_name}/F1', val_m.get('f1', 0), global_epoch_counter)
                    writer.add_scalar(f'Metrics/{phase_name}/Precision', val_m.get('precision', 0), global_epoch_counter)
                    writer.add_scalar(f'Metrics/{phase_name}/Recall', val_m.get('recall', 0), global_epoch_counter)
                    writer.add_scalar(f'Metrics/{phase_name}/MAPE_L', val_m.get('mape_length', 0), global_epoch_counter)
                    writer.add_scalar(f'Metrics/{phase_name}/MAPE_W', val_m.get('mape_width', 0), global_epoch_counter)
                    if 'mae_length_pixel' in val_m:
                        writer.add_scalar(f'Metrics/{phase_name}/MAE_L_px', val_m.get('mae_length_pixel', 0), global_epoch_counter)
                        writer.add_scalar(f'Metrics/{phase_name}/MAE_W_px', val_m.get('mae_width_pixel', 0), global_epoch_counter)

                if task in ['cls', 'multi']:
                    writer.add_scalar(f'Metrics/{phase_name}/Accuracy', val_m.get('accuracy', 0), global_epoch_counter)
                    for k, v in val_m.items():
                        if k.startswith('acc_cls_'):
                            writer.add_scalar(f'Metrics/{phase_name}/Class_Accuracy/{k}', v, global_epoch_counter)
                    
                if task == 'det_reg':
                    log_str = f"Validation epoch {epoch} | Loss: {val_m['loss_total']:.4f} | IoU: {val_m['iou']:.4f} | Precision: {val_m['precision']:.4f} | Recall: {val_m['recall']:.4f} | F1: {val_m['f1']:.4f} | Length MAPE: {val_m['mape_length']:.2f}% | Width MAPE: {val_m['mape_width']:.2f}%"
                    if 'mae_length_pixel' in val_m:
                        log_str += f" | Length MAE (px): {val_m['mae_length_pixel']:.2f} | Width MAE (px): {val_m['mae_width_pixel']:.2f}"
                    logger.info(log_str)
                    target_score = val_m['f1'] + val_m['iou']
                
                elif task == 'cls':
                    acc_str = f"Validation epoch {epoch} | Loss: {val_m['loss_total']:.4f} | Accuracy: {val_m['accuracy']:.4f}"
                    for k, v in val_m.items():
                        if k.startswith('acc_cls_'):
                            acc_str += f" | {k}: {v:.4f}"
                    logger.info(acc_str)
                    target_score = val_m['accuracy']
                
                else: 
                    log_str = f"Validation epoch {epoch} | Loss: {val_m['loss_total']:.4f} | F1: {val_m['f1']:.4f} | Recall: {val_m['recall']:.4f} | Accuracy: {val_m['accuracy']:.4f}"
                    if 'mae_length_pixel' in val_m:
                        log_str += f" | Length MAE (px): {val_m['mae_length_pixel']:.2f} | Width MAE (px): {val_m['mae_width_pixel']:.2f}"
                    for k, v in val_m.items():
                        if k.startswith('acc_cls_'):
                            log_str += f" | {k}: {v:.4f}"
                    logger.info(log_str)
                    target_score = (val_m['f1'] + val_m['accuracy']) / 2.0

                early_stopping(target_score)
                
                if early_stopping.counter == 0:
                    torch.save(model.state_dict(), checkpoint_dir / f'best_{dataset}.pt')
                    logger.info("New best model checkpoint saved.")
                elif early_stopping.early_stop:
                    logger.info("Early stopping criterion met.")
                    break
            else:
                writer.add_scalars(f'Loss/{phase_name}', {'Train': train_m['loss_total']}, global_epoch_counter)

            scheduler.step()

        best_path = checkpoint_dir / f'best_{dataset}.pt'
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device))

    torch.save(model.state_dict(), checkpoint_dir / "final_hybrid_model.pt")
    logger.info("Training completed successfully.")
    writer.close()

if __name__ == '__main__':
    main()

# train.py
import torch
import torch.optim as optim
import json
import logging
from pathlib import Path
from datetime import datetime
import argparse
import numpy as np
from collections import defaultdict
from torchvision.ops import box_iou
from torch.utils.tensorboard import SummaryWriter  # Aggiunto TensorBoard

from architecture.model import HybridVesselModel
from data import create_dataloaders

ACCUMULATION_STEPS = 32
GRAD_CLIP = 1.0

def setup_logging(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Rimuove gli handler precedenti se esistono (evita la doppia stampa)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Handler per il file
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Handler per il terminale
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # Impedisce la propagazione al root logger (altra causa di doppia stampa)
    logger.propagate = False

    return logger

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.mode = mode
        # Se cerchiamo il massimo (es. F1), partiamo da -infinito
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
    def __init__(self, phase_name='det_reg'):
        self.phase_name = phase_name
        self.reset()

    def reset(self):
        self.losses = defaultdict(list)
        self.accuracies, self.mape_widths, self.mape_lengths, self.ious = [], [], [], []
        self.num_samples = 0

        # Metriche per Detection F1 e Precision
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, losses_dict, out_dict, targets, task='det_reg'):
        for k, v in losses_dict.items():
            if isinstance(v, torch.Tensor): self.losses[k].append(v.item())

        self.num_samples += len(targets)
        if not out_dict: return

        # === 1. Detection (IoU, Precision, Recall, F1) ===
        if task in ['det_reg', 'multi'] and 'detections' in out_dict:
            for p, t in zip(out_dict['detections'], targets):
                if 'boxes' in t:
                    gt_boxes = t['boxes']
                    keep_idx = p['scores'] > 0.5
                    pred_boxes = p['boxes'][keep_idx]

                    if len(gt_boxes) == 0:
                        self.fp += len(pred_boxes)
                        continue
                    if len(pred_boxes) == 0:
                        self.fn += len(gt_boxes)
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
                        else:
                            self.fp += 1

                    self.fn += len(gt_boxes) - len(matched_gt)

        # === 2. Classificazione (FUSAR) e Regressione (HRSID) ===
        if task in ['cls', 'det_reg', 'multi']:
            valid_cls_preds, valid_cls_targs = [], []
            valid_reg_preds, valid_reg_targs = [], []
            idx = 0

            for i, t in enumerate(targets):
                # FIX CRITICO: Contiamo quante BBox sono state *effettivamente* passate al ROI Align
                b = out_dict['boxes'][i]
                num_preds = b.shape[0] if b.numel() > 0 else 0

                if num_preds > 0:
                    if 'category_id' in t: # FUSAR
                        valid_cls_preds.append(out_dict['logits'][idx:idx+num_preds])
                        valid_cls_targs.append(t['category_id'].repeat(num_preds))
                    if 'dimensions' in t: # HRSID
                        valid_reg_preds.append(out_dict['dimensions'][idx:idx+num_preds])
                        valid_reg_targs.append(t['dimensions'])

                idx += num_preds

            if valid_cls_preds:
                preds_cls = torch.argmax(torch.cat(valid_cls_preds), dim=1).cpu().numpy()
                true_cls = torch.cat(valid_cls_targs).cpu().numpy()
                self.accuracies.append((preds_cls == true_cls).mean())

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

        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1

        return metrics

def train_epoch(model, dataloader, optimizer, scaler, device, log_interval, epoch, logger, task):
    model.train()
    metrics = MetricsTracker(phase_name=task)
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
            logger.info(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] - Loss: {m['loss_total']:.4f}")

    return metrics.get_avg_metrics()

@torch.no_grad()
def val_epoch(model, dataloader, device, epoch, logger, task):
    model.train()
    loss_tracker = MetricsTracker(phase_name=task)
    for images, targets in dataloader:
        images = [im.to(device) for im in images]
        targets_gpu = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.autocast(device_type='cuda'):
            _, losses = model(images, targets_gpu, phase=task)
            if losses: loss_tracker.update(losses, {}, targets_gpu, task)

    model.eval()
    metric_tracker = MetricsTracker(phase_name=task)
    for images, targets in dataloader:
        images = [im.to(device) for im in images]
        targets_gpu = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.autocast(device_type='cuda'):
            preds, _ = model(images, targets_gpu, phase=task)
            metric_tracker.update({}, preds, targets_gpu, task)

    res = loss_tracker.get_avg_metrics()
    res.update({k: v for k, v in metric_tracker.get_avg_metrics().items() if k in ['iou', 'accuracy', 'mape_length', 'mape_width', 'precision', 'f1']})
    return res

def setup_phase(model, phase_config, logger, dataset):
    logger.info(f"\n{'='*80}\nPHASE SETUP: {phase_config['name']} ({dataset.upper()})\n{'='*80}")

    # 1. Gestione Backbone
    if dataset == 'fusar':
        # FASE 2: Sblocca SOLO Layer4 e FPN per non distruggere le feature di base
        model.unfreeze_backbone_last_layers()
        logger.info("Backbone: Partially Unfrozen (Only Layer4 & FPN)")
    elif phase_config.get('freeze_backbone', False):
        # Congela tutto (es. se vuoi fare test o fine-tuning leggero)
        for p in model.backbone.parameters(): p.requires_grad = False
        logger.info("Backbone: Fully Frozen")
    else:
        # Fase 1 o 3: Sblocca tutto
        for p in model.backbone.parameters(): p.requires_grad = True
        logger.info("Backbone: Fully Unfrozen")

    # 2. Gestione Detector (RPN)
    if phase_config.get('freeze_detection', False) or dataset == 'fusar':
        # IMPORTANTE: In Fase 2 blocchiamo la detection. FUSAR non ha bounding box 
        # e rovinerebbe la capacità della rete di trovare le navi.
        model.freeze_detection()
        logger.info("Detector: Frozen")
    else:
        model.unfreeze_detection()
        logger.info("Detector: Unfrozen")

    # 3. Gestione Head Classificazione/Regressione (Decoupled Heads)
    if dataset == 'hrsid':
        # FASE 1: Addestriamo solo Box e Dimensioni. Blocchiamo il ramo Classificazione.
        model.unfreeze_reg_head()
        model.freeze_cls_head()
        logger.info("Attribute Heads: Regression UNFROZEN, Classification FROZEN")
        
    elif dataset == 'fusar':
        # FASE 2: Addestriamo solo Classificazione. Proteggiamo la precisione spaziale di HRSID.
        model.freeze_reg_head()
        model.unfreeze_cls_head()
        logger.info("Attribute Heads: Regression FROZEN, Classification UNFROZEN")
        
    elif dataset == 'combined':
        # FASE 3: Multi-task finale. Controlliamo cosa dice il config.json
        if phase_config.get('freeze_classification', False):
            model.freeze_cls_head()
            logger.info("Attribute Heads: Classification FROZEN")
        else:
            model.unfreeze_cls_head()
            logger.info("Attribute Heads: Classification UNFROZEN")
            
        if phase_config.get('freeze_regression', False):
            model.freeze_reg_head()
            logger.info("Attribute Heads: Regression FROZEN")
        else:
            model.unfreeze_reg_head()
            logger.info("Attribute Heads: Regression UNFROZEN")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    with open(args.config, 'r') as f: config = json.load(f)

    logger = setup_logging(config['paths']['logs_dir'])
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Inizializzazione TensorBoard
    tb_dir = Path(config['paths']['logs_dir']) / f"tensorboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    logger.info(f"TensorBoard logs will be saved to: {tb_dir}")

    model = HybridVesselModel(num_classes=config['model']['num_classes'], pretrained_backbone=config['model']['pretrained_backbone']).to(device)
    dataloaders = create_dataloaders(config)
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler()
    patience = config.get('common_training', {}).get('early_stopping_patience', 10)

    global_epoch_counter = 0 # Contatore per un grafico TB continuo

    for phase_name, phase_config in config['training_phases'].items():
        dataset = phase_config['dataset']
        task = {'hrsid': 'det_reg', 'fusar': 'cls', 'combined': 'multi'}[dataset]
        setup_phase(model, phase_config, logger, dataset)

        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=phase_config['learning_rate'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase_config['num_epochs'], eta_min=1e-6)
        early_stopping = EarlyStopping(patience=patience, min_delta=0.001)

        for epoch in range(1, phase_config['num_epochs'] + 1):
            global_epoch_counter += 1

            train_m = train_epoch(model, dataloaders[f'train_{dataset}'], optimizer, scaler, device, config['common_training']['log_interval'], epoch, logger, task)

            if epoch % config['common_training']['val_interval'] == 0:
                val_m = val_epoch(model, dataloaders[f'val_{dataset}'], device, epoch, logger, task)

                # --- LOG TENSORBOARD ---
                # Loss comparata (Train vs Val)
                writer.add_scalars(f'Loss/{phase_name}', {
                    'Train': train_m['loss_total'],
                    'Val': val_m['loss_total']
                }, global_epoch_counter)

                # Altre metriche in base al task
                if task in ['det_reg', 'multi']:
                    writer.add_scalar(f'Metrics/{phase_name}/IoU', val_m['iou'], global_epoch_counter)
                    writer.add_scalar(f'Metrics/{phase_name}/F1', val_m['f1'], global_epoch_counter)
                    writer.add_scalar(f'Metrics/{phase_name}/Precision', val_m['precision'], global_epoch_counter)
                    writer.add_scalar(f'Metrics/{phase_name}/MAPE_L', val_m['mape_length'], global_epoch_counter)
                    writer.add_scalar(f'Metrics/{phase_name}/MAPE_W', val_m['mape_width'], global_epoch_counter)

                if task in ['cls', 'multi']:
                    writer.add_scalar(f'Metrics/{phase_name}/Accuracy', val_m['accuracy'], global_epoch_counter)
                    
                
                if task == 'det_reg':
                    logger.info(f"VAL Epoch {epoch} | Loss: {val_m['loss_total']:.4f} | IoU: {val_m['iou']:.4f} | Prec: {val_m['precision']:.4f} | F1: {val_m['f1']:.4f} | MAPE_L: {val_m['mape_length']:.2f}% | MAPE_W: {val_m['mape_width']:.2f}%")
                    # Score = F1 + IoU (Garantisce che troviamo la nave E che la misuriamo bene)
                    target_score = val_m['f1'] + val_m['iou']
                
                elif task == 'cls':
                    logger.info(f"VAL Epoch {epoch} | Loss: {val_m['loss_total']:.4f} | Acc: {val_m['accuracy']:.4f}")
                    # Score = Accuracy pura
                    target_score = val_m['accuracy']
                
                else: # 'multi'
                    logger.info(f"VAL Epoch {epoch} | Loss: {val_m['loss_total']:.4f} | IoU: {val_m['iou']:.4f} | Prec: {val_m['precision']:.4f} | F1: {val_m['f1']:.4f} | Acc: {val_m['accuracy']:.4f} | MAPE_L: {val_m['mape_length']:.2f}%")
                    # Score bilanciato tra Detection e Classificazione
                    target_score = (val_m['f1'] + val_m['accuracy']) / 2.0

                # Salvataggio basato sulle METRICHE REALI, non sulla loss!
                early_stopping(target_score)
                
                if early_stopping.counter == 0:
                    torch.save(model.state_dict(), checkpoint_dir / f'best_{dataset}.pt')
                    logger.info("    [!] New Best Model Saved!")
                elif early_stopping.early_stop:
                    logger.info("    [!] Early stopping triggered.")
                    break
            else:
                # Logga almeno la train loss anche se non si fa validazione a questo step
                writer.add_scalars(f'Loss/{phase_name}', {'Train': train_m['loss_total']}, global_epoch_counter)

            scheduler.step()

        best_path = checkpoint_dir / f'best_{dataset}.pt'
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device))

    torch.save(model.state_dict(), checkpoint_dir / "final_hybrid_model.pt")
    logger.info("TRAINING COMPLETE!")
    writer.close() # Chiude TensorBoard

if __name__ == '__main__':
    main()
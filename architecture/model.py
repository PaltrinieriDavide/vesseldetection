import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from typing import Dict, List, Optional
import math

class HybridVesselModel(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained_backbone: bool = True):
        super().__init__()
        
        # FIX WARNING: passato backbone_name come parametro nominale
        self.backbone = resnet_fpn_backbone(backbone_name='resnet50', weights='DEFAULT' if pretrained_backbone else None, trainable_layers=3)
        conv1 = self.backbone.body.conv1
        self.backbone.body.conv1 = nn.Conv2d(1, conv1.out_channels, kernel_size=conv1.kernel_size, stride=conv1.stride, padding=conv1.padding, bias=False)
        self.backbone.body.conv1.weight.data = conv1.weight.data.mean(dim=1, keepdim=True)

        anchor_sizes = ((8, 16, 32, 64, 128, 256),) * 5
        aspect_ratios = ((0.5, 1.0, 2.0),) * 5
        self.detector = FasterRCNN(self.backbone, num_classes=2, rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios), box_regression_loss_type="giou", image_mean=[0.0], image_std=[1.0])

        self.roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)
        self.attr_features = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024), 
            nn.LayerNorm(1024),  # <--- Stabilizza i picchi di rumore
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.Linear(1024, 512), 
            nn.LayerNorm(512),   # <--- Stabilizza i picchi di rumore
            nn.ReLU(), 
            nn.Dropout(0.3)
        )        
        self.cls_head = nn.Linear(512, num_classes)
        self.reg_head = nn.Sequential(nn.Linear(512 + 2, 256), nn.ReLU(), nn.Linear(256, 2), nn.Sigmoid())

        # Pesi delle Loss per bilanciamento
        self.loss_weights = {
            'loss_cls': 1.0,
            'loss_reg': 10.0 # Abbassato leggermente da 20 a 10 per bilanciare con la SmoothL1
        }

    def _pad_box(self, box, img_h, img_w, padding_factor=0.10):
        """Aggiunge un padding percentuale alla bbox con controlli di sicurezza"""
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        
        w = max(x2 - x1, 1.0) # FIX: evita w o h negativi/nulli
        h = max(y2 - y1, 1.0)
        
        px = w * padding_factor
        py = h * padding_factor
        
        x1 = torch.clamp(x1 - px, min=0)
        y1 = torch.clamp(y1 - py, min=0)
        x2 = torch.clamp(x2 + px, max=img_w)
        y2 = torch.clamp(y2 + py, max=img_h)
        
        return torch.tensor([x1, y1, x2, y2], device=box.device).unsqueeze(0)

    def _get_best_centered_box(self, det_out_dict, img_h, img_w, device, score_thresh=0.1):
        """Trova la box centrale o usa una fallback sicura per la classificazione"""
        boxes = det_out_dict['boxes']
        scores = det_out_dict['scores']
        
        # Funzione interna di Fallback (60% centrale dell'immagine)
        def get_fallback_box():
            cx, cy = img_w / 2.0, img_h / 2.0
            w, h = img_w * 0.6, img_h * 0.6
            x1 = max(cx - w / 2.0, 0.0)
            y1 = max(cy - h / 2.0, 0.0)
            x2 = min(cx + w / 2.0, float(img_w))
            y2 = min(cy + h / 2.0, float(img_h))
            return torch.tensor([[x1, y1, x2, y2]], device=device)

        # Se non ci sono detection dalla RPN
        if len(boxes) == 0:
            return get_fallback_box()
            
        center_x, center_y = img_w / 2.0, img_h / 2.0
        max_dist = math.sqrt(center_x**2 + center_y**2)
        
        best_box = None
        best_metric = -float('inf')
        
        for i in range(len(boxes)):
            if scores[i] < score_thresh: continue
                
            b = boxes[i]
            bx, by = (b[0]+b[2])/2.0, (b[1]+b[3])/2.0
            dist = math.sqrt((bx - center_x)**2 + (by - center_y)**2)
            
            if dist > max_dist * 0.35: continue
            
            norm_dist = dist / (max_dist + 1e-6)
            metric = scores[i].item() + (1.0 - norm_dist) * 0.8 
            
            if metric > best_metric:
                best_metric = metric
                best_box = b
                
        # FALLBACK: se nessuna box ha superato i filtri
        if best_box is None: 
            return get_fallback_box()
            
        return self._pad_box(best_box, img_h, img_w)

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict]] = None, phase: str = 'det_reg'):
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)
        padded_images =[F.pad(img, (0, max_w - img.shape[2], 0, max_h - img.shape[1])) for img in images]
        images_stack = torch.stack(padded_images)
        
        features = self.backbone(images_stack)
        losses = {}

        if phase in ['det_reg', 'multi']:
            if self.training and targets is not None:
                # FIX KeyError: Filtra le immagini che hanno effettivamente delle bounding box (HRSID)
                # e ignora quelle di classificazione (FUSAR) per il calcolo della loss di detection
                valid_det_imgs = []
                valid_det_targs =[]
                for img, t in zip(images, targets):
                    if 'boxes' in t:
                        valid_det_imgs.append(img)
                        valid_det_targs.append(t)
                
                # Calcola la loss solo se c'è almeno un'immagine HRSID nel batch
                if len(valid_det_imgs) > 0:
                    det_losses = self.detector(valid_det_imgs, valid_det_targs)
                    losses.update(det_losses)
            else:
                detections = self.detector(images)

        # Bootstrapping dei BBox
        boxes =[]
        if phase == 'cls' or (not self.training and phase == 'cls'):
            was_training = self.detector.training
            self.detector.eval()
            with torch.no_grad(): det_out = self.detector(images)
            if was_training: self.detector.train()
                
            for out, img in zip(det_out, images):
                _, h, w = img.shape
                best_box = self._get_best_centered_box(out, h, w, images_stack.device)
                boxes.append(best_box)
                
        elif phase in['det_reg', 'multi']:
            for i, t in enumerate(targets if targets else images):
                if targets and 'dimensions' in t: # HRSID
                    boxes.append(t['boxes'])
                elif targets and 'category_id' in t: # FUSAR
                    # Se stiamo validando/testando, usiamo le detections di batch già pre-calcolate!
                    if not self.training and phase in ['det_reg', 'multi']:
                        out = detections[i]
                    else:
                        was_training = self.detector.training
                        self.detector.eval()
                        with torch.no_grad(): out = self.detector([images[i]])[0]
                        if was_training: self.detector.train()
                        
                    _, h, w = images[i].shape
                    boxes.append(self._get_best_centered_box(out, h, w, images_stack.device))
                else: # Inferenza pura (Validation/Test senza target specifici)
                    # MODIFICA 2: Aggiunto 'multi' anche qui per usare le predizioni della RPN
                    if phase in ['det_reg', 'multi']: 
                        boxes.append(detections[i]['boxes'])
                    else: 
                        boxes.append(torch.empty((0, 4), device=images_stack.device))

        # Estrattore ROI: Passiamo TUTTE le box (anche vuote) per mantenere l'allineamento batch
        padded_sizes =[img.shape[-2:] for img in padded_images]
        total_boxes = sum(b.shape[0] for b in boxes)
        
        if total_boxes > 0:
            roi_feats = self.roi_pool(features, boxes, padded_sizes).flatten(start_dim=1)
            attr_feats = self.attr_features(roi_feats)
            logits = self.cls_head(attr_feats)
            
            box_dims =[]
            for i, b in enumerate(boxes):
                if b.shape[0] > 0:
                    _, h, w = images[i].shape
                    bw = (b[:, 2] - b[:, 0]) / float(w)
                    bh = (b[:, 3] - b[:, 1]) / float(h)
                    box_dims.append(torch.stack([bw, bh], dim=1))
            
            box_dims = torch.cat(box_dims, dim=0)
            lw_pred = self.reg_head(torch.cat([attr_feats, box_dims], dim=1))
        else:
            # Fallback se tutto il batch è vuoto
            logits = torch.empty((0, self.cls_head.out_features), device=images_stack.device)
            lw_pred = torch.empty((0, 2), device=images_stack.device)

        out = {"logits": logits, "dimensions": lw_pred, "boxes": boxes}
        
        # MODIFICA 3: Passiamo i box predetti al tracker delle metriche anche in 'multi'
        if phase in ['det_reg', 'multi'] and not self.training:
            out["detections"] = detections
            
        if self.training and targets is not None:
            cls_targets, reg_targets = [],[]
            valid_cls_idx, valid_reg_idx = [],[]
            current_box_idx = 0
            
            for t, b in zip(targets, boxes):
                num_boxes = b.shape[0] if b.numel() > 0 else 0
                if num_boxes > 0:
                    if 'category_id' in t:
                        cls_targets.append(t['category_id'])
                        valid_cls_idx.extend(range(current_box_idx, current_box_idx + num_boxes))
                    if 'dimensions' in t:
                        reg_targets.append(t['dimensions'])
                        valid_reg_idx.extend(range(current_box_idx, current_box_idx + num_boxes))
                current_box_idx += num_boxes
                
            if len(valid_cls_idx) > 0:
                losses['loss_cls'] = F.cross_entropy(logits[valid_cls_idx], torch.cat(cls_targets)) * self.loss_weights['loss_cls']
            if len(valid_reg_idx) > 0:
                losses['loss_reg'] = F.smooth_l1_loss(lw_pred[valid_reg_idx], torch.cat(reg_targets)) * self.loss_weights['loss_reg']

        return out, losses
    
    # === METODI MANCANTI REINSERITI ===
    def freeze_detection(self):
        for p in self.backbone.parameters(): p.requires_grad = False
        for p in self.detector.parameters(): p.requires_grad = False
        
    def unfreeze_detection(self):
        for p in self.backbone.parameters(): p.requires_grad = True
        for p in self.detector.parameters(): p.requires_grad = True
        
    def freeze_attr_heads(self):
        for p in self.attr_features.parameters(): p.requires_grad = False
        for p in self.cls_head.parameters(): p.requires_grad = False
        for p in self.reg_head.parameters(): p.requires_grad = False
        
    def unfreeze_attr_heads(self):
        for p in self.attr_features.parameters(): p.requires_grad = True
        for p in self.cls_head.parameters(): p.requires_grad = True
        for p in self.reg_head.parameters(): p.requires_grad = True
        
    def unfreeze_backbone_last_layers(self):
        """Sblocca solo l'ultimo blocco di ResNet (layer4) e FPN per adattarsi senza distruggere i pesi"""
        for p in self.backbone.parameters(): 
            p.requires_grad = False
            
        # Sblocca l'ultimo layer del ResNet
        if hasattr(self.backbone, 'body') and hasattr(self.backbone.body, 'layer4'):
            for p in self.backbone.body.layer4.parameters(): 
                p.requires_grad = True
                
        # Sblocca il Feature Pyramid Network
        if hasattr(self.backbone, 'fpn'):
            for p in self.backbone.fpn.parameters(): 
                p.requires_grad = True
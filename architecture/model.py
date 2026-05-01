import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from typing import Dict, List, Optional
import math

# --- 1. MODULO MORFOLOGICO PYTORCH ---
class MorphologicalPreprocessing(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.pad = kernel_size // 2
        self.kernel_size = kernel_size

    def erosion(self, x):
        return -F.max_pool2d(-x, self.kernel_size, stride=1, padding=self.pad)

    def dilation(self, x):
        return F.max_pool2d(x, self.kernel_size, stride=1, padding=self.pad)

    def forward(self, x):
        ero = self.erosion(x)
        dil = self.dilation(x)
        opening = self.dilation(ero)
        closing = self.erosion(dil)
        top_hat = x - opening
        black_hat = closing - x
        morph_grad = dil - ero
        
        # Ritorna ESATTAMENTE 6 canali (Originale + 5 operazioni)
        return torch.cat([x, ero, opening, top_hat, black_hat, morph_grad], dim=1)

# --- 2. COORDINATE CHANNEL ATTENTION (CCA) CON RESIDUAL ---
class CCA(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        
        torch.nn.init.constant_(self.conv_h.bias, 2.0)
        torch.nn.init.constant_(self.conv_w.bias, 2.0)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        
        return x + (x * a_h * a_w)

# --- 3. BOTTOM-UP PATH (Bidirezionale) ---
class BottomUpPath(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.down_convs = nn.ModuleDict({
            '1': nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            '2': nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            '3': nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            'pool': nn.Conv2d(channels, channels, 3, stride=2, padding=1),
        })

    def forward(self, features):
        out = OrderedDict()
        keys = list(features.keys())
        out[keys[0]] = features[keys[0]]
        
        for i in range(1, len(keys)):
            prev_k, curr_k = keys[i-1], keys[i]
            downsampled = self.down_convs[curr_k](out[prev_k])
            if downsampled.shape != features[curr_k].shape:
                downsampled = F.interpolate(downsampled, size=features[curr_k].shape[2:])
            out[curr_k] = features[curr_k] + downsampled
        return out

# --- 4. BACKBONE POTENZIATO TOTALE ---
class EnhancedBackbone(nn.Module):
    def __init__(self, base_backbone):
        super().__init__()
        self.morph = MorphologicalPreprocessing()
        self.base_backbone = base_backbone
        self.out_channels = base_backbone.out_channels
        
        self.cca_modules = nn.ModuleDict({
            k: CCA(self.out_channels) for k in ['0', '1', '2', '3', 'pool']
        })
        self.bottom_up = BottomUpPath(self.out_channels)

    def forward(self, x):
        # 1 Canale -> 6 Canali
        x = self.morph(x)
        # Il base_backbone ora si aspetta 6 canali
        features = self.base_backbone(x)
        
        cca_features = OrderedDict()
        for k, v in features.items():
            cca_features[k] = self.cca_modules[k](v) if k in self.cca_modules else v
            
        return self.bottom_up(cca_features)


class HybridVesselModel(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained_backbone: bool = True):
        super().__init__()
        
        # 1. Utilizziamo ResNeXt50
        base_backbone = resnet_fpn_backbone(
            backbone_name='resnext50_32x4d', 
            weights='DEFAULT' if pretrained_backbone else None, 
            trainable_layers=3
        )

        # 2. ADATTIAMO LOGICAMENTE IL CONV1 A 6 CANALI (La tua intuizione originale!)
        conv1 = base_backbone.body.conv1
        new_conv1 = nn.Conv2d(6, conv1.out_channels, kernel_size=conv1.kernel_size, 
                              stride=conv1.stride, padding=conv1.padding, bias=False)
        if pretrained_backbone:
            # Spalmiamo i pesi sui 6 canali in modo equilibrato
            new_conv1.weight.data = conv1.weight.data.mean(dim=1, keepdim=True).repeat(1, 6, 1, 1) / 6.0
        base_backbone.body.conv1 = new_conv1

        # 3. Assembliamo il backbone
        self.backbone = EnhancedBackbone(base_backbone)

        # 4. Ancore SAR Estreme
        anchor_sizes = ((8, 16, 32, 64, 128, 256),) * 5
        aspect_ratios = ((0.2, 0.5, 1.0, 2.0, 5.0),) * 5
        
        self.detector = FasterRCNN(
            self.backbone, 
            num_classes=2, 
            rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios), 
            box_regression_loss_type="ciou", 
            
            # IMPEDIAMO IL BROADCASTING DI PYTORCH (1 singolo valore = 1 singolo canale)
            image_mean=[0.0], 
            image_std=[1.0],
            
            box_score_thresh=0.15,
            box_nms_thresh=0.4          
        )
        
        self.roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)
        
        self.attr_features = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024), nn.LayerNorm(1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3)
        )        
        self.cls_head = nn.Linear(512, num_classes)
        self.reg_head = nn.Sequential(nn.Linear(512 + 2, 256), nn.ReLU(), nn.Linear(256, 2), nn.Sigmoid())

        self.loss_weights = {
            'loss_cls': 1.0,
            'loss_reg': 10.0
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
        for p in self.backbone.parameters(): 
            p.requires_grad = False
            
        if hasattr(self.backbone.base_backbone, 'body') and hasattr(self.backbone.base_backbone.body, 'layer4'):
            for p in self.backbone.base_backbone.body.layer4.parameters(): p.requires_grad = True
                
        if hasattr(self.backbone.base_backbone, 'fpn'):
            for p in self.backbone.base_backbone.fpn.parameters(): p.requires_grad = True
                
        if hasattr(self.backbone, 'cca_modules'):
            for p in self.backbone.cca_modules.parameters(): p.requires_grad = True
            
        # NUOVO: Sblocchiamo anche la FPN Bidirezionale!
        if hasattr(self.backbone, 'bottom_up'):
            for p in self.backbone.bottom_up.parameters(): p.requires_grad = True

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET
from architecture.model import HybridVesselModel  # <-- importa la classe corretta dal tuo repo

# Impostazioni
MODEL_PATH = '/root/dark-vessel-paltrinieri/HRSID_pipeline/final_hybrid_model.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

annotations = [
    r"/data_ssd/datasets/vessel_detection/dark_vessel/dark_vessel/dark_vessel/patches/S1A_IW_GRDH_1SDV_20240208T181019_20240208T181044_052471_06588D_F7CE.SAFE/fusion/fusion.xml",
    r"/data_ssd/datasets/vessel_detection/dark_vessel/dark_vessel/dark_vessel/patches/S1A_IW_GRDH_1SDV_20240210T053018_20240210T053047_052492_065941_1B03.SAFE/fusion/fusion.xml",
    r"/data_ssd/datasets/vessel_detection/dark_vessel/dark_vessel/dark_vessel/patches/S1A_IW_GRDH_1SDV_20240229T174453_20240229T174518_052777_066300_5FBF.SAFE/fusion/fusion.xml",
    r"/data_ssd/datasets/vessel_detection/dark_vessel/dark_vessel/dark_vessel/patches/S1A_IW_GRDH_1SDV_20240715T053017_20240715T053046_054767_06AB2E_4470.SAFE/fusion/fusion.xml",
    r"/data_ssd/datasets/vessel_detection/dark_vessel/dark_vessel/dark_vessel/patches/S1A_IW_GRDH_1SDV_20240715T175332_20240715T175357_054775_06AB70_ABDE_COG.SAFE/fusion/fusion.xml",
    r"/data_ssd/datasets/vessel_detection/dark_vessel/dark_vessel/dark_vessel/patches/S1A_IW_GRDH_1SDV_20240717T051432_20240717T051457_054796_06AC2F_AA3C.SAFE/fusion/fusion.xml"
]

images = [
    r"/data_ssd/datasets/vessel_detection/dark_vessel/dark_vessel/dark_vessel/patches/S1A_IW_GRDH_1SDV_20240208T181019_20240208T181044_052471_06588D_F7CE.SAFE/point",
    r"/data_ssd/datasets/vessel_detection/dark_vessel/dark_vessel/dark_vessel/patches/S1A_IW_GRDH_1SDV_20240210T053018_20240210T053047_052492_065941_1B03.SAFE/point",
    r"/data_ssd/datasets/vessel_detection/dark_vessel/dark_vessel/dark_vessel/patches/S1A_IW_GRDH_1SDV_20240229T174453_20240229T174518_052777_066300_5FBF.SAFE/point",
    r"/data_ssd/datasets/vessel_detection/dark_vessel/dark_vessel/dark_vessel/patches/S1A_IW_GRDH_1SDV_20240715T053017_20240715T053046_054767_06AB2E_4470.SAFE/point",
    r"/data_ssd/datasets/vessel_detection/dark_vessel/dark_vessel/dark_vessel/patches/S1A_IW_GRDH_1SDV_20240715T175332_20240715T175357_054775_06AB70_ABDE_COG.SAFE/point",
    r"/data_ssd/datasets/vessel_detection/dark_vessel/dark_vessel/dark_vessel/patches/S1A_IW_GRDH_1SDV_20240717T051432_20240717T051457_054796_06AC2F_AA3C.SAFE/point"
]

# --- PREPROCESS SAR ---
class SARPreprocess:
    def __init__(self, mean=1.33, std=3.73, clip_lo=-13.8, clip_hi=4.97, use_clahe=True):
        # default values can be changed/refined
        self.mean = float(mean)
        self.std = float(std)
        self.clip_lo = float(clip_lo)
        self.clip_hi = float(clip_hi)
        self.use_clahe = use_clahe

    def __call__(self, img: Image.Image):
        # per compatibilità modello: pil->tensor->preproc
        import torchvision.transforms.functional as TF
        x = TF.to_tensor(img)
        x = x * 255.0 if x.max() <= 1.0 else x  # solo se serve riportare a [0,255]
        x = torch.log(x + 1e-6)
        x = torch.clamp(x, self.clip_lo, self.clip_hi)
        x = (x - self.mean) / (self.std + 1e-6)
        return x

def load_multichannel_image(point_dir: str, img_name: str):
    vv_path = Path(point_dir) / 'vv' / img_name
    vh_path = Path(point_dir) / 'vh' / img_name
    img_vv = Image.open(vv_path).convert("L")
    img_vh = Image.open(vh_path).convert("L")
    return img_vv, img_vh

def parse_xml_ann(xml_path: str):
    # Custom function: estrarre bounding box/point dagli xml
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall('.//object'):
        # es: per ogni oggetto, estrai bbox, label...
        label_elem = obj.find('name')
        bndbox = obj.find('bndbox')
        if bndbox is not None:
            box = [
                float(bndbox.find('xmin').text),
                float(bndbox.find('ymin').text),
                float(bndbox.find('xmax').text),
                float(bndbox.find('ymax').text)
            ]
            label = label_elem.text if label_elem is not None else "vessel"
            objects.append({'bbox': box, 'label': label})
    return objects

def main():
    preprocess = SARPreprocess()  # Modifica parametri se devi simulare train/test

    # --- MODELLO ---
    model = HybridVesselModel(num_classes=4, pretrained_backbone=False)  # impostare num_classes come nel training!
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    model.to(DEVICE)

    all_results = []

    # Ciclo sulle patch/dataset esterno
    for ann_xml_path, point_path in zip(annotations, images):
        # --- Trova tutte le immagini (prende la lista dei nomi)
        img_names = [f.name for f in (Path(point_path) / "vv").iterdir() if f.is_file()]
        for img_name in img_names:
            img_vv, img_vh = load_multichannel_image(point_path, img_name)
            # Creare "immagine 2 canali": shape [2, H, W]
            arr_vv, arr_vh = np.array(img_vv), np.array(img_vh)
            arr = np.stack([arr_vv, arr_vh], axis=0)  # shape [2, H, W]
            # Per adattare il preprocess usato dal modello (richiede batch di 1 canale)
            img_combined = [preprocess(Image.fromarray(arr_vv)), preprocess(Image.fromarray(arr_vh))]
            img_tensor = torch.stack(img_combined, dim=0).sum(0, keepdim=True)  # Combina come preferisci (es: somma, media, solo VV, ...)

            img_tensor = img_tensor.to(DEVICE)  # shape [1, H, W]
            with torch.no_grad():
                # Inference (batch size 1)
                output, _ = model([img_tensor])
            # --- (Facoltativo) carica GT --- 
            objects = parse_xml_ann(ann_xml_path)

            # --- Estrai info da output, confronta se vuoi con GT, salva metriche/output ---
            pred_boxes = output['boxes'][0].cpu().numpy()  # [N, 4]
            pred_logits = output['logits'].cpu().numpy()
            print(f"\nPatch: {img_name}")
            print(f"Pred detections: {pred_boxes}")
            print(f"Pred logits: {pred_logits}")

            all_results.append({
                "image": img_name,
                "predict_boxes": pred_boxes.tolist(),
                "predict_logits": pred_logits.tolist(),
                "gt_objects": objects
            })
    
    # --- salva su file riassuntivo se serve ---
    # with open('test_results.json', 'w') as f:
    #     import json; json.dump(all_results, f)

if __name__ == "__main__":
    main()
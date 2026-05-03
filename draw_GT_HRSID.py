import json
import cv2
import os

def draw_gt_on_image(image_filename, json_path, images_dir, output_path="output.jpg"):
    # 1. Carica le annotazioni JSON
    print(f"Caricamento del file JSON da: {json_path} ...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 2. Trova l'image_id corrispondente al nome del file
    image_id = None
    for img in data['images']:
        if img['file_name'] == image_filename:
            image_id = img['id']
            break
            
    if image_id is None:
        print(f"Errore: L'immagine '{image_filename}' non è stata trovata nel file JSON.")
        return

    print(f"Trovata immagine con ID: {image_id}")

    # 3. Trova tutte le annotazioni (le navi) per quell'image_id
    # Nel formato COCO bbox è [x_min, y_min, width, height]
    bboxes = []
    for ann in data['annotations']:
        if ann['image_id'] == image_id:
            bboxes.append(ann['bbox'])
            
    print(f"Trovate {len(bboxes)} navi (GT) per questa immagine.")

    # 4. Carica l'immagine
    image_path = os.path.join(images_dir, image_filename)
    if not os.path.exists(image_path):
        print(f"Errore: Immagine non trovata nel percorso {image_path}")
        return
        
    img_cv2 = cv2.imread(image_path)

    # 5. Disegna ogni bounding box (verde sottile)
    # Colore in OpenCV è BGR, quindi (0, 255, 0) è verde puro. Thickness = 1 (sottile)
    for bbox in bboxes:
        x, y, w, h = bbox
        # OpenCV richiede le coordinate come interi
        start_point = (int(x), int(y))
        end_point = (int(x + w), int(y + h))
        
        cv2.rectangle(img_cv2, start_point, end_point, (0, 255, 0), 1)

    # 6. Salva il risultato
    cv2.imwrite(output_path, img_cv2)
    print(f"Immagine salvata con successo in: {output_path}")

# ==========================================
# CONFIGURAZIONE (modifica con i tuoi dati)
# ==========================================
if __name__ == "__main__":
    # Inserisci qui il nome dell'immagine che vuoi testare
    NOME_IMMAGINE = "P0005_600_1400_8189_8989.jpg" 
    
    # Percorsi ai dati
    JSON_PATH = "/root/dark-vessel-paltrinieri/HRSID_pipeline/HRSID_JPG/HRSID_JPG/annotations/train_test2017.json"
    IMAGES_DIR = "/root/dark-vessel-paltrinieri/HRSID_pipeline/HRSID_JPG/HRSID_JPG/JPEGImages"
    
    # Esegui la funzione
    draw_gt_on_image(NOME_IMMAGINE, JSON_PATH, IMAGES_DIR, output_path="risultato_gt.jpg")
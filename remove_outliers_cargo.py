import json
import os
import glob

def process_dataset(split_name, json_path, images_dir, files_to_check):
    print(f"\n{'='*50}")
    print(f"Inizio elaborazione dataset: {split_name}")
    print(f"{'='*50}")

    if not os.path.exists(json_path):
        print(f"File JSON non trovato: {json_path}")
        return

    # 1. Carica le annotazioni
    print(f"Caricamento di {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 2. Trova il category_id per la classe cargo
    cargo_category_id = None
    for category in data.get('categories', []):
        if 'cargo' in category['name'].lower():
            cargo_category_id = category['id']
            break
            
    if cargo_category_id is None:
        print("Errore: Categoria 'cargo' o 'general_cargo' non trovata nel JSON.")
        return
        
    print(f"ID categoria Cargo trovato: {cargo_category_id}")

    # 3. Mappa image_id -> file_name e file_name -> image_id
    id_to_filename = {img['id']: img['file_name'] for img in data.get('images', [])}

    # 4. Trova quali di questi file sono effettivamente classificati come cargo
    ids_to_delete = set()
    for ann in data.get('annotations', []):
        img_id = ann.get('image_id')
        cat_id = ann.get('category_id')
        
        if img_id in id_to_filename:
            file_name = id_to_filename[img_id]
            # Se il file è nella nostra lista e la sua categoria è cargo, eliminiamolo
            if file_name in files_to_check and cat_id == cargo_category_id:
                ids_to_delete.add(img_id)

    if not ids_to_delete:
        print("Nessun file della lista corrisponde alla classe 'cargo' per questo split. Nessuna modifica apportata.")
        return

    print(f"Trovate {len(ids_to_delete)} immagini cargo da eliminare in {split_name}. Procedo...")

    # 5. Filtra immagini e annotazioni per rimuovere quelle segnate
    filtered_images = [img for img in data.get('images', []) if img['id'] not in ids_to_delete]
    filtered_annotations = [ann for ann in data.get('annotations', []) if ann.get('image_id') not in ids_to_delete]

    # Aggiorna il dizionario
    data['images'] = filtered_images
    data['annotations'] = filtered_annotations

    # Opzionale: Aggiorna le statistiche se presenti nel dizionario
    if 'statistics' in data:
        data['statistics']['total_images'] = len(filtered_images)
        data['statistics']['total_annotations'] = len(filtered_annotations)

    # 6. Salva il file JSON aggiornato
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"File {json_path} aggiornato con successo.")

    # 7. Elimina i file fisici dalla cartella images
    deleted_files_count = 0
    for img_id in ids_to_delete:
        file_name_base = id_to_filename[img_id]
        
        # Cerca il file fisico (potrebbe avere estensioni diverse es. .tiff, .jpg, .png)
        search_pattern = os.path.join(images_dir, f"{file_name_base}.*")
        matching_files = glob.glob(search_pattern)
        
        for file_path in matching_files:
            try:
                os.remove(file_path)
                print(f"Eliminato fisicamente: {file_path}")
                deleted_files_count += 1
            except Exception as e:
                print(f"Errore durante l'eliminazione di {file_path}: {e}")

    print(f"Operazione su {split_name} completata. Eliminati {deleted_files_count} file immagine fisici.")


def main():
    # ===============================
    # CONFIGURAZIONE PER 'VAL'
    # ===============================
    val_json_path = '/root/dark-vessel-paltrinieri/ICIP_vessel_detection_regression/datasets/FUSAR/val/annotations.json'
    val_images_dir = '/root/dark-vessel-paltrinieri/ICIP_vessel_detection_regression/datasets/FUSAR/val/images'
    
    val_files = [
        "Ship_C01S02N0193", "Ship_C01S07N0035", "Ship_C01S07N0094", "Ship_C01S07N0096",
        "Ship_C01S07N0175", "Ship_C01S07N0176", "Ship_C01S07N0308", "Ship_C01S07N0323",
        "Ship_C01S07N0419", "Ship_C01S07N0462", "Ship_C01S07N0859", "Ship_C01S07N1103",
        "Ship_C01S07N1152", "Ship_C04S02N0088", "Ship_C04S02N0367", "Ship_C04S02N0578",
        "Ship_C12S09N0092", "Ship_C12S09N0125"
    ]

    # ===============================
    # CONFIGURAZIONE PER 'TEST'
    # ===============================
    test_json_path = '/root/dark-vessel-paltrinieri/ICIP_vessel_detection_regression/datasets/FUSAR/test/annotations.json'
    test_images_dir = '/root/dark-vessel-paltrinieri/ICIP_vessel_detection_regression/datasets/FUSAR/test/images'
    
    test_files = [
        "Ship_C01S07N0017", "Ship_C01S07N0040", "Ship_C01S07N0048", "Ship_C01S07N0051", 
        "Ship_C01S07N0183", "Ship_C01S07N0336", "Ship_C01S07N0406", "Ship_C01S07N0408", 
        "Ship_C01S07N1012", "Ship_C01S07N1223", "Ship_C01S07N1496", "Ship_C01S07N1647", 
        "Ship_C01S07N1650", "Ship_C01S07N1689", "Ship_C01S10N0031", "Ship_C04S02N0071", 
        "Ship_C04S02N0099", "Ship_C04S02N0104", "Ship_C04S02N0309", "Ship_C04S02N0525", 
        "Ship_C04S02N0554", "Ship_C04S02N0577", "Ship_C04S02N0582", "Ship_C04S02N0583", 
        "Ship_C04S02N0590", "Ship_C12S02N0001", "Ship_C12S09N0059", "Ship_C12S09N0065", 
        "Ship_C13S01N0001"
    ]

    # Esegui per entrambi i set
    process_dataset("VAL", val_json_path, val_images_dir, val_files)
    process_dataset("TEST", test_json_path, test_images_dir, test_files)
    
    print("\nScript completato per tutti i dataset.")

if __name__ == "__main__":
    main()
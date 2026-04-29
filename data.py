import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from preprocessing import FUSARDataset, HRSIDDataset

def collate_fn(batch):
    images = [item['image'] for item in batch]
    targets = [{k: v for k, v in item.items() if k != 'image'} for item in batch]
    return images, targets

def make_fusar_sampler(fusar_ds):
    labels = [fusar_ds.class_to_idx[ann["category_name"]] for ann in fusar_ds.annotations.values()]
    counts = torch.bincount(torch.tensor(labels))
    weights_per_class = 1.0 / (counts + 1e-6)
    weights = torch.tensor([weights_per_class[l] for l in labels], dtype=torch.float)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

def create_dataloaders(config):
    dataloaders = {}
    default_batch_size = list(config['training_phases'].values())[0]['batch_size']
    num_workers = config['common_training']['num_workers']
    
    for split in ['train', 'val', 'test']:
        is_train = (split == 'train')
        
        fusar_ds = FUSARDataset(
            fusar_json=config['paths'][f'fusar_{split}_json'],
            image_dir=config['paths'][f'fusar_{split}_images'],
            augment=is_train
        )
        
        hrsid_ds = HRSIDDataset(
            hrsid_json=config['paths'][f'hrsid_{split}_json'],
            image_dir=config['paths'][f'hrsid_{split}_images'],
            augment=is_train
        )
        
        combined_ds = torch.utils.data.ConcatDataset([fusar_ds, hrsid_ds])
        fusar_sampler = make_fusar_sampler(fusar_ds) if is_train else None
        
        dataloaders[f'{split}_fusar'] = DataLoader(fusar_ds, batch_size=default_batch_size, shuffle=(fusar_sampler is None and is_train), sampler=fusar_sampler, num_workers=num_workers, collate_fn=collate_fn)
        dataloaders[f'{split}_hrsid'] = DataLoader(hrsid_ds, batch_size=default_batch_size, shuffle=is_train, num_workers=num_workers, collate_fn=collate_fn)
        dataloaders[f'{split}_combined'] = DataLoader(combined_ds, batch_size=default_batch_size, shuffle=is_train, num_workers=num_workers, collate_fn=collate_fn)
    
    return dataloaders
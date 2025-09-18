import torch
import torchvision.transforms as transforms 
from torchvision import datasets
from torch.utils.data import DataLoader

class MoonDataLoader:
    def __init__(self, config):
        self.config = config
        self.transform = self._create_transform()
    
    def _create_transform(self):
        return transforms.Compose([
            transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
            transforms.CenterCrop(self.config.IMAGE_SIZE),

            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15, fill=0),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2, saturation=0.1,
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    

    def get_dataloader(self):
        try:
            dataset = datasets.ImageFolder(
                root=self.config.DATA_PATH, 
                transform=self.transform
            )
            
            print(f"Dataset loaded successfully. Number of images: {len(dataset)}")
            
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=True,
                num_workers=self.config.NUM_WORKERS,
                pin_memory=True, 
                drop_last=True    
            )
            
            return dataloader
                
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
def get_dataloader(config):
    loader = MoonDataLoader(config)
    return loader.get_dataloader()
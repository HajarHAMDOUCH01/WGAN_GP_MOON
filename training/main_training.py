import torch
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('/content/moon-wgan-gp/')

from config import Config
from models import create_models
from data_loader import get_dataloader
from trainer import WGANGPTrainer

def main():
    config = Config()
    print("Loading dataset...")
    dataloader = get_dataloader(config)
    print(f"Loaded {len(dataloader.dataset)} images")

    print("Creating models ...")
    generator, discriminator = create_models(config)

    total_params_g = sum(p.numel() for p in generator.parameters())
    total_params_d = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator parameters: {total_params_g:,}")
    print(f"Discriminator parameters: {total_params_d:,}")

    trainer = WGANGPTrainer(generator, discriminator, dataloader, config)
    trainer.train()

def load_and_generate(checkpoint_path, num_samples=16):
    config = Config()

    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    generator, _ = create_models(config)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    with torch.no_grad():
        noise = torch.randn(num_samples, config.LATENT_DIM, device=config.DEVICE)
        fake_images = generator(noise)

        import torchvision.utils as vutils
        import matplotlib.pyplot as plt 
        import numpy as np

        grid = vutils.make_grid(fake_images, nrow=4, normalize=True)
        plt.figure(figsize=(12,12))
        plt.imshow(np.transpose(grid.cpu(), (1,2,0)))
        plt.title('generated moon images')
        plt.axis('off')
        plt.show()

        for i, img in enumerate(fake_images):
            vutils.save_image(img, f"{config.SAMPLES_SAVE_PATH}/generated_{i:03d}.png", 
                            normalize=True)

if __name__ == "__main__":
    main()

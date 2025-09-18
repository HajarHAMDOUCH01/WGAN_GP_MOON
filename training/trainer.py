import torch
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib as plt
import numpy as np
import os 
from pathlib import Path
from config import Config

class WGANGPTrainer:
    def __init__(self, generator, discriminator, dataloader, config):
        self.generator = generator
        self.discriminator = discriminator
        self.dataloader = dataloader
        self.config = config

        Path(config.MODEL_SAVE_PATH).mkdir(parents=True, exist_ok=True)
        Path(config.SAMPLES_SAVE_PATH).mkdir(parents=True, exist_ok=True)

        self.opt_g = optim.Adam(
            self.generator.parameters(),
            lr=config.LEARNING_RATE_G,
            betas=(config.BETA1, config.BETA2)
        )
        self.opt_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config.LEARNING_RATE_D,
            betas=(config.BETA1, config.BETA2)
        )

        self.scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_g, mode='min', factor=0.5, patience=20, verbose=True
        )
        self.scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_d, mode='min', factor=0.5, patience=20, verbose=True
        )

        self.scaler = torch.cuda.amp.GradScaler('cuda') if config.USE_MIXED_PRECISION else None

        self.fixed_noise = torch.randn(16, config.LATENT_DIM, device=config.DEVICE)

        self.g_losses = []
        self.d_losses = []
        self.wasserstein_distances = []

    def compute_gadient_penality(self, real_samples, fake_samples):
        batch_size = real_samples.size(0)
        epsilon = torch.randn(batch_size, 1,1,1, device=self.config.DEVICE)

        interpolates = epsilon * real_samples + (1-epsilon)*fake_samples
        interpolates.requires_grad_(True)

        d_interpolates = self.discriminator(interpolates)

        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradients_penalty = torch.mean((gradient_norm - 1) ** 2)

        return gradients_penalty
    
    def train_discriminator(self, real_images):
        self.discriminator.train()
        d_loss_total=0

        for _ in range(self.config.N_CRITIC):
            self.opt_d.zero_grad()

            batch_size = real_images.size(0)

            if self.config.USE_MIXED_PRECISION:
                with torch.cuda.amp.autocast(enabled=True, dtype=float, cache_enabled=True):
                    real_scores = self.discriminator(real_images)

                    noise = torch.randn(batch_size, self.config.LATENT_DIM, device=self.config.DEVICE)
                    fake_images = self.generator(noise).detach()
                    fake_scores = self.discriminator(fake_images)

                    gp = self.compute_gadient_penality(real_images, fake_images)

                    d_loss = -torch.mean(real_scores) + torch.mean(fake_scores) + self.config.LAMBDA_GP * gp
                
                self.scaler.scale(d_loss).backward()
                self.scaler.step(self.opt_d)
                self.scaler.update()
            
            else:
                real_scores = self.discriminator(real_images)
                noise = torch.randn(batch_size, self.config.LATENT_DIM, device=self.config.DEVICE)
                fake_images = self.generator(noise).detach()
                fake_scores = self.discriminator(fake_images)

                gp = self.compute_gradient_penalty(real_images, fake_images)

                d_loss = -torch.mean(real_scores) + torch.mean(fake_scores) + self.config.LAMBDA_GP * gp
                
                d_loss.backward()
                self.opt_d.step()
            
            d_loss_total += d_loss.item()
        return d_loss_total / self.config.N_CRITIC, -torch.mean(real_scores) + torch.mean(fake_scores)

    def train_generator(self, batch_size):
        self.generator.train()
        self.opt_g.zero_grad()

        noise = torch.randn(batch_size, self.config.LATENT_DIM, device=self.config.DEVICE)

        if self.config.USE_MIXED_PRECISION:
            with torch.cuda.amp.autocast(enabled=True, cache_enabled=True, dtype=float):
                fake_images = self.generator(noise)
                fake_scores = self.discriminator(fake_images)
                g_loss = -torch.mean(fake_scores)

            self.scaler.scale(g_loss).backward()
            self.scaler.step(self.opt_g)
            self.scaler.update()

        else:
            fake_images = self.generator(noise)
            fake_scores = self.discriminator(fake_images)
            g_loss = -torch.mean(fake_scores)

            g_loss.backward()
            self.opt_g.step()
        return g_loss.item()
    
    def generate_samples(self, epoch):
        self.generator.eval()
        with torch.no_grad():
            fake_images = self.generator(self.fixed_noise)
            vutils.save_image(
                fake_images,
                f"{self.config.SAMPLES_SAVE_PATH}/epoch_{epoch:03d}.png",
                normalize=True,
                nrow=4
            )
            grid = vutils.make_grid(fake_images, nrow=4, normalize=True)
            plt.figure(figsize=(10,10))
            plt.imshow(np.transpose(grid.cpu(), (1,2,0)))
            plt.title(f'Generated Moon Images - Epoch {epoch}')
            plt.axis('off')
            plt.savefig(f"{self.config.SAMPLES_SAVE_PATH}/display_epoch_{epoch:03d}.png", 
                       bbox_inches='tight', dpi=100)
            plt.show()
    
    def save_models(self, epoch):
        """Save model checkpoints"""
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.opt_g.state_dict(),
            'optimizer_d_state_dict': self.opt_d.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
            'config': self.config
        }, f"{self.config.MODEL_SAVE_PATH}/checkpoint_epoch_{epoch:03d}.pth")
    
    def plot_losses(self):
        """Plot training losses"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.g_losses, label='Generator Loss')
        plt.plot(self.d_losses, label='Discriminator Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Losses')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.wasserstein_distances, label='Wasserstein Distance')
        plt.xlabel('Iterations')
        plt.ylabel('Distance')
        plt.legend()
        plt.title('Wasserstein Distance')
        
        plt.tight_layout()
        plt.savefig(f"{self.config.SAMPLES_SAVE_PATH}/training_curves.png")
        plt.show()

    def train(self):
        print(f"Starting training on {self.config.DEVICE}")
        print(f"Number of epochs: {self.config.NUM_EPOCHS}")
        print(f"Batch size: {self.config.BATCH_SIZE}")

        iteration=0

        for epoch in range(self.config.NUM_EPOCHS):
            epoch_d_loss = 0
            epoch_g_loss = 0

            for i, (real_images, _) in enumerate(self.dataloader):
                real_images = real_images.to(self.config.DEVICE)
                d_loss, wasserstein_dist = self.train_discriminator(real_images)
                g_loss = self.train_generator(real_images.size(0))

                self.d_losses.append(d_loss)
                self.g_losses.append(g_loss)
                self.wasserstein_distances.append(wasserstein_dist.item())
                epoch_d_loss += d_loss
                epoch_g_loss += g_loss
                
                if i % self.config.LOG_INTERVAL == 0:
                    print(f"Epoch [{epoch+1}/{self.config.NUM_EPOCHS}] "
                          f"Step [{i+1}/{len(self.dataloader)}] "
                          f"D_loss: {d_loss:.4f} G_loss: {g_loss:.4f} "
                          f"W_dist: {wasserstein_dist:.4f}")
                
                iteration += 1

            avg_d_loss = epoch_d_loss / len(self.dataloader)
            avg_g_loss = epoch_g_loss / len(self.dataloader)
            self.scheduler_d.step(avg_d_loss)
            self.scheduler_g.step(avg_g_loss)

            # Generating samples
            if (epoch + 1) % self.config.SAMPLE_INTERVAL == 0:
                self.generate_samples(epoch + 1)
            
            # Saving the models
            if (epoch + 1) % self.config.SAVE_INTERVAL == 0:
                self.save_models(epoch + 1)

        self.save_models(self.config.NUM_EPOCHS)
        self.plot_losses()
        print("Training completed!")
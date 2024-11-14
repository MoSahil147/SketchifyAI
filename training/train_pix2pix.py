# training/train_pix2pix.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pix2pix_model import Pix2PixGenerator, Pix2PixDiscriminator
import os
import cv2

class SketchDataset(Dataset):
    def __init__(self, sketch_dir, real_dir, transform=None):
        self.sketch_dir = sketch_dir
        self.real_dir = real_dir
        self.transform = transform
        self.sketch_images = os.listdir(sketch_dir)
        
    def __len__(self):
        return len(self.sketch_images)

    def __getitem__(self, idx):
        sketch_path = os.path.join(self.sketch_dir, self.sketch_images[idx])
        real_path = os.path.join(self.real_dir, self.sketch_images[idx])
        
        sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
        real = cv2.imread(real_path, cv2.IMREAD_COLOR)
        
        if self.transform:
            sketch = self.transform(sketch)
            real = self.transform(real)
            
        return sketch, real

def train_pix2pix(generator, discriminator, dataloader, num_epochs=50, lr=0.0002):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator, discriminator = generator.to(device), discriminator.to(device)
    
    gen_opt = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    disc_opt = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion_gan = nn.BCELoss()
    criterion_l1 = nn.L1Loss()
    
    for epoch in range(num_epochs):
        for i, (sketch, real) in enumerate(dataloader):
            sketch, real = sketch.to(device), real.to(device)
            valid = torch.ones((sketch.size(0), 1), device=device)
            fake = torch.zeros((sketch.size(0), 1), device=device)
            
            # Train Generator
            gen_opt.zero_grad()
            generated_images = generator(sketch)
            gan_loss = criterion_gan(discriminator(generated_images), valid)
            l1_loss = criterion_l1(generated_images, real) * 100
            g_loss = gan_loss + l1_loss
            g_loss.backward()
            gen_opt.step()

            # Train Discriminator
            disc_opt.zero_grad()
            real_loss = criterion_gan(discriminator(real), valid)
            fake_loss = criterion_gan(discriminator(generated_images.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            disc_opt.step()

            if i % 10 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

        # Save model checkpoint
        torch.save(generator.state_dict(), "../models/sketch_to_image_model.pth")

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = SketchDataset("datasets/sketch_images", "datasets/realistic_images", transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    generator = Pix2PixGenerator()
    discriminator = Pix2PixDiscriminator()
    
    train_pix2pix(generator, discriminator, dataloader, num_epochs=50, lr=0.0002)
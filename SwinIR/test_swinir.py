import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from swinir import SwinIRBlock

BATCH_SIZE = 16
IMG_SIZE = 32
WINDOW_SIZE = 8      
EMBED_DIM = 64     
NUM_HEADS = 4  
SHIFT_SIZE = WINDOW_SIZE // 2 


print("data loader...")
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

class SwinIRTestWrapper(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, img_size=32):
        super().__init__()
        self.embed = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.swin_block = SwinIRBlock(
            dim=embed_dim,
            input_resolution=(img_size, img_size),
            num_heads=NUM_HEADS,
            window_size=WINDOW_SIZE,
            shift_size=SHIFT_SIZE
        )
        self.unembed = nn.Conv2d(embed_dim, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.embed(x)
        x = x.permute(0, 2, 3, 1) 
        x = x.view(B, H * W, -1)

        x = self.swin_block(x)
        

        x = x.view(B, H, W, -1) 
        x = x.permute(0, 3, 1, 2) 
        x = self.unembed(x)
        
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SwinIRTestWrapper(embed_dim=EMBED_DIM, img_size=IMG_SIZE).to(device)
print(f"Model initialized on {device}.")

images, _ = next(iter(train_loader))
images = images.to(device)

with torch.no_grad():
    output = model(images)

print("\n--- Tensor Shapes ---")
print(f"Input shape:  {images.shape}")
print(f"Output shape: {output.shape}")
assert images.shape == output.shape, "Input and output shapes do not match!"
print("Shapes match as expected.")

def imshow(tensor, title):
    tensor = tensor.cpu().detach().clamp(0, 1)
    grid_img = make_grid(tensor, nrow=4, normalize=False, pad_value=1)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.title(title)
    plt.axis('off')

print("\nGenerating visualization...")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
imshow(images, "Original CIFAR-10 Images")

plt.subplot(1, 2, 2)
imshow(output, "Transformed by SwinIR Block")

plt.suptitle("SwinIR Block Input vs. Output", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
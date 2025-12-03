import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os
import matplotlib.pyplot as plt   # <--- NEW: do rysowania wykresów
import time
from tqdm.auto import tqdm

# Upewniamy się, że katalog na wyniki istnieje
os.makedirs("generated", exist_ok=True)   # <--- NEW

# Hiperparametry – można je modyfikować do eksperymentów
latent_dim = 100       # wymiar wektora losowego (generator input)
image_size = 64        # rozmiar obrazów (64x64)
num_channels = 3       # liczba kanałów obrazów (3 dla RGB)
batch_size = 64        # rozmiar wsadu
learning_rate = 0.0002 # szybkość uczenia (dla generatora i dyskryminatora)
num_epochs = 100        # liczba epok początkowo

# Transformacja danych: przeskalowanie obrazów do [-1, 1] i rozmiaru 64x64
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*num_channels, [0.5]*num_channels)  # normalizacja do [-1,1]
])

# Załaduj dataset (CIFAR10)
dataset = torchvision.datasets.CelebA(root='./data', split='train', download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


print("Urządzenie domyślne:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Definicja generatora (konwertuje wektor losowy (latent_dim) do obrazu 64x64x3)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Używamy ngf (generator feature maps) jako bazowej liczby filtrów
        ngf = 64
        self.main = nn.Sequential(
            # Warstwa 1: start od wektora latent_dim, transformacja do ngf*8 map cech (rozmiar 4x4)
            nn.ConvTranspose2d(latent_dim, ngf*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            # Warstwa 2: ngf*8 -> ngf*4 map cech (rozmiar 8x8)
            nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # Warstwa 3: ngf*4 -> ngf*2 map cech (16x16)
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # Warstwa 4: ngf*2 -> ngf map cech (32x32)
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Warstwa 5 (ostatnia): ngf -> 3 kanały (RGB obraz 64x64)
            nn.ConvTranspose2d(ngf, num_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # funkcja aktywacji tanh, aby wartości były w [-1,1]
        )
    def forward(self, z):
        return self.main(z)

# Definicja dyskryminatora (ocenia, czy obraz jest prawdziwy czy wygenerowany)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        ndf = 64  # discriminator feature maps
        self.main = nn.Sequential(
            # Warstwa 1: 3 kanały -> ndf (32x32)
            nn.Conv2d(num_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Warstwa 2: ndf -> ndf*2 (16x16)
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # Warstwa 3: ndf*2 -> ndf*4 (8x8)
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # Warstwa 4: ndf*4 -> ndf*8 (4x4)
            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # Warstwa 5 (ostatnia): ndf*8 -> 1 (wyjście skalarne)
            nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()  # wynik to prawdopodobieństwo, że obraz jest prawdziwy
        )
    def forward(self, img):
        return self.main(img).view(-1)  # spłaszczamy wyjście do wektora

# Inicjalizacja modeli generatora i dyskryminatora
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator().to(device)
D = Discriminator().to(device)

# Funkcja straty i optymalizatory (Adam z beta1=0.5, zgodnie z zaleceniami DCGAN)
criterion = nn.BCELoss()
optim_G = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optim_D = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# LISTY NA STRATY PO EPOKACH  <--- NEW
loss_D_history = []
loss_G_history = []

total_start = time.time()

# Pętla treningowa
for epoch in range(num_epochs):
    epoch_start = time.time()

    running_loss_D = 0.0   # <--- NEW: sumowanie strat w epoce
    running_loss_G = 0.0
    num_batches = 0

    # Pasek postępu batchy w danej epoce
    for i, (real_images, _) in enumerate(
        tqdm(dataloader, desc=f"Epoka {epoch+1}/{num_epochs}", leave=False, position=1)
    ):
        real_images = real_images.to(device)
        batch_size_cur = real_images.size(0)
        
        # Etykiety dla prawdziwych (1) i wygenerowanych (0) obrazów
        real_labels = torch.ones(batch_size_cur, device=device)
        fake_labels = torch.zeros(batch_size_cur, device=device)
        
        # 1. TRENING DYSKRYMINATORA
        outputs = D(real_images)
        loss_real = criterion(outputs, real_labels)

        z = torch.randn(batch_size_cur, latent_dim, 1, 1, device=device)
        fake_images = G(z)
        outputs_fake = D(fake_images.detach())
        loss_fake = criterion(outputs_fake, fake_labels)

        loss_D = loss_real + loss_fake
        optim_D.zero_grad()
        loss_D.backward()
        optim_D.step()
        
        # 2. TRENING GENERATORA
        outputs_fake = D(fake_images)
        loss_G = criterion(outputs_fake, real_labels)

        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        running_loss_D += loss_D.item()
        running_loss_G += loss_G.item()
        num_batches += 1

    # Średnie straty po epoce
    avg_loss_D = running_loss_D / num_batches
    avg_loss_G = running_loss_G / num_batches
    loss_D_history.append(avg_loss_D)
    loss_G_history.append(avg_loss_G)

    epoch_time = time.time() - epoch_start
    print(f"Epoka {epoch+1}/{num_epochs} zakończona w {epoch_time:.2f} s "
          f"(Strata D: {avg_loss_D:.4f}, Strata G: {avg_loss_G:.4f})")

    # Podgląd co 5 epok
    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            fake = G(torch.randn(64, latent_dim, 1, 1, device=device)).detach().cpu()
        vutils.save_image(fake,
                          f"generated/fake_images_epoch_{epoch+1}.png",
                          normalize=True)
        print(f"Zapisano obrazy: generated/fake_images_epoch_{epoch+1}.png")
# === PO ZAKOŃCZENIU TRENINGU – GENEROWANIE WYKRESÓW STRAT ===  <--- NEW

total_time = time.time() - total_start
print(f"\nCałkowity czas treningu: {total_time/60:.2f} minut")

epochs = range(1, num_epochs + 1)

# Wspólny wykres dla D i G
plt.figure()
plt.plot(epochs, loss_D_history, label="Strata D")
plt.plot(epochs, loss_G_history, label="Strata G")
plt.xlabel("Epoka")
plt.ylabel("Strata")
plt.title("Strata D i G w czasie treningu")
plt.legend()
plt.grid(True)
plt.savefig("generated/loss_D_G.png")
plt.close()

# Osobny wykres dla D
plt.figure()
plt.plot(epochs, loss_D_history, label="Strata D")
plt.xlabel("Epoka")
plt.ylabel("Strata")
plt.title("Strata dyskryminatora (D)")
plt.grid(True)
plt.savefig("generated/loss_D.png")
plt.close()

# Osobny wykres dla G
plt.figure()
plt.plot(epochs, loss_G_history, label="Strata G")
plt.xlabel("Epoka")
plt.ylabel("Strata")
plt.title("Strata generatora (G)")
plt.grid(True)
plt.savefig("generated/loss_G.png")
plt.close()

print("Zapisano wykresy strat w folderze 'generated'.")

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os


print("Wersja Pythona:", __import__("sys").version)
print("Wersja torch:", torch.__version__)
print("CUDA dostępna:", torch.cuda.is_available())
print("Urządzenie domyślne:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))


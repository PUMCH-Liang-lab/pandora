import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import time
from model.convnextv2 import convnextv2_H, convnextv2_B, convnextv2_L, convnextv2_N, convnextv2_T

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

if __name__ == "__main__":
    # get weights from https://huggingface.co/PUMCH-Liang-lab/pandoras
    weight_path = "./weight/Pandora-B.pt"
    model = convnextv2_B(Linear_only=False)
    # Remove the classification head; keep the feature extractor only
    model.convnextv2.head = nn.Sequential()
    model.convnextv2.load_state_dict(
        torch.load(weight_path, map_location="cpu")
    )
    model.eval()

    img = Image.open("./data_sample/sample_patch.png").convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)
    with torch.inference_mode():
        _, features = model(img)
    # features is a list that contains the feature outputs from each layer
    for idx, feature in enumerate(features):
        print(f"{idx}th output feature shape: {feature.shape}")

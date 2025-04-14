from typing import Iterable

import numpy as np
import torch
import torchvision.models as tm  # Ensure this matches your library
from torch import nn
from PIL import Image
from torchvision import transforms

class ENV2Ships(nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = tm.EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()
        self.pretrained = tm.efficientnet_v2_s(weights=None)
        self.pretrained.classifier = nn.Linear(1280, 512, bias=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512, 9)

    def forward(self, x):
        x = self.transforms(x)
        x = self.pretrained(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        return self.fc(x)
    
    def predict(self, image_paths: Iterable[str]) -> np.ndarray:
        """
        Performs classification on arr of image of shape (1, 240, 320, 3)
        
        Parameters
        ----------
        img_arr: Image array
        
        Returns
        -------
        prediction: Int
        """
        transform = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.ToTensor(), #Converts the PIL image to a PyTorch tensor and scales the pixel values to [0, 1]
        ])
        images = [Image.open(path).convert("RGB") for path in image_paths]
        image_tensors = torch.stack([transform(image) for image in images])

        self.eval()
        with torch.no_grad():
            predictions = self(image_tensors).argmax(dim=1).detach().cpu().numpy()
        return predictions

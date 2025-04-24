# # import os
# # from PIL import Image
# # from torch.utils.data import Dataset
# # from torchvision import transforms

# # class CustomImageDataset(Dataset):
# #     def __init__(self, root_dir, transform=None):
# #         self.samples = []
# #         self.transform = transform
# #         self.class_to_idx = {}
# #         self._load_dataset(root_dir)

# #     def _load_dataset(self, root_dir):
# #         # Get list of top-level folders (folder_1, folder_2, etc.)
# #         folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

# #         class_names = set()
# #         for folder in folders:
# #             for class_name in os.listdir(folder):
# #                 class_path = os.path.join(folder, class_name)
# #                 if not os.path.isdir(class_path):
# #                     continue
# #                 class_names.add(class_name)

# #         # Create class name to index mapping
# #         self.class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(class_names))}

# #         # Load image paths and labels
# #         for folder in folders:
# #             for class_name in os.listdir(folder):
# #                 class_path = os.path.join(folder, class_name)
# #                 if not os.path.isdir(class_path):
# #                     continue
# #                 label = self.class_to_idx[class_name]
# #                 for fname in os.listdir(class_path):
# #                     fpath = os.path.join(class_path, fname)
# #                     if os.path.isfile(fpath) and fname.lower().endswith(('.png', '.jpg', '.jpeg')):
# #                         self.samples.append((fpath, label))

# #     def __len__(self):
# #         return len(self.samples)

# #     def __getitem__(self, idx):
# #         image_path, label = self.samples[idx]
# #         image = Image.open(image_path).convert("RGB")
# #         if self.transform:
# #             image = self.transform(image)
# #         return image, label


# # from torchvision import transforms

# # transform = transforms.Compose([
# #     transforms.ToTensor()
# # ])

# # dataset = CustomImageDataset(root_dir="dataset_20200803", transform=transform)

import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset

def create_tensor_dataset(root_dir, image_size=(224, 224)):
    image_paths = []
    labels = []
    class_to_idx = {}

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    # Get top-level folders like folder_1, folder_2, etc.
    top_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    # First pass: collect all class names
    all_classes = set()
    for folder in top_folders:
        for class_name in os.listdir(folder):
            class_path = os.path.join(folder, class_name)
            if os.path.isdir(class_path):
                all_classes.add(class_name)

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(all_classes))}

    # Second pass: collect image paths and labels
    for folder in top_folders:
        for class_name in os.listdir(folder):
            class_path = os.path.join(folder, class_name)
            if not os.path.isdir(class_path):
                continue
            label = class_to_idx[class_name]
            for fname in os.listdir(class_path):
                fpath = os.path.join(class_path, fname)
                if os.path.isfile(fpath) and fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(fpath)
                    labels.append(label)

    # Load images into memory as tensors
    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img_tensor = transform(img)
        images.append(img_tensor)

    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return TensorDataset(images_tensor, labels_tensor)

# dataset = create_tensor_dataset("dataset_20200803")
# print(type(dataset))  # should be <class 'torch.utils.data.dataset.TensorDataset'>

# torch.save(dataset, "dataset_20200803.pt")

import os
os.environ['MODEL_WEIGHTS'] = 'classifier/clf_files/weights.pt'
os.environ["CLASSES_TXT"] = 'classifier/clf_files/classes.txt'
from classifier.model import CLFModel
model = CLFModel()

dataset = torch.load("dataset_20200803.pt", weights_only=False)

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def evaluate_model(model, dataset, batch_size=32):
    # Create DataLoader from TensorDataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Evaluating"):
        # Convert tensor batch to list of numpy arrays
        images_np = [img.permute(1, 2, 0).numpy() for img in images]  # HWC format

        # Predict
        preds = model.predict(images_np)

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    # Compute accuracy
    acc = accuracy_score(all_labels, all_preds)
    return all_preds, all_labels, acc

all_preds, all_labels, acc = evaluate_model(model, dataset)

print("Accuracy:", acc)
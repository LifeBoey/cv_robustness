from cvrob import *
import os
import pandas as pd
from PIL import Image
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np

# class ImageLabelDataset(Dataset):
#     def __init__(self, csv_file, root_dir, transform=None):
#         self.labels_df = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.labels_df)

#     def __getitem__(self, idx):
#         img_file = self.labels_df.iloc[idx]['image']
#         label = int(self.labels_df.iloc[idx]['label'])

#         img_path = os.path.join(self.root_dir, str(img_file))
#         image = Image.open(img_path).convert("RGB")

#         if self.transform:
#             image = self.transform(image)

#         return image, label




# # Define any transformations you want
# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])

# dataset = ImageLabelDataset(csv_file='selected_images_labels_50.csv', root_dir='X_reference_png_50', transform=transform)

# all_images = torch.stack([data[0] for data in dataset])
# all_labels = torch.tensor([data[1] for data in dataset])
# test_dataset = torch.utils.data.TensorDataset(all_images, all_labels)
# test_loader = DataLoader(dataset, batch_size=5, shuffle=False)
test_dataset = torch.load("dataset_ships.pt", weights_only=False)

device = torch.device("cpu")
model = torch.load('efficient_full.pt', weights_only=False)
model.to(device)

list_of_methods = [ label_update_indice_method, 
                    loss_indice_method,
                    aum_indice_method,
                    fine_indice_method,
                    wrong_prediction_indice_method,
                    cleanlab_indice_method,
                    deep_knn_indice_method]

with open('config.yaml', 'r') as file:
    param_dict = yaml.safe_load(file)

param_dict['num_epochs'] = 5
noisy_indices_all, true_noisy_indices, test_loader = label_noise_method(test_dataset, 
                                                                        model, 
                                                                        device,
                                                                        list_of_methods, 
                                                                        param_dict,
                                                                        evaluate=False, 
                                                                        noise_ratio=69, 
                                                                        batch_size=5)

final_noisy_indices = ensemble_method(noisy_indices_all)
print('final noisy indices, ', final_noisy_indices)

print("noisy indices done! let's go to: robustness evaluation")

for lib in ['album', 'nrtk', 'augly', 'ic']:
    augmentation_list, augmentation_str, corrupt_func = get_corruption_helpers('album')
    aug_dict_g = generic_combination_gradients(
        model, 
        test_loader, 
        device, 
        corrupt_func, 
        augmentation_list, 
        augmentation_str, 
        plot_graphs=False
    )
    print(aug_dict_g)
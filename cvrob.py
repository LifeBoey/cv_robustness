#Import libraries and data and model
import torch
import yaml
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
import torch.nn.functional as F
import numpy as np
from aum import AUMCalculator, DatasetWithIndex
import pandas as pd
from FINE_official.dynamic_selection.selection.svd_classifier import get_score, get_singular_vector, get_mean_vector
from tqdm import tqdm
from cleanlab.filter import find_label_issues
from scipy.special import softmax
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import copy
import matplotlib.pyplot as plt
import inspect
from skimage import io
import requests
from PIL import Image
from io import BytesIO

from imagecorruptions import corrupt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from nrtk.impls.perturb_image.generic.cv2.blur import AverageBlurPerturber, GaussianBlurPerturber, MedianBlurPerturber
from nrtk.impls.perturb_image.generic.PIL.enhance import (
    BrightnessPerturber,
    ColorPerturber,
    ContrastPerturber,
    SharpnessPerturber,
)
from nrtk.impls.perturb_image.generic.skimage.random_noise import (
    GaussianNoisePerturber,
    PepperNoisePerturber,
    SaltAndPepperNoisePerturber,
    SaltNoisePerturber,
    SpeckleNoisePerturber,
)
from augly.image import blur, brightness, random_noise, contrast, color_jitter, pixelization, sharpen
from augly.image import aug_np_wrapper

def corrupt_func_album(images_np, severity, param_dict):
    aug_func = param_dict['aug_method']
    param_dict2 = {k:v for k,v in param_dict.items() if k != 'aug_method'}
    aug_params = {k: (v(severity) if callable(v) else v) for k, v in param_dict2.items()}
    transform = A.Compose([aug_func(**aug_params), ToTensorV2()])
    corrupted_imgs = np.array([ transform(image=img)['image'].permute(1, 2, 0).numpy() for img in images_np ])
    return corrupted_imgs

def corrupt_func_augly(images_np, severity, param_dict):
    aug_func = param_dict['aug_method']
    param_dict2 = {k:v for k,v in param_dict.items() if k != 'aug_method'}
    aug_params = {k: (v(severity) if callable(v) else v) for k, v in param_dict2.items()}
    corrupted_imgs = np.array([aug_np_wrapper(img, aug_func, **aug_params) for img in images_np])
    return corrupted_imgs

def corrupt_func_nrtk(images_np, severity, param_dict):
    aug_func = param_dict['aug_method']
    param_dict2 = {k:v for k,v in param_dict.items() if k != 'aug_method'}
    aug_params = {k: (v(severity) if callable(v) else v) for k, v in param_dict2.items()}
    perturber = aug_func(**aug_params)
    corrupted_imgs = np.array([ perturber(img)[0] for img in images_np ])
    return corrupted_imgs

def corrupt_func_imagecorrupt(images_np, severity, param_dict):
    aug_func = param_dict['corrname']
    corrupted_imgs = np.array([ corrupt(img.astype(np.uint8), corruption_name=aug_func, severity=severity) for img in images_np ])
    return corrupted_imgs

def corrupt_and_plot_generic(img, augmentations, aug_names, corrupt_func, filename=None):
    severities = [0, 1, 2]
    fig, axes = plt.subplots(len(augmentations), len(severities), figsize=(15, 15))
    
    for i, (aug_class, param_dict) in enumerate(augmentations):
        param_dict['aug_method'] = aug_class
        for j, severity in enumerate(severities):
            if severity != 0:
                corrupted_img = corrupt_func([img], severity, param_dict)[0]
            else:
                corrupted_img = img
            axes[i, j].imshow(corrupted_img)
            #axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel(aug_names[i], fontsize=12, rotation=0, labelpad=40, verticalalignment='center')
    
    for j, severity in enumerate(severities):
        axes[0, j].set_title(f'Severity {severity}', fontsize=12)
    
    plt.tight_layout()
    plt.show()

    if filename is not None:
        fig.savefig(filename)

def get_image_from_url(image_url):
    # Fetch the image
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image_array = np.array(image)
    return image_array

def get_image_from_url(image_path):
    image = Image.open(image_path)
    image_np = np.array(image)
    return image_np

# =============================================================================================

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*8*8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def evaluate(model, loader, device):
    """
    Evaluate model using data from loader

    Args:
        model (torch.nn.Module): torch model
        loader (torch.Dataloader): data loader
        device (torch.device): device model is on
    Returns: 
        tuple:
            accuracy (float): percentage of correctly predicted labels
            predicted_labels (np.array): predictions output by label
            true_labels (np.array): ground truth labels
    """
    model.eval()
    correct, total = 0, 0
    predicted_labels, true_labels = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(targets.cpu().numpy())
    return 100 * correct / total, np.array(predicted_labels), np.array(true_labels)

def evaluate_noisy_indices(detected_indices, true_indices):
    """
    Evaluate performance of label noise method based on output and ground truth 

    Args:
        detected_indices (list): list of predicted indices for wrong labels 
        true_indices (list): list of actual indices where labels were changed to be wrong

    Results:
        tuple:
            precision (float)
            recall (float)
            accuracy (float)
    """
    detected_noisy_set = set(detected_indices)
    true_noisy = set(true_indices)

    correct_detections = len(detected_noisy_set.intersection(true_noisy))
    precision = correct_detections / len(detected_noisy_set) if len(detected_noisy_set) > 0 else 0
    recall = correct_detections / len(true_noisy) if len(true_noisy) > 0 else 0
    accuracy = correct_detections / len(true_noisy.union(detected_noisy_set))


    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}")

    return precision, recall, accuracy

def wrong_prediction_indice_method(test_loader, model, device):
    """
    Method to predict where the noisy labels are for indices. 

    This way uses wrong predictions, i.e. where the wrong predictions are, those indices are the noisy labels

    Args:
        test_loader (torch.Dataloader): data loader for test data
        model (torch.nn.Module): torch model
        device (torch.device): device model is on
    """
    _, predicted_labels, truth_labels = evaluate(model, test_loader, device)
    mismatched_indices = np.where(predicted_labels != truth_labels)[0]
    return mismatched_indices   

def track_loss_per_sample_multiple_epochs(model, train_loader, num_epochs, device):
    loss_per_sample_over_epochs = {i: [] for i in range(len(train_loader.dataset))}  # ✅ Initialize dictionary with all indices

    model.train()  # Ensure model is in training mode

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        batch_start_idx = 0  # ✅ Track sample index across batches

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute per-sample loss
            per_sample_loss = F.cross_entropy(outputs, labels, reduction='none')  # ✅ Fix

            # Store loss for each sample
            for i in range(len(inputs)):
                global_idx = batch_start_idx + i  # ✅ Compute global index correctly
                loss_per_sample_over_epochs[global_idx].append(per_sample_loss[i].item())

            batch_start_idx += len(inputs)  # ✅ Move index forward

    return loss_per_sample_over_epochs

def loss_indice_method(test_loader, model, device, thres=20, num_epochs=10):
    """
    Method to predict noisy indices: rudimentary loss checking

    Basic idea: loss for a sample is higher when noisier.

    Args:
        test_loader (torch.Dataloader): data loader for test data
        model (torch.nn.Module): torch model
        device (torch.device): device model is on
        num_epochs (int): number of epochs to track loss on
        thres (float): percent of indices that you guess are noisy
    """

    # Example: track loss for 10 epochs
    loss_per_sample_over_epochs = track_loss_per_sample_multiple_epochs(model, 
                                                                        test_loader, 
                                                                        num_epochs=num_epochs, 
                                                                        device=device
                                                                       )

    # Aggregate losses per sample
    sample_losses = [(idx, np.mean(losses)) for idx, losses in loss_per_sample_over_epochs.items()]
    sample_losses.sort(key=lambda x: x[1], reverse=True)  # Sort by highest loss
    num_noisy = int(len(sample_losses) * (thres/100))
    lossy_noisy_indices = [idx for idx, _ in sample_losses[:num_noisy]]
    return lossy_noisy_indices

def aum_indice_method(test_loader, model, device):
    """
    Method to detect noisy indices using area under margin

    AUM: margin determined by diff between logit of labelled class and max of logits of every other class
    taken as average over multiple epochs. negative / lower value = noisier
    
    Args:
        test_loader (torch.Dataloader): data loader for test data
        model (torch.nn.Module): torch model
        device (torch.device): device model is on
    """
    save_dir = '.'
    aum_calculator = AUMCalculator(save_dir, compressed=True)
    test_dataset = test_loader.dataset
    test_dataset_idx = DatasetWithIndex(test_dataset)
    test_loader_idx = torch.utils.data.DataLoader(test_dataset_idx, batch_size=test_loader.batch_size, shuffle=False)

    model.eval()
    for batch in test_loader_idx:
        inputs, targets, sample_ids = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        records = aum_calculator.update(logits, targets, sample_ids)

    aum_calculator.finalize()
    aum_values = pd.read_csv('aum_values.csv')
    aum_idx = [int(x.replace('tensor(', '').replace(')', '')) for x in aum_values[-2000:]['sample_id']]
    return aum_idx

def get_logits(model, dataloader, device):
    """
    Get the features, or the inputs before the last layer

    Args:
        test_loader (torch.Dataloader): data loader for test data
        model (torch.nn.Module): torch model
        device (torch.device): device model is on
    """
    labels = np.empty((0,))

    model.eval()  # Ensure the model is in evaluation mode
    with tqdm(dataloader) as progress:
        for batch_idx, (data, label) in enumerate(progress):
            data, label = data, label.long()  # No need to move to GPU, stay on CPU
            data = data.to(device)
            label = label.to(device)
            feature = model(data)  # Forward pass

            labels = np.concatenate((labels, label.cpu()))  # Ensure labels are on CPU
            if batch_idx == 0:
                features = feature.detach().cpu()  # Ensure features are on CPU
            else:
                features = np.concatenate((features, feature.detach().cpu()), axis=0)
    
    return features, labels

def fine_indice_method(test_loader, model, device, norm=True, eigen=True, thres=20):
    """
    Method to detect noisy indices: using the FINE method (filtering noisy instances via their eigenvectors)

    Create gram matrix of all features -> eigen decomposition -> inner product of each sample with it
    Lower value of alignment = higher noise

    
    Args:
        test_loader (torch.Dataloader): data loader for test data
        model (torch.nn.Module): torch model
        device (torch.device): device model is on
    """
    features, labels = get_logits(model, test_loader, device)  # Extract feature representations

    # Eigen decomposition or mean-based features
    if eigen:
        vector_dict = get_singular_vector(features, labels)
    else:
        vector_dict = get_mean_vector(features, labels)

    # Calculate alignment scores
    scores = get_score(vector_dict, features=features, labels=labels, normalization=norm)

    # Dynamically set threshold based on percentile
    threshold = np.percentile(scores, thres)  
    print(f"Threshold for clean labels: {threshold}")

    # Mark labels above threshold as clean
    clean_labels = np.where(scores > threshold)[0]
    noisy_labels = np.setdiff1d(np.arange(len(scores)), clean_labels)
    return noisy_labels

class LabelUpdate:
    def __init__(self, dataloader, model, device, args):
        self.dataloader = dataloader  # DataLoader for test or training set
        self.model = model.to(device)  # Move model to GPU if available
        self.device = device
        self.args = args

        self.count = 0  # Track epoch count
        self.num_samples = len(dataloader.dataset)
        all_labels = []
        for _, labels in dataloader:
            all_labels.extend(labels.cpu().numpy())
        self.num_classes = len(set(all_labels))

        # Store past 10 epoch predictions
        self.prediction = np.zeros((self.num_samples, 10, self.num_classes))  

        # Soft label estimates & noisy label tracking
        self.soft_labels = np.zeros((self.num_samples, self.num_classes))
        self.noisy_indices = set()  # Track detected noisy labels

    @torch.no_grad()
    def label_update(self):
        self.count += 1
        self.model.eval()  # Set model to evaluation mode

        all_preds = []
        all_targets = []
        all_indices = []

        # Run inference on dataset
        for batch_idx, (inputs, targets) in enumerate(self.dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)  # Get predictions
            probs = torch.nn.functional.softmax(outputs, dim=1)  # Convert logits to probabilities
            
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_indices.append(np.arange(batch_idx * self.dataloader.batch_size, 
                                         batch_idx * self.dataloader.batch_size + len(targets)))

        # Stack predictions & labels
        all_preds = np.vstack(all_preds)
        all_targets = np.hstack(all_targets)
        all_indices = np.hstack(all_indices)

        # Store latest predictions (sliding window of 10 epochs)
        idx = (self.count - 1) % 10
        self.prediction[all_indices, idx] = all_preds  

        # Start detecting noisy labels after a few epochs
        if self.count >= int(0.5 * self.args):
            self.soft_labels = self.prediction.mean(axis=1)  # Average soft labels
            predicted_labels = np.argmax(self.soft_labels, axis=1)

            # Detect noisy labels (where soft label disagrees with ground truth)
            for i in range(len(all_targets)):
                true_label = all_targets[i]
                pred_label = predicted_labels[all_indices[i]]

                if true_label != pred_label:
                    self.noisy_indices.add(all_indices[i])  # Flag as noisy

def label_update_indice_method(test_loader, model, device='cpu', num_epochs=50):
    """
    Method to detect noisy indices: using the label update 

    model alternates between updating network parameters and correcting labels.

    Args:
        test_loader (torch.Dataloader): data loader for test data
        model (torch.nn.Module): torch model
        device (torch.device): device model is on
        num_epochs (int): number of epochs to track loss on
    """
    label_updater = LabelUpdate(dataloader=test_loader, model=model, device=device, args=num_epochs)
    for epoch in range(num_epochs):
        print('epoch', epoch)
        label_updater.label_update()
        
    return label_updater.noisy_indices

def cleanlab_indice_method(test_loader, model, device='cpu'):
    """
    Uses cleanlab to detect noisy labels in the test set.

    Args:
        test_loader: DataLoader containing noisy test set.
        model: Trained model.
        device: CUDA/CPU device.
    """
    features, labels = get_logits(model, test_loader, device)
    labels = [int(l) for l in labels]
    pred_probs = softmax(features, axis=1) 
    ranked_label_issues = find_label_issues(
        labels,
        pred_probs,
        return_indices_ranked_by="self_confidence",
    )
    return list(ranked_label_issues[:len(ranked_label_issues)//2])

def deep_knn_indice_method(test_loader, model, device='cpu', k=500, thres=20):

    """
    Uses Deep k-NN to detect noisy labels in the test set.

    Args:
        test_loader: DataLoader containing noisy test set.
        model: Trained model.
        device: CUDA/CPU device.
        k: Number of nearest neighbors.
        threshold: Agreement threshold to consider a label noisy.

    Returns:
        noisy_indices: Indices of suspected noisy labels.
    """
    # Step 1: Extract logits (pre-softmax activations)
    logits, true_labels = get_logits(model, test_loader, device) 
    indices = np.array(list(range(len(true_labels))))

    # Step 2: Apply k-NN in logit space
    knn = NearestNeighbors(n_neighbors=min(k, len(logits)), metric="euclidean").fit(logits)
    knn_indices = knn.kneighbors(logits, return_distance=False)

    # Step 3: Compute agreement scores
    agreement_scores = np.array([
        np.sum(true_labels[neighbors] == true_labels[i]) / len(true_labels[neighbors])
        for i, neighbors in enumerate(knn_indices)
    ])

    # Step 4: Determine dynamic threshold (bottom `noise_ratio` agreement scores)
    num_noisy = int(len(agreement_scores) * thres/100  )
    threshold = np.partition(agreement_scores, num_noisy)[num_noisy]

    # Step 5: Select noisy indices
    noisy_indices = indices[agreement_scores <= threshold]

    return noisy_indices

def ensemble_method(noisy_indices_all):
    """
    Ensemble method for how to determine the final indices given the conglomerate of indices

    Args:
        noisy_indices_all (list): list of lists, each sublist has indices from a method's guess of the noisy labels
    """
    noisy_sets = [set(x) for x in noisy_indices_all]
    # Flatten the lists and count occurrences
    all_noisy_indices = [idx for s in noisy_sets for idx in s]
    count_dict = Counter(all_noisy_indices)

    # Majority voting: keep indices that appear in at least 3 out of 5 methods
    threshold = len(noisy_sets)//2  + 1 # Change this value if needed
    final_noisy_indices = {idx for idx, count in count_dict.items() if count >= threshold}

    print(f"Final Noisy Indices ({len(final_noisy_indices)} samples)")
    return final_noisy_indices

def label_noise_method(test_dataset, 
                       model, 
                       device,
                       list_of_methods, 
                       param_dict,
                       evaluate=False, 
                       noise_ratio=0.2, 
                       batch_size=128):
    
    true_noisy_indices = None
    print('label noise method: initialization')
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    np.random.seed(42)
    if evaluate:
        print("changing the indices for noisy index searching")
        class_labels = np.unique(test_dataset.targets)
        num_test_samples = len(test_dataset.targets)
        true_noisy_indices = random.sample(range(num_test_samples), int(noise_ratio * num_test_samples))
        ground_truth_labels = np.array(test_dataset.targets).copy() # Store true labels
        noisy_test_labels = ground_truth_labels.copy()

        for idx in true_noisy_indices:
            while(noisy_test_labels[idx] == ground_truth_labels[idx]):
                noisy_test_labels[idx] = random.choice(class_labels)

        test_dataset.targets = noisy_test_labels
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    param_dict['test_loader'] = test_loader
    param_dict['model'] = model
    param_dict['device'] = device

    noisy_indices_all = []
    for method in list_of_methods:
        print('method:', method)
        sig = inspect.signature(method)
        accepted_params = {
            name: param_dict[name]
            for name in sig.parameters
            if name in param_dict
        }
        noisy_indices = method(**accepted_params)
        # noisy_indices = method(test_loader, model, device)
        if evaluate:
            evaluate_noisy_indices(noisy_indices, true_noisy_indices)
            print('and that was for method:', method)
        noisy_indices_all.append(noisy_indices)
        print()
    
    return noisy_indices_all, true_noisy_indices, test_loader

# ============================================================================================

def get_album_augmentations_list():
    augmentations_album = [
        (A.RandomRain, {'slant_range': (0, 30), 
                        'drop_length': lambda s: 2*s, 
                        'drop_width': lambda s: s, 
                        'drop_color': (200, 200, 200), 
                        'blur_value': lambda s: 3 + s, 
                        'brightness_coefficient': lambda s: 1 - 0.1*s, 
                        'rain_type': 'drizzle', 
                        'p': 1.0}),
        (A.RandomSnow, {'snow_point_range': lambda s: (0.3+0.1 * s, 0.5 + 0.1* s), 
                        'brightness_coeff': lambda s: 1.0+.25*s, 
                        'p': 1.0}),
        (A.ColorJitter, {'brightness': lambda s: (0.85+0.55*s, 1+0.55*s),
                        'contrast': (1,1),
                        'saturation': (1,1),
                        'hue': (0,0),
                        'p':1}),
        (A.GaussianBlur, {'blur_limit': lambda s: (9 + 6*s, 13 + 6*s), 
                        'sigma_limit':lambda s: (0.25 + 0.25*s, 1.0 + 0.25*s),
                        'p': 1.0}),
        (A.GlassBlur, {'sigma': lambda s: s*2, 'p': 1.0}),
        (A.Defocus, {'radius': lambda s: (3*s,3*s+1), 'alias_blur': lambda s: 2*s, 'p': 1.0}),
        (A.MotionBlur, {'blur_limit': lambda s: (5+8*s, 7+12*s), 'p': 1.0}),
        (A.ZoomBlur, {'max_factor': lambda s: 1 + 0.25 * s, 'p': 1.0})
    ]
    aug_names_album = ["Random Rain", "Random Snow", "Brightness", "Gaussian Blur", "Glass Blur", "Defocus Blur", "Motion Blur", "Zoom Blur"]
    return augmentations_album, aug_names_album

def get_nrtk_augmentations_list(seed=42):
    perturbations = [
        (SaltNoisePerturber, {'rng': seed, 'amount': lambda s: 0.15 * s}),
        (PepperNoisePerturber, {'rng': np.random.default_rng(seed), 'amount': lambda s: 0.15 * s}),
        (SaltAndPepperNoisePerturber, {'rng': np.random.default_rng(seed), 'amount': lambda s: 0.15 * s}),
        (GaussianNoisePerturber, {'rng': seed, 'mean': lambda s: 0.1 * s, 'var': lambda s: 0.01 * s}),
        (SpeckleNoisePerturber, {'rng': seed, 'mean': lambda s: 0.3 * s, 'var': lambda s: 0.01 * s}),
        (AverageBlurPerturber, {'ksize': lambda s: 11 + 2 * s}),
        (GaussianBlurPerturber, {'ksize': lambda s: 11 + 4 * s}),
        (MedianBlurPerturber, {'ksize': lambda s: 11 + 2 * s}),
        (BrightnessPerturber, {'factor': lambda s: 1 + 0.15 * s}),
        (ColorPerturber, {'factor': lambda s: 1 - 0.15 * s}),
        (ContrastPerturber, {'factor': lambda s: 1 + 0.2 * s}),
        (SharpnessPerturber, {'factor': lambda s: 1 - 0.2 * s}),
    ]
    perturb_names = ["Salt Noise", "Pepper Noise", "Salt Pepper Noise", "Gaussian Noise", "Speckle Noise", \
                     "Average Blur", "Gaussian Blur", "Median Blur", "Brightness", "Color", "Contrast", "Sharpness"]
    return perturbations, perturb_names

def get_augly_augmentations_list():
    augmentations = [
        (blur, {'radius': lambda s: s}),  # Corrected: Replaced 'severity' with lambda
        (brightness, {'factor': lambda s: 1 + 0.1 * s}),
        (contrast, {'factor': lambda s: s}),  # Corrected
        (random_noise, {'mean': lambda s: 0.0001 * s, 'var': lambda s: 0.00001})
    ]
    aug_str = ['blur', 'brightness', 'contrast', 'random_noise']
    return augmentations, aug_str

def get_imagecorrupt_augmentations_list():
    # gaussian_noise, shot_noise, impulse_noise, defocus_blur,
    #                 glass_blur, motion_blur, zoom_blur, snow, frost, fog,
    #                 brightness, contrast, elastic_transform, pixelate,
    #                 jpeg_compression, speckle_noise, gaussian_blur, spatter,
    #                 saturate
    augmentations = [
        (corrupt, {'corrname': 'gaussian_noise'}),
        (corrupt, {'corrname': 'fog'}),
        (corrupt, {'corrname': 'brightness'}),
        (corrupt, {'corrname': 'zoom_blur'}),
    ]
    aug_str = ['gaussian_noise', 'fog', 'brightness', 'zoom_blur']
    return augmentations, aug_str

def get_corruption_helpers(library='albumentations'):
    if library in ['albumentations', 'album']:
        a1, a2 = get_album_augmentations_list()
        return a1, a2, corrupt_func_album
    if library in ['nrtk']:
        n1, n2 = get_nrtk_augmentations_list()
        return n1, n2, corrupt_func_nrtk
    if library in ['imagecorruptions', 'imagecorr', 'imagecorrupt', 'ic', 'imcor']:
        i1, i2 = get_imagecorrupt_augmentations_list()
        return i1, i2, corrupt_func_imagecorrupt
    if library in ['augly']:
        u1, u2 = get_augly_augmentations_list()
        return u1, u2, corrupt_func_imagecorrupt
    else:
        raise ValueError('Invalid library called')

def get_corrupted_dataloader(testloader, corr_func, severity=1, corr_kwargs=None):
    if corr_kwargs is None:
        corr_kwargs = {}  # Default to an empty dictionary
    
    corrupted_images = []
    corrupted_labels = []
    
    for images, labels in testloader:
        images_np = (images * 255).byte().numpy().transpose(0, 2, 3, 1)  # Convert to HWC format and uint8
        
        # Apply corruption function with provided parameters
        corrupted = corr_func(images_np, severity, corr_kwargs)
        
        corrupted = torch.tensor(corrupted.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0  # Convert back to CHW format and normalize
        corrupted_images.append(corrupted)
        corrupted_labels.append(labels)
    
    corrupted_dataset = torch.utils.data.TensorDataset(torch.cat(corrupted_images), torch.cat(corrupted_labels))
    return torch.utils.data.DataLoader(corrupted_dataset, batch_size=128, shuffle=False)

def best_fit_gradient(x_values, y_values):
    """
    Calculate the gradient (slope) of the best-fit line using the least squares method.
    
    Parameters:
    x_values (list or array): Independent variable values.
    y_values (list or array): Dependent variable values.
    
    Returns:
    float: Slope of the best-fit line.
    """
    x_mean = np.mean(x_values)
    y_mean = np.mean(y_values)
    
    numerator = np.sum((x_values - x_mean) * (y_values - y_mean))
    denominator = np.sum((x_values - x_mean) ** 2)
    
    return numerator / denominator

def augmentation_gradient(model, test_loader, device, corr_func, plot_graphs=False, corr_kwargs=None):
    
    print(f"Evaluating on severity 0...")
    base_acc, _ , _ = evaluate(model, test_loader, device)
    print(f"Accuracy at severity 0: {base_acc:.4f}")
    severities = [0, 1, 2, 3, 4, 5]
    accuracies = [base_acc]
    for severity in severities[1:]:
        print(f"Evaluating on severity {severity}...")
        corrupted_loader = get_corrupted_dataloader(test_loader, 
                                                    corr_func, 
                                                    severity=severity,
                                                    corr_kwargs=corr_kwargs)
        acc, _,_ = evaluate(model, corrupted_loader, device)
        accuracies.append(acc)
        print(f"Accuracy at severity {severity}: {acc:.4f}")

    # Plot results
    if plot_graphs:
        plt.figure(figsize=(8, 5))
        plt.plot(severities, accuracies, marker='o', linestyle='-', color='b')
        plt.xlabel("Severity")
        plt.ylabel("Accuracy")
        plt.title("Model Accuracy vs Severity")
        plt.xticks(severities)
        plt.grid(True)
        plt.show()    
    return best_fit_gradient(severities, accuracies), accuracies

def plot_accuracy_vs_severity(accuracies):
    severities = list(range(len(accuracies)))
    plt.figure(figsize=(8, 5))
    plt.plot(severities, accuracies, marker='o', linestyle='-', color='b')
    plt.xlabel("Severity")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy vs Severity")
    plt.xticks(severities)
    plt.grid(True)
    plt.show()  

def generic_combination_gradients(
    model, 
    test_loader, 
    device, 
    augmentation_func_wrapper,
    augmentation_list,
    augmentation_str,
    plot_graphs=False
    ):

    assert len(augmentation_list) == len(augmentation_str)

    model = model.to(device)
    augmentation_dict = {}
    for k, (m,d) in zip(augmentation_str, augmentation_list):
        d['aug_method'] = m
        gradient, accuracies = augmentation_gradient(model, test_loader, device, augmentation_func_wrapper, plot_graphs, d)
        first_drop = accuracies[1] - accuracies[0]
        
        print(accuracies, first_drop)
        print(k, 'augmentation method gradient:', gradient)
        augmentation_dict[k] = (gradient, first_drop)
        
        print()
    return augmentation_dict                                                                                                                                                                                                                                                                                                           

# =============================================================================================

def get_partially_corrupted_imagecorr_dataloader(testloader):
    aug_methods = ['gaussian_noise', 'fog', 'brightness', 'zoom_blur']
    
    corrupted_images = []
    corrupted_labels = []
    
    for images, labels in testloader:

        if np.random.random() > 0.8:
            corrupted = images
        else:
            images_np = (images * 255).byte().numpy().transpose(0, 2, 3, 1)  # Convert to HWC format and uint8
            
            random_index = int(4 * np.random.random())
            aug_method = aug_methods[random_index]
            
            
            # Apply corruption function with provided parameters
            corr_kwargs = {'corrname': aug_method}
            corrupted = corrupt_func_imagecorrupt(images_np, 1, corr_kwargs)

            corrupted = torch.tensor(corrupted.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0  # Convert back to CHW format and normalize
        corrupted_images.append(corrupted)
        corrupted_labels.append(labels)
        corrupted_images.append(images)
        corrupted_labels.append(labels)
    
    corrupted_dataset = torch.utils.data.TensorDataset(torch.cat(corrupted_images), torch.cat(corrupted_labels))
    return torch.utils.data.DataLoader(corrupted_dataset, batch_size=testloader.batch_size, shuffle=False)

def train_model(model, optimizer, criterion, train_loader, num_epochs=15):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss / len(train_loader):.4f}")

def retrain_experiment(model, train_loader, test_loader, device, get_corrupted_dataloader_func):
    # Assume you have an unknown model
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("making copies of model")
    model_copy = copy.deepcopy(model)
    model_no_weights = copy.deepcopy(model)
    model_no_weights.apply(
        lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None
    )
    train_loader_corr = get_corrupted_dataloader_func(train_loader)
    
    print("training new models")
    optimizer = optim.Adam(model_copy.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_model(model_copy, optimizer, criterion, train_loader_corr, num_epochs=15)
    optimizer = optim.Adam(model_no_weights.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_model(model_no_weights, optimizer, criterion, train_loader_corr, num_epochs=15)
    
    print("evaluating new models")
    test_loader_corr = get_corrupted_dataloader_func(test_loader)
    
    test_acc_dict = {}
    test_acc_dict['original_model'] = {}
    test_acc_dict['copy_model'] = {}
    test_acc_dict['scratch_model'] = {}
    acc, _, _ = evaluate(model, test_loader, device); test_acc_dict['original_model']['non_corrupted'] = acc
    acc, _, _ = evaluate(model, test_loader_corr, device); test_acc_dict['original_model']['corrupted'] = acc
    acc, _, _ = evaluate(model_copy, test_loader, device); test_acc_dict['copy_model']['non_corrupted'] = acc
    acc, _, _ = evaluate(model_copy, test_loader_corr, device); test_acc_dict['copy_model']['corrupted'] = acc    
    acc, _, _ = evaluate(model_no_weights, test_loader, device); test_acc_dict['scratch_model']['non_corrupted'] = acc
    acc, _, _ = evaluate(model_no_weights, test_loader_corr, device); test_acc_dict['scratch_model']['corrupted'] = acc
    
    return model_copy, model_no_weights, test_acc_dict

# =============================================================================================

if __name__ == "__main__":

    device = torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load('label_noise_simplecnn.h5', weights_only=True))

    list_of_methods = [ wrong_prediction_indice_method,
                        loss_indice_method,
                        aum_indice_method,
                        fine_indice_method,
                        label_update_indice_method,
                        cleanlab_indice_method,
                        deep_knn_indice_method]
    with open('config.yaml', 'r') as file:
        param_dict = yaml.safe_load(file)
        
    noisy_indices_all, true_noisy_indices, test_loader = label_noise_method(test_dataset, 
                                                                            model, 
                                                                            device,
                                                                            list_of_methods, 
                                                                            param_dict,
                                                                            evaluate=True, 
                                                                            noise_ratio=0.2, 
                                                                            batch_size=128)

    print('time to ensemble the indices!')
    final_noisy_indices = ensemble_method(noisy_indices_all)
    evaluate_stats = evaluate_noisy_indices(final_noisy_indices, true_noisy_indices)
    print("evaluating noisy indices:", evaluate_stats)

    # ===============================================================================================================

    print("noisy indices done! let's go to: robustness evaluation")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load('label_noise_simplecnn.h5', weights_only=True))

    clean_test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    clean_test_loader = torch.utils.data.DataLoader(clean_test_dataset, batch_size=128, shuffle=False)

    augmentation_list, augmentation_str, corrupt_func = get_corruption_helpers('nrtk')
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

    # ===============================================================================================================

    print("done with evaluation! lastly: retraining")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    print("Training model on clean train data...")
    model_copy, model_no_weights, test_acc_dict = retrain_experiment(model, 
                                                                    train_loader, 
                                                                    clean_test_loader, 
                                                                    device,
                                                                    get_partially_corrupted_imagecorr_dataloader)

    print(test_acc_dict)

    # =))
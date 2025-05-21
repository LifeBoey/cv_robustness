#Import libraries and data and model
import torch
import yaml
import torch.nn as nn
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
import inspect

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
        loader (torch.utils.data.DataLoader): data loader
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
        test_loader (torch.utils.data.DataLoader): data loader for test data
        model (torch.nn.Module): torch model
        device (torch.device): device model is on
    """
    _, predicted_labels, truth_labels = evaluate(model, test_loader, device)
    mismatched_indices = np.where(predicted_labels != truth_labels)[0]
    return mismatched_indices   

def track_loss_per_sample_multiple_epochs(model, train_loader, num_epochs, device):

    loss_per_sample_over_epochs = {i: [] for i in range(len(train_loader.dataset))}  # ✅ Initialize dictionary with all indices

    model.eval()
    with torch.no_grad():
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
        test_loader (torch.utils.data.DataLoader): data loader for test data
        model (torch.nn.Module): torch model
        device (torch.device): device model is on
        num_epochs (int): number of epochs to track loss on
        thres (float): percent of indices that you guess are noisy
    """
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

def aum_indice_method(test_loader, model, device, thres=20):
    """
    Method to detect noisy indices using area under margin

    AUM: margin determined by diff between logit of labelled class and max of logits of every other class
    taken as average over multiple epochs. negative / lower value = noisier
    
    Args:
        test_loader (torch.utils.data.DataLoader): data loader for test data
        model (torch.nn.Module): torch model
        device (torch.device): device model is on
    """
    save_dir = '.'
    aum_calculator = AUMCalculator(save_dir, compressed=True)
    test_dataset = test_loader.dataset
    threshold = int(len(test_dataset) * thres/100)
    test_dataset_idx = DatasetWithIndex(test_dataset)
    test_loader_idx = torch.utils.data.DataLoader(test_dataset_idx, batch_size=test_loader.batch_size, shuffle=False)

    model.eval()
    with torch.no_grad():
        for batch in test_loader_idx:
            inputs, targets, sample_ids = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            records = aum_calculator.update(logits, targets, sample_ids)

    aum_calculator.finalize()
    aum_values = pd.read_csv('aum_values.csv')
    aum_idx = [int(x.replace('tensor(', '').replace(')', '')) for x in aum_values[-threshold:]['sample_id']]
    return aum_idx

def get_logits(model, dataloader, device):
    """
    Get the features, or the inputs before the last layer

    Args:
        test_loader (torch.utils.data.DataLoader): data loader for test data
        model (torch.nn.Module): torch model
        device (torch.device): device model is on

    Returns: 
        logits (np.array): array of outputs just before they pass through the softmax/max/last layer for prediction
    """
    labels = np.empty((0,))

    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
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
        test_loader (torch.utils.data.DataLoader): data loader for test data
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
        test_loader (torch.utils.data.DataLoader): data loader for test data
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
        test_loader (torch.utils.data.DataLoader): data loader for test data
        model (torch.nn.Module): torch model
        device (torch.device): device model is on
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

def ensemble_method(noisy_indices_all, strictness=1):
    """
    Ensemble method for how to determine the final indices given the conglomerate of indices

    (simple majority voting, no weighting, can KIV different ensemble methods)

    Args:
        noisy_indices_all (list): list of lists, each sublist has indices from a method's guess of the noisy labels
        strictness (int): how strict you want to be in ensembling (simple way); higher = stricter; can be negative
    """
    noisy_sets = [set(x) for x in noisy_indices_all]
    # Flatten the lists and count occurrences
    all_noisy_indices = [idx for s in noisy_sets for idx in s]
    count_dict = Counter(all_noisy_indices)

    threshold = len(noisy_sets)//2  + strictness
    final_noisy_indices = {idx for idx, count in count_dict.items() if count >= threshold}

    print(f"Final Noisy Indices ({len(final_noisy_indices)} samples)")
    return final_noisy_indices

def get_all_label_noise_methods():
    """
    Returns all of the label noise methods as defined above. 
    
    If you want to have a different order or different types of indice methods, HERE is where you choose it.
    """
    return [label_update_indice_method, 
            wrong_prediction_indice_method,
            loss_indice_method,
            aum_indice_method,
            fine_indice_method,
            cleanlab_indice_method,
            deep_knn_indice_method]

def label_noise_correction(model, device, final_noisy_indices, test_dataset):
    from torchvision.utils import save_image  
    import os
    os.makedirs("noisy_images", exist_ok=True)
    model.eval()
    model.to(device)
    correct_dict = {'filenames': [],
                    'noisy_labels': [],
                    'corrected_labels': []}
    for idx in final_noisy_indices:
        # Get image and label
        image, label = test_dataset[idx]

        # Optional: get filename if your dataset supports it
        if hasattr(test_dataset, "samples"):  # ImageFolder
            filename = test_dataset.samples[idx][0]
        elif hasattr(test_dataset, "imgs"):  # Some torchvision versions use this
            filename = test_dataset.imgs[idx][0]
        elif hasattr(test_dataset, "get_filename"):  # Custom dataset with a method
            filename = test_dataset.get_filename(idx)
        else:
            filename = f"noisy_images/noisy_image_{idx}.png"
            # Make sure image is in [0,1] range for save_image
            save_image(image, filename)


        # Run image through model
        image_tensor = image.unsqueeze(0).to(device)  # Add batch dimension
        with torch.no_grad():
            outputs = model(image_tensor)
            predicted_label = torch.argmax(outputs, dim=1).item()

        print(f"File: {filename} | Original Label: {label} | Predicted Label: {predicted_label}")  
        correct_dict['filenames'].append(filename)
        correct_dict['noisy_labels'].append(label)
        correct_dict['corrected_labels'].append(predicted_label)  

    return correct_dict

def label_noise_method(test_dataset, 
                       model, 
                       device,
                       list_of_methods, 
                       param_dict,
                       evaluate=False, 
                       noise_ratio=0.2, 
                       batch_size=128):
    """
    Applies one or more label noise detection methods to a dataset using a given model.

    Optionally corrupts the test dataset with synthetic label noise for evaluation purposes,
    then runs each method in `list_of_methods` to detect noisy labels.

    evaluate=False for actual detection of given dataset
    evaluate=True for when we want to test how good the model is at detecting the noise when we know it's noise.

    Args:
        test_dataset (torch.utils.data.Dataset): The dataset on which to perform noise detection.
        model (torch.nn.Module): The model used by noise detection methods.
        device (torch.device or str): The device to run the model on.
        list_of_methods (List[Callable]): A list of functions/methods that detect noisy labels.
        param_dict (dict): A dictionary of parameters to pass to each method in `list_of_methods`.
            Must include any keys required by those methods (e.g., 'test_loader', 'model', 'device').
        evaluate (bool, optional): Whether to inject synthetic noise into the labels for evaluation.
            If True, returns ground-truth noisy indices. Defaults to False.
        noise_ratio (float, optional): Fraction of labels to corrupt if `evaluate` is True. Defaults to 0.2.
        batch_size (int, optional): Batch size for the test data loader. Defaults to 128.

    Returns:
        Tuple[List[np.ndarray], Optional[List[int]], torch.utils.data.DataLoader]:
            - A list containing noisy index arrays from each method.
            - The list of true noisy indices (if `evaluate=True`), otherwise None.
            - The test DataLoader used in detection.
    """
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

# =============================================================================================

if __name__ == "__main__":

    device = torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load('label_noise_simplecnn.h5', weights_only=True))

    list_of_methods = get_all_label_noise_methods()[:5]
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
    correct_dict = label_noise_correction(model, device, final_noisy_indices, test_dataset)
    print("evaluating noisy indices:", evaluate_stats)
    print()
    print("corrected dictionary")
    print(correct_dict)

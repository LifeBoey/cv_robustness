#Import libraries and data and model
import torch
import yaml
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image 
import random
import torch.nn.functional as F
import numpy as np
from aum import AUMCalculator, DatasetWithIndex
import pandas as pd
from FINE_official.dynamic_selection.selection.svd_classifier import get_score, get_singular_vector, get_mean_vector
from cleanlab.filter import find_label_issues
from scipy.special import softmax
from sklearn.neighbors import NearestNeighbors
from collections import Counter, defaultdict
import inspect
from scipy.stats import skew
import os
from cvrob_util import get_logits, evaluate

# =============================================================================================

def evaluate_noisy_indices(detected_indices, true_indices):
    """
    Evaluate the performance of a noisy label detection method.

    This function compares the set of detected noisy label indices with the ground truth indices
    of actual noisy labels, and computes three evaluation metrics:
    
    - Precision: Proportion of detected noisy labels that are actually noisy.
    - Recall: Proportion of true noisy labels that were successfully detected.
    - Jaccard Index (called 'accuracy' here): Overlap between detected and true noisy labels.

    Args:
        detected_indices (list or set): Indices predicted to have noisy (incorrect) labels.
        true_indices (list or set): Ground truth indices of actual noisy labels.

    Returns:
        tuple:
            precision (float): Correct detections / total detected.
            recall (float): Correct detections / total actual noisy labels.
            jaccard_index (float): Intersection over union of detected and true noisy sets.
    """
    detected_noisy_set = set(detected_indices)
    true_noisy_set = set(true_indices)

    correct_detections = len(detected_noisy_set.intersection(true_noisy_set))
    precision = correct_detections / len(detected_noisy_set) if detected_noisy_set else 0.0
    recall = correct_detections / len(true_noisy_set) if true_noisy_set else 0.0
    jaccard_index = correct_detections / len(detected_noisy_set.union(true_noisy_set)) if (detected_noisy_set or true_noisy_set) else 0.0

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {jaccard_index:.4f}")

    correct_detections = len(detected_noisy_set.intersection(true_noisy_set))
    precision = correct_detections / len(detected_noisy_set) if detected_noisy_set else 0.0
    recall = correct_detections / len(true_noisy_set) if true_noisy_set else 0.0
    jaccard_index = correct_detections / len(detected_noisy_set.union(true_noisy_set)) if (detected_noisy_set or true_noisy_set) else 0.0

    return precision, recall, jaccard_index

def wrong_prediction_indice_method(test_loader, model, device):
    """
    Method to predict where the noisy labels are for indices. 

    This way uses wrong predictions, i.e. where the wrong predictions are, those indices are the noisy labels

    Args:
        test_loader (torch.utils.data.DataLoader): data loader for test data
        model (torch.nn.Module): torch model
        device (torch.device): device model is on

    Returns:
        np.array: List of predicted noisy labels
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

    Returns:
        Tuple:
            np.array: List of predicted noisy labels
            np.array: List of (normalized) weighted scores for confidence of predictions of noisy labels
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
    scores = [x[1] for x in sample_losses]
    norm_scores = auto_normalize_weights(scores)[:num_noisy]
    return lossy_noisy_indices, norm_scores

def aum_indice_method(test_loader, model, device, thres=20):
    """
    Method to detect noisy indices using area under margin

    AUM: margin determined by diff between logit of labelled class and max of logits of every other class
    taken as average over multiple epochs. negative / lower value = noisier
    
    Args:
        test_loader (torch.utils.data.DataLoader): data loader for test data
        model (torch.nn.Module): torch model
        device (torch.device): device model is on

    Returns:
        tuple:
            noisy_labels (np.array): array of indices identified to be noisy
            noisy_scores (np.array): array of weights / scores assigned to labels for confidence in noisy prediction
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
    aum_idx = np.array([int(x.replace('tensor(', '').replace(')', '')) for x in aum_values[-threshold:]['sample_id']]) 
    normalized_scores = auto_normalize_weights(np.array(aum_values[:]['aum']))
    aum_scores = normalized_scores[-threshold:]
    # aum_scores = np.array(aum_values[-threshold:]['aum'])
    return aum_idx, aum_scores

def fine_indice_method(test_loader, model, device, norm=True, eigen=True, thres=20):
    """
    Method to detect noisy indices: using the FINE method (filtering noisy instances via their eigenvectors)

    Create gram matrix of all features -> eigen decomposition -> inner product of each sample with it
    Lower value of alignment = higher noise

    Args:
        test_loader (torch.utils.data.DataLoader): data loader for test data
        model (torch.nn.Module): torch model
        device (torch.device): device model is on

    Returns:
        tuple:
            noisy_labels (np.array): array of indices identified to be noisy
            noisy_scores (np.array): array of weights / scores assigned to labels for confidence in noisy prediction
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
    normalized_scores = auto_normalize_weights(scores)
    noisy_scores = normalized_scores[noisy_labels]
    return noisy_labels, noisy_scores

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

    Returns:
        np.array: List of predicted noisy labels
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

    Returns:
        Tuple:
            np.array: List of predicted noisy labels
            np.array: List of (normalized) weighted scores for confidence of predictions of noisy labels
    """
    features, labels = get_logits(model, test_loader, device)
    labels = [int(l) for l in labels]
    pred_probs = softmax(features, axis=1) 
    ranked_label_issues = find_label_issues(
        labels,
        pred_probs,
        return_indices_ranked_by="self_confidence",
    )
    noisy_indices = list(ranked_label_issues[:len(ranked_label_issues)//2])
    weights = [len(labels) - x for x in range(len(labels))]
    weights = auto_normalize_weights(weights)[:len(noisy_indices)]
    #weights = [len(noisy_indices) - x for x in range(len(noisy_indices))]
    return noisy_indices, weights

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
    normalized_scores = auto_normalize_weights(agreement_scores)
    agreement_scores2 = normalized_scores[agreement_scores <= threshold]

    return noisy_indices, agreement_scores2

def ensemble_method_voting(noisy_indices_all, strictness=1):
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
    return final_noisy_indices

def ensemble_method_weighted(noisy_indices_all, weight_scores, strictness):
    """
    Ensemble method for how to determine the final indices given the conglomerate of indices

    (weighted voting, assumes all the weights are normalized)

    Args:
        noisy_indices_all (list): list of lists, each sublist has indices from a method's guess of the noisy labels
        weight_scores (list of floats, optional): Confidence or performance score for each method.
        strictness (float): how strict you want to be in ensembling; value from 0-1, higher = stricter/smaller result

    Returns:
        set: Final predicted noisy indices after ensembling.
    """
    vote_scores = defaultdict(float)

    # Accumulate scores per index
    for method_indices, method_weights in zip(noisy_indices_all, weight_scores):
        if len(method_indices) != len(method_weights):
            raise ValueError("Each method's indices and weights must have the same length")
        for idx, score in zip(method_indices, method_weights):
            vote_scores[idx] += float(score)

    max_score = max(float(v) for v in vote_scores.values()) if vote_scores else 0
    threshold = max_score * strictness

    final_noisy_indices = {idx for idx, score in vote_scores.items() if score >= threshold}
    return final_noisy_indices

def ensemble_method_general(noisy_indices_all, weight_scores=None, strictness=1, weight_ensemble=False):
    """
    Ensemble method to determine final noisy label indices from multiple methods.

    Supports simple majority voting or weighted voting using method confidence scores.

    Args:
        noisy_indices_all (list of lists): Each sublist contains predicted noisy indices from one method.
        weight_scores (list of floats, optional): Confidence or performance score for each method.
        strictness (int or float): Degree of strictness or tuning of threshold
        weight_ensemble (bool): Whether to use weights for voting instead of majority.

    Returns:
        set: Final predicted noisy indices after ensembling.
    """
    num_methods = len(noisy_indices_all)

    if weight_ensemble:
        if weight_scores is None:
            raise ValueError("weight_scores must be provided when weight_ensemble=True")
        if len(weight_scores) != num_methods:
            raise ValueError("Length of weight_scores must match number of methods")

        final_noisy_indices = ensemble_method_weighted(noisy_indices_all, weight_scores, strictness)

    else:
        final_noisy_indices = ensemble_method_voting(noisy_indices_all, strictness)

    print(f"Final Noisy Indices ({len(final_noisy_indices)} samples)")
    return sorted(final_noisy_indices)

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
    """
    Corrects noisy labels by using the model's predictions for suspected noisy samples.

    This function processes the suspected noisy samples (identified by `final_noisy_indices`),
    passes each through the model to get a corrected label prediction, and optionally saves
    the image to disk for inspection. It returns a dictionary containing filenames, original
    (noisy) labels, and corrected labels.

    Args:
        model (torch.nn.Module): Trained PyTorch classification model.
        device (str or torch.device): The device on which computation should be performed (e.g., "cpu" or "cuda").
        final_noisy_indices (list of int): Indices of samples in the dataset suspected to have noisy labels.
        test_dataset (torch.utils.data.Dataset): The dataset containing samples and (possibly incorrect) labels.

    Returns:
        dict: A dictionary with keys:
            - 'filenames' (list of str): Filenames or generated identifiers for the noisy samples.
            - 'noisy_labels' (list of int): Original (potentially incorrect) labels from the dataset.
            - 'corrected_labels' (list of int): Labels predicted by the model for the corresponding samples.

    Notes:
        - If the dataset supports filename access via `.samples`, `.imgs`, or a custom `.get_filename(idx)` method,
          the filenames are retrieved. Otherwise, images are saved into a `noisy_images/` folder.
        - Requires torchvision's `save_image` if image saving is needed for fallback identifiers.
    """
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

    noisy_indices_all = []; weight_scores_all = []
    for method in list_of_methods:
        print('method:', method)
        sig = inspect.signature(method)
        accepted_params = {
            name: param_dict[name]
            for name in sig.parameters
            if name in param_dict
        }
        noisy_indices = method(**accepted_params)
        if type(noisy_indices) == tuple and len(noisy_indices) == 2:
            noisy_indices, weight_scores = noisy_indices
        else:
            weight_scores =  np.ones(len(noisy_indices))
        # noisy_indices = method(test_loader, model, device)
        if evaluate:
            evaluate_noisy_indices(noisy_indices, true_noisy_indices)
            print('and that was for method:', method)
        noisy_indices_all.append(noisy_indices)
        weight_scores_all.append(weight_scores)
        print()
    
    return noisy_indices_all, weight_scores_all, true_noisy_indices, test_loader                                                                                                                                                                                                                                                                                              

def auto_normalize_weights(scores):
    """
    Automatically normalize a list/array of weight scores to [0,1] based on heuristics
    about their distribution and value range.

    Args:
        scores (list / np.array): list of weighted scores (unnormalized)

    Returns:
        np.array: normalized weights
    """
    scores = np.array(scores, dtype=np.float64)
    
    # Basic stats
    s_min = np.min(scores)
    s_max = np.max(scores)
    s_mean = np.mean(scores)
    s_std = np.std(scores)
    s_skew = skew(scores)

    # Heuristic rules:
    # 1. If negative values present, assume log scores -> exponentiate then min-max normalize
    if s_min < 0:
        scores = np.exp(scores - s_min)  # shift to avoid too small exps if very negative
        scores = (scores - np.min(scores)) / (np.ptp(scores) + 1e-8)
        return scores

    # 2. If values mostly between 0 and 1, assume already probabilities or bounded scores
    if s_min >= 0 and s_max <= 1.05:
        # Just clip to [0,1]
        return np.clip(scores, 0, 1)

    # 3. If large range or large mean or high skew, assume arbitrary raw scores -> min-max normalize
    if (s_max - s_min) > 1 or s_mean > 1 or abs(s_skew) > 1:
        scores = (scores - s_min) / (s_max - s_min + 1e-8)
        return scores

    # 4. Default fallback: min-max normalize (covers most cases)
    scores = (scores - s_min) / (s_max - s_min + 1e-8)
    return scores

# =============================================================================================

if __name__ == "__main__":
    from cvrob_util import SimpleCNN
    device = torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load('label_noise_simplecnn.h5', weights_only=True))

    list_of_methods = get_all_label_noise_methods()[:5]
    with open('config.yaml', 'r') as file:
        param_dict = yaml.safe_load(file)
        
    noisy_indices_all, weight_scores, true_noisy_indices, test_loader = label_noise_method(test_dataset, \
                                                                                            model, \
                                                                                            device,\
                                                                                            list_of_methods, \
                                                                                            param_dict,\
                                                                                            evaluate=True, \
                                                                                            noise_ratio=0.2, \
                                                                                            batch_size=128)

    print('time to ensemble the indices!')
    final_noisy_indices = ensemble_method_general(noisy_indices_all, weight_scores)
    evaluate_stats = evaluate_noisy_indices(final_noisy_indices, true_noisy_indices)
    correct_dict = label_noise_correction(model, device, final_noisy_indices, test_dataset)
    print("evaluating noisy indices:", evaluate_stats)
    print()
    for l in weight_scores:
        print(l)
        print()
    # print("corrected dictionary")
    # print(correct_dict)

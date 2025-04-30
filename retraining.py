import numpy as np
from augmentations import corrupt_func_imagecorrupt
from cvrob_util import evaluate
import torch
import torch.optim as optim
import torch.nn as nn
import copy

def get_partially_corrupted_dataloader_generic(testloader, corr_func, augmentation_list, augmentation_str, severity=1, corr_kwargs=None):
    """
    Make a dataloader with data that is of the augmented/corrupted version of the original dataloader

    Args:
        testloader (torch.utils.data.DataLoader): original dataloader
        corr_func (function): corruption function  that takes in ([numpy array of images], severity, corr_kwargs)
        severity (int); extent of severity between 0-5.
        corr_kwargs (dict): dictionary of extra parameters to send into corr_func

    Returns:
        corrupted_dataloader (torch.utils.data.DataLoader): corrupted version of original dataloader
    """
    if corr_kwargs is None:
        corr_kwargs = {}  # Default to an empty dictionary
    
    corrupted_images = []
    corrupted_labels = []
    
    for images, labels in testloader:
        if np.random.random() > (1 - 1/len(augmentation_list)):
            corrupted = images
        else:
            images_np = (images * 255).byte().numpy().transpose(0, 2, 3, 1)  # Convert to HWC format and uint8
            random_index = int(len(augmentation_list) * np.random.random())
            aug_method = augmentation_list[random_index]
            aug_str = augmentation_str[random_index]
            aug_method['aug_method'] = aug_str

            # Apply corruption function with provided parameters
            corrupted = corr_func(images_np, severity, aug_method)
            
            corrupted = torch.tensor(corrupted.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0  # Convert back to CHW format and normalize
        corrupted_images.append(corrupted)
        corrupted_labels.append(labels)
    
    corrupted_dataset = torch.utils.data.TensorDataset(torch.cat(corrupted_images), torch.cat(corrupted_labels))
    return torch.utils.data.DataLoader(corrupted_dataset, batch_size=128, shuffle=False)

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

def train_model(model, optimizer, criterion, train_loader, device, num_epochs=15):
    """
    Trains a PyTorch model for a specified number of epochs.

    Iterates over the training data and updates model parameters using the provided
    optimizer and loss criterion. Prints average loss per epoch.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters.
        criterion (Callable): The loss function used to compute the training loss.
        train_loader (torch.utils.data.DataLoader): DataLoader providing the training data.
        device (torch.device or str): Device on which training is performed (e.g., 'cuda' or 'cpu').
        num_epochs (int, optional): Number of training epochs. Defaults to 15.

    Returns:
        None
    """
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
    """
    Performs a retraining experiment to compare model performance under label corruption.

    The function creates two model variants:
    - A deep copy of the original model with weights preserved.
    - A fresh copy with reinitialized weights.

    Both models are trained on a corrupted version of the training data and evaluated
    on both clean and corrupted test data. The original model is also evaluated for comparison.

    Args:
        model (torch.nn.Module): The original trained model to be evaluated and copied.
        train_loader (torch.utils.data.DataLoader): DataLoader for clean training data.
        test_loader (torch.utils.data.DataLoader): DataLoader for clean test data.
        device (torch.device or str): Device to use for computation (e.g., 'cuda' or 'cpu').
        get_corrupted_dataloader_func (Callable): A function that takes a DataLoader and returns
            a version with corrupted labels.

    Returns:
        Tuple[torch.nn.Module, torch.nn.Module, Dict[str, Dict[str, float]]]:
            - A model trained from the original weights (`model_copy`).
            - A model trained from scratch with reinitialized weights (`model_no_weights`).
            - A dictionary mapping model versions to their test accuracies on clean and corrupted test sets.
    """
    #model = model.to(device)
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
    train_model(model_copy, optimizer, criterion, train_loader_corr, device, num_epochs=15)
    optimizer = optim.Adam(model_no_weights.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_model(model_no_weights, optimizer, criterion, train_loader_corr, device, num_epochs=15)
    
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
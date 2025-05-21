import torch 
import numpy as np
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
import plotly.graph_objects as go
from augly.image import blur, brightness, random_noise, contrast, color_jitter, pixelization, sharpen
from augly.image import aug_np_wrapper
from cvrob_util import (plot_accuracy_vs_severity,
                        evaluate,
                        best_fit_gradient)
from plotly.subplots import make_subplots
from imagecorruptions import corrupt
import matplotlib.pyplot as plt

def corrupt_func_album(images_np, severity, param_dict):
    """Corrupts all the images provided with the given severity of augmentation

    This is for corrupt function: albumentations. You can apply the same template for the rest.

    Args:
        images_np (np.array): array of images (also np array) to be augmented
        severity (int): severity from 1-5
        param_dict (dict): dictionary of all parameters

    Returns:
        corrupted_imgs (np.array): array of augmented images
    """
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

def corrupt_and_plot_generic_plotly(img, augmentations, aug_names, corrupt_func, filename=None):
    """Method for corrupting (read: one!) an image and plotting it using: plotly

    Takes the base image, and severity=1 and severity=2 images.

    Args:
        img (np.array): np.array representing image
        augmentations (list): list of tuples of corruption function and 
        aug_names (_type_): _description_
        corrupt_func (_type_): corruption (helper) method e.g. corrupt_func_album
        filename (str, optional): string of filename to save the matrix of augmented images. Defaults to None.
        graph_lib (str, optional): graphing library used. Defaults to 'matplotlib'.

    Raises:
        ValueError: invalid graphing library provided. Must be either matplotlib or plotly.

    Returns:
        fig (plotly.Figure): (matrix) figure of augmented image.
    """
    severities = [0, 1, 2]
    rows = len(augmentations)
    cols = len(severities)
    
    # Create a subplot grid
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f"Severity {s}" for s in severities] * (1 if rows == 1 else 0),
                        vertical_spacing=0.02, horizontal_spacing=0.02)

    for i, (aug_class, param_dict) in enumerate(augmentations):
        param_dict['aug_method'] = aug_class
        for j, severity in enumerate(severities):
            # Apply corruption
            corrupted_img = img if severity == 0 else corrupt_func([img], severity, param_dict)[0]
            # Convert to uint8 if needed
            if corrupted_img.dtype != np.uint8:
                corrupted_img = np.clip(corrupted_img, 0, 255).astype(np.uint8)

            # Add image to subplot
            fig.add_trace(
                go.Image(z=corrupted_img),
                row=i + 1,
                col=j + 1
            )

            # Custom Y-axis label (as annotation)
            if j == 0:
                fig.add_annotation(
                    text=aug_names[i],
                    xref="paper", yref="paper",
                    x=0, y=1 - (i / rows),
                    showarrow=False,
                    font=dict(size=12),
                    xanchor="right",
                    yanchor="middle"
                )

    # Set layout
    fig.update_layout(
        height=150 * rows,
        width=200 * cols,
        title_text="Corrupted Image Grid (by Severity & Augmentation)",
        showlegend=False
    )

    # Hide axes
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=i, col=j)
            fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=i, col=j)

    # Save to file if filename is provided
    if filename:
        fig.write_image(filename)

    fig.show()
    return fig

def corrupt_and_plot_generic_mpl(img, augmentations, aug_names, corrupt_func, filename=None):
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

    return fig

def corrupt_and_plot_generic(img, augmentations, aug_names, corrupt_func, filename=None, graph_lib='matplotlib'):
    """Header method for corrupting (read: one!) an image and plotting it

    Args:
        img (np.array): np.array representing image
        augmentations (list): list of tuples of corruption function and param dictionaries for each function.
        aug_names (_type_): _description_
        corrupt_func (_type_): corruption (helper) method e.g. corrupt_func_album
        filename (str, optional): string of filename to save the matrix of augmented images. Defaults to None.
        graph_lib (str, optional): graphing library used. Defaults to 'matplotlib'.

    Raises:
        ValueError: invalid graphing library provided. Must be either matplotlib or plotly.

    Returns:
        fig: (matrix) figure of augmented image.
    """
    if graph_lib == 'matplotlib':
        return corrupt_and_plot_generic_mpl(img, augmentations, aug_names, corrupt_func, filename)
    elif graph_lib == 'plotly':
        return corrupt_and_plot_generic_plotly(img, augmentations, aug_names, corrupt_func, filename)
    else:
        raise ValueError('not valid graphing library')

def get_album_augmentations_list():
    """
    Get the augmentation list and string for albumentations.

    Returns:
        augmentation_list (list): list of two-length tuples 
        augmentation_str (list): list of names of the given libraries
    """
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
    """
    Get the augmentation list and string for nrtk.

    Args:
        seed (int): random seed (fixed for now)

    Returns:
        augmentation_list (list): list of two-length tuples 
        augmentation_str (list): list of names of the given libraries
    """
    perturbations = [
        # (SaltNoisePerturber, {'rng': seed, 'amount': lambda s: 0.15 * s}),
        # (PepperNoisePerturber, {'rng': np.random.default_rng(seed), 'amount': lambda s: 0.15 * s}),
        (SaltAndPepperNoisePerturber, {'rng': np.random.default_rng(seed), 'amount': lambda s: 0.15 * s}),
        (GaussianNoisePerturber, {'rng': seed, 'mean': lambda s: 0.1 * s, 'var': lambda s: 0.01 * s}),
        # (SpeckleNoisePerturber, {'rng': seed, 'mean': lambda s: 0.3 * s, 'var': lambda s: 0.01 * s}),
        # (AverageBlurPerturber, {'ksize': lambda s: 11 + 2 * s}),
        (GaussianBlurPerturber, {'ksize': lambda s: 11 + 4 * s}),
        # (MedianBlurPerturber, {'ksize': lambda s: 11 + 2 * s}),
        (BrightnessPerturber, {'factor': lambda s: 1 + 0.15 * s}),
        # (ColorPerturber, {'factor': lambda s: 1 - 0.15 * s}),
        (ContrastPerturber, {'factor': lambda s: 1 + 0.2 * s}),
        (SharpnessPerturber, {'factor': lambda s: 1 - 0.2 * s}),
    ]
    perturb_names = ["Salt Pepper Noise", "Gaussian Noise", "Gaussian Blur", "Brightness", "Contrast", "Sharpness"]
    # perturb_names = ["Salt Noise", "Pepper Noise", "Salt Pepper Noise", "Gaussian Noise", "Speckle Noise", \
    #                  "Average Blur", "Gaussian Blur", "Median Blur", "Brightness", "Color", "Contrast", "Sharpness"]
    return perturbations, perturb_names

def get_augly_augmentations_list():
    """
    Get the augmentation list and string for augly.

    Returns:
        augmentation_list (list): list of two-length tuples 
        augmentation_str (list): list of names of the given libraries
    """
    augmentations = [
        (blur, {'radius': lambda s: s}),  # Corrected: Replaced 'severity' with lambda
        (brightness, {'factor': lambda s: 1 + 0.1 * s}),
        (contrast, {'factor': lambda s: s}),  # Corrected
        (random_noise, {'mean': lambda s: 0.0001 * s, 'var': lambda s: 0.00001})
    ]
    aug_str = ['blur', 'brightness', 'contrast', 'random_noise']
    return augmentations, aug_str

def get_imagecorrupt_augmentations_list():
    """
    Get the augmentation list and string for imagecorrupt.

    Returns:
        augmentation_list (list): list of two-length tuples 
        augmentation_str (list): list of names of the given libraries
    """
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
    """
    Returns the corruption parameters associated with a given library

    Two-length tuples: first being corruption function (pure from library), second being parameters to be passed in

    Args:
        library (str): corruption library name
    
    Returns:
        augmentation_list (list): list of two-length tuples 
        augmentation_str (list): list of names of the given libraries
        corrupt_func (function): custom corrupt function: takes in images np array, severity, and augmentation parameters and returns corrupted images np array
    """
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
        return u1, u2, corrupt_func_augly
    else:
        raise Exception('Invalid library called')

def get_corrupted_dataloader(testloader, corr_func, severity=1, corr_kwargs=None):
    """
    Make a dataloader with data that is of the augmented/corrupted version of the original dataloader

    Args:
        testloader (torch.Dataloader): original dataloader
        corr_func (function): corruption function  that takes in ([numpy array of images], severity, corr_kwargs)
        severity (int); extent of severity between 0-5.
        corr_kwargs (dict): dictionary of extra parameters to send into corr_func

    Returns:
        corrupted_dataloader (torch.Dataloader): corrupted version of original dataloader
    """
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

def augmentation_gradient(model, test_loader, device, corr_func, plot_graphs=False, corr_kwargs=None):
    """
    Evaluates how the model performance varies against the given augmentation/corruption

    Augments across severity 1-5 (and 0) and outputs the performance change

    Args:
        model: model
        test_loader (torch.dataloader): test data loader
        device (torch.device): device model is on
        corr_func (function): corruption function to take in images (np array) / give corrupted dataloader
        plot_graphs: either False for no graph, or string for which graphing library to use
        corr_kwargs (dict): corruption arguments

    Returns:
        best_fit_gradient (float): best fit line gradient of graph of performance vs severity (of augmentation)
        accuracies (list): list of floats of performance metric 
        fig (figure): outputs figure of plot_graphs library if not plot_graphs not False, else None
    """
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
    fig = None
    if plot_graphs is not False:
        fig = plot_accuracy_vs_severity(accuracies, severities, plot_graphs)  
    return best_fit_gradient(severities, accuracies), accuracies, fig

def augmentation_perf_gradient_method(
    model, 
    test_loader, 
    device, 
    augmentation_func_wrapper, 
    augmentation_list,
    augmentation_str,
    plot_graphs=False
    ):
    """Big header method for determining how model behaves with different augmentation severities

    Iterate through the augmentation iterables for each augmentation technique (e.g. blur, contrast, etc for imagecorr)

    Args:
        model (torch.nn): model (torch model for now)
        test_loader (torch.utils.data.DataLoader): data loader
        device (torch.device): cuda/cpu device
        augmentation_func_wrapper (function): corruption function (e.g. corrupt_func_album)
        augmentation_list (list): list of tuples for augmentation
        augmentation_str (str): list of augmentation techniques for given augmentation method
        plot_graphs (bool, optional): False or the graphing library to plot graph. Defaults to False.

    Returns:
        Tuple: 
        - augmentation_dict (dict): dictionary of augmentations <-> 
    """

    assert len(augmentation_list) == len(augmentation_str)

    model = model.to(device)
    augmentation_dict = {}
    augmentation_fig_dict = {}
    for k, (m,d) in zip(augmentation_str, augmentation_list):
        d['aug_method'] = m
        gradient, accuracies, fig = augmentation_gradient(model, test_loader, device, augmentation_func_wrapper, plot_graphs, d)
        first_drop = accuracies[1] - accuracies[0]
        
        print(accuracies, first_drop)
        print(k, 'augmentation method gradient:', gradient)
        augmentation_dict[k] = (gradient, first_drop)
        augmentation_fig_dict[k] = fig
        
        print()
    return augmentation_dict, augmentation_fig_dict 


# =============================================================================================

if __name__ == "__main__":
    from cvrob_util import SimpleCNN
    from torchvision import datasets
    import torchvision.transforms as transforms

    device = torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    # ===============================================================================================================

    print("noisy indices done! let's go to: robustness evaluation")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load('label_noise_simplecnn.h5', weights_only=True))

    clean_test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    clean_test_loader = torch.utils.data.DataLoader(clean_test_dataset, batch_size=128, shuffle=False)

    augmentation_list, augmentation_str, corrupt_func = get_corruption_helpers('nrtk')
    aug_dict_g, aug_dict_f = augmentation_perf_gradient_method(
        model, 
        clean_test_loader, 
        device, 
        corrupt_func, 
        augmentation_list,
        augmentation_str,
        plot_graphs=False
    )
    
    print(aug_dict_g)

def augly_combination_gradients(model, test_loader, device, plot_graphs=False):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    augmentation_methods = [blur, brightness, contrast, random_noise]
    augmentation_dict = {}
    augmentation_methods_str=  ['blur', 'brightness', 'contrast', 'random_noise']
    for m, mstr in zip(augmentation_methods, augmentation_methods_str):
        corr_kwargs = {"augly_func": m}
        gradient, accuracies = augmentation_gradient(model, test_loader, device, augly_wrapper, plot_graphs, corr_kwargs)
        first_drop = accuracies[1] - accuracies[0]
        
        print(accuracies, first_drop)
        print(mstr, 'augmentation method gradient:', gradient)
        augmentation_dict[mstr] = (gradient, first_drop)
        
        print()
    return augmentation_dict

def imagecorr_combination_gradients(model, test_loader, device, plot_graphs=False):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    augmentation_dict = {}
    augmentation_methods = ['gaussian_noise', 'fog', 'brightness', 'zoom_blur']
    augmentation_methods_str=  ['gaussian_noise', 'fog', 'brightness', 'zoom_blur']
    for m, mstr in zip(augmentation_methods, augmentation_methods_str):
        corr_kwargs = {"corrname": m}
        gradient, accuracies = augmentation_gradient(model, test_loader, device, imagecorrupt_func, plot_graphs, corr_kwargs)
        first_drop = accuracies[1] - accuracies[0]
        augmentation_dict[mstr] = (gradient, first_drop)
        print(accuracies, first_drop)
        print(mstr, 'augmentation method gradient:', gradient)
        print()
    return augmentation_dict

def augly_func(images_np, augly_func, aug_params):
    return np.array([aug_np_wrapper(img, augly_func, **aug_params) for img in images_np])

def augly_wrapper(images_np, severity, augmentations):
    augmentations = [
        (blur, {'radius': lambda s: s}),  # Corrected: Replaced 'severity' with lambda
        (brightness, {'factor': lambda s: 1 + 0.1 * s}),
        (contrast, {'factor': lambda s: s}),  # Corrected
        (random_noise, {'mean': lambda s: 0.0001 * s, 'var': lambda s: 0.00001})
    ]
    
    aug_kwargs = None
    for aug_func, param_dict in augmentations:
        if corr_kwargs['augly_func'] == aug_func:
            aug_kwargs = {k: (v(severity) if callable(v) else v) for k, v in param_dict.items()}
            break
            
    if aug_kwargs is None:
        raise ValueError("Augmentation function not found in list.")
    
    return augly_func(images_np, corr_kwargs['augly_func'], aug_kwargs)

def imagecorrupt_func(images_np, severity, aug_kwargs):
    return np.array([corrupt(img.astype(np.uint8), 
                             corruption_name=aug_kwargs['corrname'], 
                             severity=severity) for img in images_np])

    # aug_dict = augly_combination_gradients(model, clean_test_loader, device, plot_graphs=False)
    # print(aug_dict)
    # aug_dict2 = imagecorr_combination_gradients(model, clean_test_loader, device, plot_graphs=False)
    # print(aug_dict2)

    # augmentation_method_dict = {'blur': [blur], 'brightness': [brightness], 'contrast': [contrast], 'random_noise': [random_noise]}
    # args_header = ["augly_func"]
    # aug_dict_g = generic_combination_gradients(model, clean_test_loader, device, augmentation_method_dict, args_header, plot_graphs=False)

def generic_combination_gradients(
    model, 
    test_loader, 
    device, 
    # augmentation_method_dict, 
    # args_header, 
    augmentation_func_wrapper,
    augmentation_list,
    augmentation_str,
    plot_graphs=False
    ):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    augmentation_dict = {}
    for k, (m,d) in zip(augmentation_str, augmentation_list):
        d['aug_method'] = m
        gradient, accuracies = augmentation_gradient(model, test_loader, device, augmentation_func_wrapper, plot_graphs, d)
    # for k,v in augmentation_method_dict.items(): 
    #     corr_kwargs = {}#"augly_func": m}
    #     for i,arg in enumerate(args_header):
    #         corr_kwargs[arg] = v[i]
    #     gradient, accuracies = augmentation_gradient(model, test_loader, device, augly_wrapper, plot_graphs, corr_kwargs)
        first_drop = accuracies[1] - accuracies[0]
        
        print(accuracies, first_drop)
        print(k, 'augmentation method gradient:', gradient)
        augmentation_dict[k] = (gradient, first_drop)
        
        print()
    return augmentation_dict

#     # Step 3: Compute agreement scores
#     noisy_indices = []
#     agreement_scores = []
#     for i, neighbors in enumerate(knn_indices):
#         neighbor_labels = true_labels[neighbors]  # Labels of k-nearest neighbors
#         agreement_score = np.sum(neighbor_labels == true_labels[i]) / len(neighbor_labels)
#         agreement_scores.append(agreement_score)

#         # If agreement is below threshold, mark as noisy
#         if agreement_score < threshold:
#             noisy_indices.append(indices[i])

# #     print(f"Detected {len(noisy_indices)} noisy labels.")
#     return noisy_indices
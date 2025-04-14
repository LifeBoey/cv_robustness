import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print('current dir', os.getcwd())
import ast 
from collections import OrderedDict
import pandas as pd
import argparse
import pickle
from tensorflow.keras.models import load_model
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import yaml
from cvrob import (label_noise_method,
                    ensemble_method,
                    label_update_indice_method, 
                    wrong_prediction_indice_method,
                    loss_indice_method,
                    aum_indice_method,
                    fine_indice_method,
                    cleanlab_indice_method,
                    deep_knn_indice_method,
                    generic_combination_gradients,
                    get_corruption_helpers)
import json

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="drift_main_initialize Argparser")

    parser.add_argument("--data_path",
                        type=str,
                        help="Paths of the data you are using as reference",
                        default='dataset_ships.pt') #'uber_train.csv'
    parser.add_argument("--test_file_path",
                        type=str,
                        help="Paths of the data you are using as reference",
                        default='fashion_test') #'adult_test2.csv'
    parser.add_argument("--model_file_path",
                        type=str,
                        help="Path of the category map to show what categories map to what",
                        default="fashion_mnist.h5") 
    parser.add_argument("params_file_path",
                        type=str,
                        help="Path of the category map to show what categories map to what",
                        default="fashion_mnist.h5") 
    


        
    parser.add_argument("--dataset_id", 
                        default="")
    parser.add_argument("--dataset_name", 
                        default="image_dataset") #
    parser.add_argument("--remote_data_url", 
                        default="s3://ecs.dsta.ai:80/public-data/digitalhub/data_drift/data/images")
    parser.add_argument("--output_uri", 
                        type=str,
                        help="output URI for artifacts, datasets",
                        default="s3://ecs.dsta.ai:80/public-data/digitalhub/data_drift/outputs",
                        required=False)
    parser.add_argument("--file_name_appendum", 
                        type=str,
                        help="output URI for artifacts, datasets",
                        default="_fashion_embed",
                        required=False)
    parser.add_argument("--custom_inference", 
                        type=str,
                        help="output URI for artifacts, datasets",
                        default="image_inference",
                        required=False)
    parser.add_argument("--model_api", 
                        type=str,
                        help="output URI for artifacts, datasets",
                        default="",
                        required=False)
    parser.add_argument("--feature_extractor_api", 
                        type=int,
                        help="output URI for artifacts, datasets",
                        default=1,
                        required=False)
          
    args = parser.parse_args()

    # Init Clearml
    CLEARML_INITIALIZED = False
    try:
        
        from clearml import Task, StorageManager, Dataset
        
        task = Task.init(
            project_name="Drift_Metric_Expt",
            task_name="aip_pipeline_expt_set",
            output_uri=args.output_uri,
        )
        task.connect(args)

        # Set base Docker image
        task.set_base_docker("harbor.dsta.ai/public/drift_expt_base:1b")
        # task.set_repo('http://gitlab.dev.pc8.dsta/BJIEYONG/drift-metric-expt/tree/main/aip_scripts')
        # task.set_script(repository='http://gitlab.dev.pc8.dsta/BJIEYONG/drift-metric-expt.git',
        #                 branch='main',
        #                 working_dir='aip_scripts',
        #                 )
        #TODO????
        
        # Remote Execution
        # -------------------
        # only create task, execute later
        #task.execute_remotely(queue_name="queue-1xV100-16ram") 
        # Putting this on the GPU is technically faster, but it eats up all the GPUs on clearml which screws everyone else over so.
        task.execute_remotely(queue_name="queue-cpu-only-large")

        CLEARML_INITIALIZED = True

    except Exception as e:
        print("Clearml Exception: {0:s}\n".format(str(e)))

    device = torch.device('cpu') 
    #import data

    # if CLEARML_INITIALIZED:
    #     #TODO Storage Manager case
    #     data_path = Dataset.get(dataset_id=args.data_file_path).get_local_copy()
    #     catmap_path = Dataset.get(dataset_id=args.catmap_file_path).get_local_copy()
    #     data_df = pd.read_csv(data_path)      
    #     with open(catmap_path, 'rb') as handle:
    #         category_map = pickle.load(handle)        
    if CLEARML_INITIALIZED:
        # Get a local copy of the data
        # Technicallyyyy all of these work but I couldn't find the way to get dataset_id to work, despite it being the "good" way
        # So I used dataset_name, and it works fine, so.
        if args.dataset_id:
            # (A) download from a specific dataset_id (and the parents before it)
            dataset_path = Dataset.get(dataset_id=args.dataset_id).get_local_copy()
            print(
                f"Downloading local copy of data from dataset id {args.dataset_id} to ~/.clearml/cache/storage_manager/datasets/"
            )
        elif args.dataset_name:
            # (B) download from a specific dataset_name (latest child and parents before it)
            dataset_path = Dataset.get(dataset_name=args.dataset_name).get_local_copy()
            print(
                f"Downloading local copy of data from dataset name {args.dataset_name} to ~/.clearml/cache/storage_manager/datasets/"
            )
        else:
            # (C) download from data URL via StorageManager
            manager = StorageManager()
            dataset_path = manager.download_folder(remote_url=args.remote_data_url)
            print(
                f"Downloading local copy of data from remote path: {args.remote_data_url}"
            )

        DATA_DIR = dataset_path      
        print("Data directory from clearml:", DATA_DIR)
        print()

    else:
        DATA_DIR = os.path.join('..', 'data')
        print("Data directory from local machine:", DATA_DIR)
        print()


    model = torch.load(os.path.join(DATA_DIR, args.model_file_path), weights_only=False)
    test_dataset = torch.load(os.path.join(DATA_DIR, args.test_file_path), weights_only=False)
    with open('config.yaml', 'r') as file:
        param_dict = yaml.safe_load(file)

    device = torch.device("cpu")
    model.to(device)

    list_of_methods = [ label_update_indice_method, 
                       wrong_prediction_indice_method,
                        loss_indice_method,
                        aum_indice_method,
                        fine_indice_method,
                        cleanlab_indice_method,
                        deep_knn_indice_method]
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
    
    # Create/define output directory
    OUTPUT_DIR = ".."
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, "outputs")
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)    
            
    expt_set_path = os.path.join(OUTPUT_DIR, 'expt_set'+args.file_name_appendum+'.pickle') 
    model_set_path = os.path.join(OUTPUT_DIR, 'model_set'+args.file_name_appendum+'.pickle') 
    drift_dict_path = os.path.join(OUTPUT_DIR, 'drift_dict'+args.file_name_appendum+'.pickle')
    driftgen_dict_path = os.path.join(OUTPUT_DIR, 'driftgen_dict'+args.file_name_appendum+'.pickle')
    # feat_ext_set_path = os.path.join(OUTPUT_DIR, 'feat_ext_set'+args.file_name_appendum+'.pickle')

    with open(expt_set_path, 'wb') as handle:
        pickle.dump(expt_set, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    with open(model_set_path, 'wb') as handle:
        pickle.dump(model_set, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    with open(drift_dict_path, 'wb') as handle:
        pickle.dump(drift_dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    with open(driftgen_dict_path, 'wb') as handle:
        pickle.dump(driftgen_dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    # with open(feat_ext_set_path, 'wb') as handle:
    #     pickle.dump(feat_ext_set, handle, protocol=pickle.HIGHEST_PROTOCOL) 

    if CLEARML_INITIALIZED:
        task.upload_artifact("expt_set", artifact_object=expt_set_path)
        task.upload_artifact("model_set", artifact_object=model_set_path)
        task.upload_artifact("drift_dict", artifact_object=drift_dict_path)
        task.upload_artifact("driftgen_dict", artifact_object=driftgen_dict_path)
        #task.upload_artifact("feat_ext_set", artifact_object=feat_ext_set_path)
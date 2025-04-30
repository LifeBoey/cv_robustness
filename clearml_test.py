import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# print('current dir', os.getcwd())
import argparse
import pickle
import torch
import io
import yaml
from cvrob import (label_noise_method,
                    ensemble_method,
                    get_all_label_noise_methods,
                    generic_combination_gradients,
                    get_corruption_helpers,
                    get_image_from_path,
                    corrupt_and_plot_generic,
                    retrain_experiment,
                    get_partially_corrupted_imagecorr_dataloader)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="drift_main_initialize Argparser")


    parser.add_argument("--test_file_path",
                        type=str,
                        help="Paths of the data you are using as reference",
                        default='dataset_ships.pt') #'adult_test2.csv'
    parser.add_argument("--train_file_path",
                        type=str,
                        help="Paths of the data you are using as reference",
                        default='dataset_ships.pt') #'adult_test2.csv'
    parser.add_argument("--model_file_path",
                        type=str,
                        help="Path of the category map to show what categories map to what",
                        default="fashion_mnist.h5") 
    parser.add_argument("params_file_path",
                        type=str,
                        help="Path of the category map to show what categories map to what",
                        default="config.yaml") 
    parser.add_argument("augmentatinon_libraries",
                        type=str,
                        help="Path of the category map to show what categories map to what",
                        default="album,nrtk") 
    parser.add_argument("image_file_path",
                        type=str,
                        help="Path of the category map to show what categories map to what",
                        default="hello.jpg") 
    
    parser.add_argument("endpoint_url",
                        type=str,
                        help="Path of the category map to show what categories map to what",
                        default="hello.jpg") 
    parser.add_argument("access_key",
                        type=str,
                        help="Path of the category map to show what categories map to what",
                        default="hello.jpg") 
    parser.add_argument("secret_key",
                        type=str,
                        help="Path of the category map to show what categories map to what",
                        default="hello.jpg") 
    parser.add_argument("bucket_name",
                        type=str,
                        help="Path of the category map to show what categories map to what",
                        default="hello.jpg") 

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
                        default="_cvrobtest",
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
        task.set_base_docker("harbor.dsta.ai/public/cvrob_image:latest")
        # task.set_repo('http://gitlab.dev.pc8.dsta/BJIEYONG/drift-metric-expt/tree/main/aip_scripts')
        # task.set_script(repository='http://gitlab.dev.pc8.dsta/BJIEYONG/drift-metric-expt.git',
        #                 branch='main',
        #                 working_dir='aip_scripts',
        #                 )
        #TODO????
        
        # Remote Execution
        # -------------------
        # only create task, execute later
        # Putting this on the GPU is technically faster, but it eats up all the GPUs on clearml which screws everyone else over so.
        task.execute_remotely(queue_name="queue-cpu-only-large")

        CLEARML_INITIALIZED = True

    except Exception as e:
        print("Clearml Exception: {0:s}\n".format(str(e)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #import data      
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
        try:
            import boto3; from botocore.client import Config
            s3 = boto3.client(
                's3',
                endpoint_url = args.endpoint_url,
                aws_access_key_id = args.access_key,
                aws_secret_access_key = args.secret_key,
                config=Config(signature_version='s3v4'),
                region_name = 'us-east-1',
                verify = False
            )
            DATA_DIR = 's3'

        except:
            DATA_DIR = os.path.join('')
            print("Data directory from local machine:", DATA_DIR)
            print()

    if DATA_DIR == 's3':
        response = s3.get_object(Bucket=args.bucket_name, Key='bjieyong/'+args.model_file_path)
        pt_bytes = response['Body'].read(); buffer = io.BytesIO(pt_bytes)
        model = torch.load(buffer, weights_only=False)
        response = s3.get_object(Bucket=args.bucket_name, Key='bjieyong/'+args.test_file_path)
        pt_bytes = response['Body'].read(); buffer = io.BytesIO(pt_bytes)
        train_dataset = torch.load(buffer, weights_only=False)
        response = s3.get_object(Bucket=args.bucket_name, Key='bjieyong/'+args.test_file_path)
        pt_bytes = response['Body'].read(); buffer = io.BytesIO(pt_bytes)
        test_dataset = torch.load(buffer, weights_only=False)
        response = s3.get_object(Bucket=args.bucket_name, Key='bjieyong/'+args.test_file_path)
        pt_bytes = response['Body'].read(); buffer = io.BytesIO(pt_bytes)
        clean_test_dataset = torch.load(buffer, weights_only=False)
        response = s3.get_object(Bucket=args.bucket_name, Key='bjieyong/'+args.params_file_path)
        yaml_bytes = response['Body'].read(); yaml_text = yaml_bytes.decode('utf-8')
        param_dict = yaml.safe_load(yaml_text) 
    else:
        model = torch.load(os.path.join(DATA_DIR, args.model_file_path), weights_only=False)
        test_dataset = torch.load(os.path.join(DATA_DIR, args.test_file_path), weights_only=False)
        with open(os.path.join(DATA_DIR, args.params_file_path), 'r') as file:
            param_dict = yaml.safe_load(file) 
        train_dataset = torch.load(os.path.join(DATA_DIR, args.train_file_path), weights_only=False)
        clean_test_dataset = torch.load(os.path.join(DATA_DIR, args.test_file_path), weights_only=False)

    clean_test_loader = torch.utils.data.DataLoader(clean_test_dataset, batch_size=param_dict['batch_size'], shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=param_dict['batch_size'], shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    list_of_methods = get_all_label_noise_methods()
    noisy_indices_all, true_noisy_indices, test_loader = label_noise_method(test_dataset, 
                                                                            model, 
                                                                            device,
                                                                            list_of_methods, 
                                                                            param_dict,
                                                                            evaluate=False, 
                                                                            noise_ratio=0, 
                                                                            batch_size=param_dict['batch_size'])

    final_noisy_indices = ensemble_method(noisy_indices_all)
    print('final noisy indices, ', final_noisy_indices)

    augmentation_libraries = args.augmentation_libraries.split(",")
    aug_dict = {}; aug_fig_dict = {}
    for lib in augmentation_libraries:
        augmentation_list, augmentation_str, corrupt_func = get_corruption_helpers(lib)
        aug_dict_g, aug_dict_f = generic_combination_gradients(
            model, 
            test_loader, 
            device, 
            corrupt_func, 
            augmentation_list, 
            augmentation_str, 
            plot_graphs=False
        )
        aug_dict[lib] = aug_dict_g
        aug_fig_dict[lib] = aug_dict_f
    

    model_copy, model_no_weights, test_acc_dict = retrain_experiment(model, 
                                                                     train_loader, 
                                                                     clean_test_loader, 
                                                                     device,
                                                                     get_partially_corrupted_imagecorr_dataloader)
    
    img_array = get_image_from_path(args.image_file_path)
    fig_dict = {}
    for lib in augmentation_libraries:
        augmentation_list, augmentation_str, corrupt_func = get_corruption_helpers(lib)
        fig = corrupt_and_plot_generic(img_array, 
                                       augmentation_list, 
                                       augmentation_str, 
                                       corrupt_func, 
                                       filename=None,
                                       graph_lib='plotly')
        fig_dict[lib] = fig

    if CLEARML_INITIALIZED:
        for lib, fig in fig_dict.items():
            task.get_logger().report_plotly(
                title="Library "+lib, 
                series="Plotly Figure for augmented ship picture",
                iteration=0,
                figure=fig
            )

        for lib, d in aug_fig_dict.items():
            for aug_str, fig in d.items():
                task.get_logger().report_plotly(
                    title="Library "+lib+" & augmentation method "+aug_str, 
                    series="Plotly Graph for augmentation method accuracy vs severity",
                    iteration=0,
                    figure=fig
                )
 
    # Create/define output directory
    OUTPUT_DIR = ""
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, "outputs")
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)    
            
    final_noisy_indices_path = os.path.join(OUTPUT_DIR, 'final_noisy_indices'+args.file_name_appendum+'.pickle') 
    augmentation_result_dict_path = os.path.join(OUTPUT_DIR, 'augmentation_result_dict'+args.file_name_appendum+'.pickle') 
    test_acc_dict_path = os.path.join(OUTPUT_DIR, 'test_acc_dict'+args.file_name_appendum+'.pickle') 
    fig_dict_path = os.path.join(OUTPUT_DIR, 'fig_dict'+args.file_name_appendum+'.pickle') 

    with open(final_noisy_indices_path, 'wb') as handle:
        pickle.dump(final_noisy_indices, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    with open(augmentation_result_dict_path, 'wb') as handle:
        pickle.dump(aug_dict, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    with open(test_acc_dict_path, 'wb') as handle:
        pickle.dump(test_acc_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(fig_dict_path, 'wb') as handle:
        pickle.dump(fig_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if CLEARML_INITIALIZED:
        task.upload_artifact("final_noisy_indices", artifact_object=final_noisy_indices_path)
        task.upload_artifact("aug_dict", artifact_object=augmentation_result_dict_path)
        task.upload_artifact("test_acc_dict", artifact_object=test_acc_dict_path)
        task.upload_artifact("fig_dict", artifact_object=fig_dict_path)
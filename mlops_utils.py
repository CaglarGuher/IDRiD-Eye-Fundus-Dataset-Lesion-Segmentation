import datetime
import json
import os

import cv2
import numpy as np

from os.path import join


def make_subfolder(dirname,parent_path):
    path = os.path.join(parent_path, dirname)
    os.mkdir(path)
    print("Directory '%s' created" %dirname)
    return path + '/'

def make_log_dir(parent_path = "out"):
    os.makedirs(parent_path, exist_ok =True)
    current_date = datetime.datetime.now()
    dirname = current_date.strftime("%Y_%B_%d-%H_%M_%S")
    path = make_subfolder(dirname,parent_path)
    return path

def write_to_log(log_dir ="", log_entry = ""):
    with open(os.path.join(log_dir, "log.txt", "a")) as file:
        file.write(log_entry)

def save_configs(task_config,log_dir):
    '''Save config files in to log_dir'''
    dataset_conf = task_config['dataset_conf']
    model_conf = task_config['model_conf']
    training_conf = task_config['training_conf']
    augment_conf = task_config['augment_conf']
    # save them as json file with indentations
    json_file_path = os.path.join(log_dir, 'dataset_conf.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(dataset_conf, json_file, indent=4)
    json_file_path = os.path.join(log_dir, 'model_conf.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(model_conf, json_file, indent=4)
    json_file_path = os.path.join(log_dir, 'training_conf.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(training_conf, json_file, indent=4)
    json_file_path = os.path.join(log_dir, 'augment_conf.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(augment_conf, json_file, indent=4)

def check_masks_dir_format(mask_dir):
    '''Check if lesion folders are in mask_dir'''
    lesion_folders = ['ma', 'he', 'ex', 'se']
    for lesion in lesion_folders:
        if not os.path.isdir(os.path.join(mask_dir, lesion)):
            print("Lesion folder not found: ", lesion)
            return False
    return True

def check_and_fix_masks_dir(mask_dir):
    '''Check if masks are binary and fix them if not'''
    if not check_masks_dir_format(mask_dir):
        # if lesion folders are not found, create them
        lesion_folders = ['ma', 'he', 'ex', 'se']
        for lesion in lesion_folders:
            os.mkdir(os.path.join(mask_dir, lesion))
        # list png files in mask_dir
        img_names = [img_name for img_name in os.listdir(mask_dir) if img_name.endswith('.png')]
        # convert masks to binary and save them
        for mask_name in img_names:
            mask_path = os.path.join(mask_dir,mask_name)
            mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
            ma_mask = np.where(mask == 3,255,0)
            he_mask = np.where(mask == 2,255,0)
            se_mask = np.where(mask == 4,255,0)
            ex_mask = np.where(mask == 1,255,0)
            # save masks in corresponding folders
            cv2.imwrite(os.path.join(mask_dir, 'ma', mask_name), ma_mask)
            cv2.imwrite(os.path.join(mask_dir, 'he', mask_name), he_mask)
            cv2.imwrite(os.path.join(mask_dir, 'se', mask_name), se_mask)
            cv2.imwrite(os.path.join(mask_dir, 'ex', mask_name), ex_mask)
    else:
        print(f"Masks are in correct format in {mask_dir}. Skipping mask preparation step...")

def wandb_epoch_log(train_logs, valid_logs):
    logs = {}
    for key, value in train_logs.items():
        logs["train_"+key] = value
    for key, value in valid_logs.items():
        logs["valid_"+key] = value  
    return logs
    
def wandb_final_log(auc_pr_result, metrics_merged, metrics_cropped):
    logs = {}
    logs["auc_pr"] = auc_pr_result
    for key, value in metrics_merged.items():
        logs["merged_"+key] = value
    for key, value in metrics_cropped.items():
        logs["cropped_"+key] = value  
    return logs

def derive_dataset_conf_parameters(dataset_conf):
    datasets_root = dataset_conf['dataset_root']
    if dataset_conf['denoised']:
        dir_annex = join('Denoised', f'all_{dataset_conf["denoising_size"]}')
    elif dataset_conf['preprocessed']:
        dir_annex = 'Preprocessed'
    else:
        dir_annex = 'Original'
    dataset_conf['train_image_dir']   = join(join(datasets_root, dir_annex), 'train')
    dataset_conf['train_mask_dir']    = join(join(datasets_root, 'labels'), 'train')
    dataset_conf['val_image_dir']     = join(join(datasets_root, dir_annex), 'val')
    dataset_conf['val_mask_dir']      = join(join(datasets_root, 'labels'), 'val')
    dataset_conf['test_image_dir']    = join(join(datasets_root, dir_annex), 'test')
    dataset_conf['test_mask_dir']     = join(join(datasets_root, 'labels'), 'test')

    if dataset_conf['cropped']:
        crop_name = f"_crop{dataset_conf['crop_size']}_s{dataset_conf['stride']}"
        dataset_conf['train_image_dir_cropped']   = dataset_conf['train_image_dir'] + crop_name
        dataset_conf['train_mask_dir_cropped']    = dataset_conf['train_mask_dir'] + crop_name
        dataset_conf['val_image_dir_cropped']     = dataset_conf['val_image_dir'] + crop_name
        dataset_conf['val_mask_dir_cropped']      = dataset_conf['val_mask_dir'] + crop_name
        dataset_conf['test_image_dir_cropped']    = dataset_conf['test_image_dir'] + crop_name
        dataset_conf['test_mask_dir_cropped']     = dataset_conf['test_mask_dir'] + crop_name
    return dataset_conf
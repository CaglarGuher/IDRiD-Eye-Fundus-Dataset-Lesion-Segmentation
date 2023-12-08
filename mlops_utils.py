import os
import json
import datetime
import cv2
import numpy as np
import yagmail

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

def send_results_via_mail(log_dir):
    results = os.path.join(log_dir, 'results.json')
    model_conf = os.path.join(log_dir, 'model_conf.json')
    training_conf = os.path.join(log_dir, 'training_conf.json')
    augment_conf = os.path.join(log_dir, 'augment_conf.json')
    dataset_conf = os.path.join(log_dir, 'dataset_conf.json')
    contents = [ "Train sonuçları ve konfigürasyonu ekte yer almaktadır",
    results, model_conf, training_conf, augment_conf, dataset_conf
    ]
    with yagmail.SMTP('viventedevelopment', 'yeniparrola2.1') as yag:
        yag.send('ademgunesen+viventedev@gmail.com', 'Train Sonuçları' + log_dir, contents)

    
    

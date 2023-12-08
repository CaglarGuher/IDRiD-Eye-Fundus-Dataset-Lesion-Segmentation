from utils import *
from train_test_val_initialize import *
from mlops_utils import make_log_dir,write_to_log,save_configs, check_and_fix_masks_dir

device = "cuda:2"

dataset_conf = {}
model_conf = {}
training_conf = {}
augment_conf = {}

datasets_root = "/home/braincreator/daniel/ademgunesen/IDRiD Dataset Collection/Adamlarin Format/"

dataset_conf['preprocessed']      = False
dataset_conf['denoised']          = True
dataset_conf['cropped']           = True
dataset_conf['crop_size']         = 256
dataset_conf['stride']            = 256
dataset_conf['black_ratio_train'] = 1 # TODO: Implement this
dataset_conf['denoising_size']    = 4096
dataset_conf['resolution']        = 0
dataset_conf['data']              = "ma"
############################################################################################################
# Derived parameters : Do not change these
if dataset_conf['denoised']:
    dataset_conf['train_image_dir']   = datasets_root + 'Denoised/' + f'all_{dataset_conf["denoising_size"]}/' + 'train'
    dataset_conf['train_mask_dir']    = datasets_root + "labels/" + "train"
    dataset_conf['val_image_dir']     = datasets_root + 'Denoised/' + f'all_{dataset_conf["denoising_size"]}/' + 'test'
    dataset_conf['val_mask_dir']      = datasets_root + "labels/" + "test"
    dataset_conf['test_image_dir']    = datasets_root + 'Denoised/' + f'all_{dataset_conf["denoising_size"]}/' + 'test'
    dataset_conf['test_mask_dir']     = datasets_root + "labels/" + "test"
elif dataset_conf['preprocessed']:
    dataset_conf['train_image_dir']   = datasets_root + 'Preprocessed/' + 'train'
    dataset_conf['train_mask_dir']    = datasets_root + "labels/" + "train"
    dataset_conf['val_image_dir']     = datasets_root + 'Preprocessed/' + 'test'
    dataset_conf['val_mask_dir']      = datasets_root + "labels/" + "test"
    dataset_conf['test_image_dir']    = datasets_root + 'Preprocessed/' + 'test'
    dataset_conf['test_mask_dir']     = datasets_root + "labels/" + "test"
else:
    dataset_conf['train_image_dir']   = datasets_root + 'Orjinal/' + 'train'
    dataset_conf['train_mask_dir']    = datasets_root + "labels/" + "train"
    dataset_conf['val_image_dir']     = datasets_root + 'Orjinal/' + 'test'
    dataset_conf['val_mask_dir']      = datasets_root + "labels/" + "test"
    dataset_conf['test_image_dir']    = datasets_root + 'Orjinal/' + 'test'
    dataset_conf['test_mask_dir']     = datasets_root + "labels/" + "test"

if dataset_conf['cropped']:
    crop_name = f"_crop{dataset_conf['crop_size']}_s{dataset_conf['stride']}"
    dataset_conf['train_image_dir_cropped']   = dataset_conf['train_image_dir'] + crop_name
    dataset_conf['train_mask_dir_cropped']    = dataset_conf['train_mask_dir'] + crop_name 
    dataset_conf['val_image_dir_cropped']     = dataset_conf['val_image_dir'] + crop_name  
    dataset_conf['val_mask_dir_cropped']      = dataset_conf['val_mask_dir'] + crop_name   
    dataset_conf['test_image_dir_cropped']    = dataset_conf['test_image_dir'] + crop_name 
    dataset_conf['test_mask_dir_cropped']     = dataset_conf['test_mask_dir'] + crop_name 
############################################################################################################ 

model_conf['decoder']           = "Unet"
model_conf['encoder']           = "vgg19"
model_conf['encoder_weight']    = "imagenet"
model_conf['activation']        = "sigmoid"

training_conf['batch_size'] = 8
training_conf['epoch'] = 40
training_conf['lr'] = 5e-5
training_conf['weight_decay'] = 1e-2

task_conf = {}
task_conf['dataset_conf'] = dataset_conf
task_conf['model_conf'] = model_conf
task_conf['training_conf'] = training_conf
task_conf['augment_conf'] = augment_conf

prepare_data_step = True
train_step = True
test_step = True

steps = [prepare_data_step,train_step,test_step]


def main_task(task_config, steps, device):
    prepapre_data_step,train_step,test_step = steps

    log_dir = make_log_dir('out')
    save_configs(task_config,log_dir)
    dataset_conf = task_config['dataset_conf']
    model_conf = task_config['model_conf']
    training_conf = task_config['training_conf']
    augment_conf = task_config['augment_conf']
    

    
    if prepapre_data_step:
        check_and_fix_masks_dir(dataset_conf['train_mask_dir'])
        check_and_fix_masks_dir(dataset_conf['val_mask_dir'])
        check_and_fix_masks_dir(dataset_conf['test_mask_dir'])
        if dataset_conf['cropped']:
            initialize_crop_save(dataset_conf)

        #copy_and_paste_folder("images/test/cropped_image")
        #copy_and_paste_folder("images/test/mask/cropped_ma")


        #delete_black_masks("images/train/cropped_image","images/train/mask/cropped_ma",threshold=black_ratio)
        #delete_black_masks("images/test/cropped_image_copy","images/test/mask/cropped_ma_copy",threshold=0)
    


    
    model,train_loader= initialize_train_val(
                                            batch_size = training_conf['batch_size'],
                                            decoder = model_conf['decoder'],
                                            encoder = model_conf['encoder'],
                                            encoder_weight= model_conf['encoder_weight'],
                                            train_image_dir= dataset_conf['train_image_dir_cropped'],
                                            train_mask_dir = dataset_conf['train_mask_dir_cropped'],
                                            resolution= dataset_conf['resolution'],
                                            activation = model_conf['activation'], 
                                            data = dataset_conf['data']
                                            )
                                
    test_loader  = get_test_data(model_conf['encoder'],
                                model_conf['encoder_weight'],
                                dataset_conf['test_image_dir_cropped'],
                                os.path.join(dataset_conf['test_mask_dir_cropped'],dataset_conf['data']),
                                resolution=0)
    if train_step:
        model = train_validate(epoch = training_conf['epoch'],
                                    lr= training_conf['lr'],
                                    weight_decay=training_conf['weight_decay'],
                                    train_loader = train_loader,
                                    valid_loader=test_loader,
                                    encoder=model_conf['encoder'],
                                    model = model,
                                    device = device,
                                    log_dir = log_dir)
        
    if test_step:
        test_model2(model, device, model_conf, dataset_conf, log_dir)

main_task(task_conf,steps,device)


from os.path import join

from mlops_utils import (check_and_fix_masks_dir, make_log_dir, save_configs,
                         send_results_via_mail, write_to_log)
from train_test_val_initialize import *
from utils import *

torch.manual_seed(0)


def main_task(task_config, steps, device):
    prepapre_data_step,train_step,test_step,email_step = steps

    log_dir = make_log_dir('out')
    save_configs(task_config,log_dir)
    dataset_conf = task_config['dataset_conf']
    model_conf = task_config['model_conf']
    training_conf = task_config['training_conf']
    augment_conf = task_config['augment_conf']
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="ret-seg-tuning1a",
        # track hyperparameters and run metadata
        config = {**dataset_conf, **model_conf, **training_conf, **augment_conf}
    )
    

    
    if prepapre_data_step:
        check_and_fix_masks_dir(dataset_conf['train_mask_dir'])
        check_and_fix_masks_dir(dataset_conf['val_mask_dir'])
        check_and_fix_masks_dir(dataset_conf['test_mask_dir'])
        if dataset_conf['cropped']:
            initialize_crop_save(dataset_conf)

        #copy_and_paste_folder("images/test/cropped_image")
        #copy_and_paste_folder("images/test/mask/cropped_ma")


        delete_black_masks(dataset_conf['train_image_dir_cropped'],join(dataset_conf['train_mask_dir_cropped'],dataset_conf['data']),threshold=dataset_conf['black_ratio'])
        delete_black_masks(dataset_conf['val_image_dir_cropped'],join(dataset_conf['val_mask_dir_cropped'],dataset_conf['data']),threshold=dataset_conf['black_ratio'])

    
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
    
    val_loader = get_test_data(model_conf['encoder'],
                                model_conf['encoder_weight'],
                                dataset_conf['val_image_dir_cropped'],
                                join(dataset_conf['val_mask_dir_cropped'],dataset_conf['data']),
                                resolution=0)
                                
    test_loader  = get_test_data(model_conf['encoder'],
                                model_conf['encoder_weight'],
                                dataset_conf['test_image_dir_cropped'],
                                join(dataset_conf['test_mask_dir_cropped'],dataset_conf['data']),
                                resolution=0)
    if train_step:
        model = train_validate(epoch = training_conf['epoch'],
                                lr = training_conf['lr'],
                                weight_decay = training_conf['weight_decay'],
                                train_loader = train_loader,
                                valid_loader =val_loader,
                                encoder = model_conf['encoder'],
                                model = model,
                                device = device,
                                log_dir = log_dir,
                                wandb = wandb)
        
    if test_step:
        test_model2(model, device, model_conf, dataset_conf, log_dir)
    if email_step:
        #send_results_via_mail(log_dir)
        pass
    wandb.finish()


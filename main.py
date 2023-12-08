from utils import *
from train_test_val_initialize import *
from os.path import join
from mlops_utils import make_log_dir,write_to_log,save_configs, check_and_fix_masks_dir,send_results_via_mail


def main_task(task_config, steps, device):
    prepapre_data_step,train_step,test_step,email_step = steps

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
                                join(dataset_conf['test_mask_dir_cropped'],dataset_conf['data']),
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
    if email_step:
        #send_results_via_mail(log_dir)
        pass


import datetime
import json
import logging
import os
import shutil

import segmentation_models_pytorch as smp
import torch
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from segmentation_models_pytorch import utils as ut
from torch.optim.lr_scheduler import StepLR

from Get_model_and_data import *
from mlops_utils import wandb_epoch_log, wandb_final_log
from utils import (auc_pr_folder_calculation, auc_pr_paper_calculation,
                   calculate_metrics, merge_cropped_arrays,
                   merge_cropped_images, plot_save_mismatches,
                   predict_and_save_folder)
from visualiser import plot_pr_curve
from loss import WeightedCombinationLoss



######################ELIMINATE RANDOMNESS#####################
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
##############################################################

def initialize_train_val(
               batch_size,
               decoder,
               encoder,
               encoder_weight,
               train_image_dir,
               train_mask_dir,
               activation,
               data,
               resolution = 0,
               ):
    # TODO: This function is just a wrapper for get_train_val_data_and_model. It should be reconsidered.



    model, train_loader= get_train_val_data_and_model(
        encoder=encoder,
        encoder_weight=encoder_weight,
        decoder=decoder,
        batch_size=batch_size,
        train_image_dir=train_image_dir,
        train_mask_dir= os.path.join(train_mask_dir,data),

        resolution=resolution,
        activation=activation
    )

    return model,train_loader

def initialize_model_info(data,decoder,
               batch_size,
               encoder,
               resolution = 0):
    model_info = {'encoder': encoder,"batch_size" : batch_size, 'resolution': resolution, "data":data,"decoder":decoder}
    return model_info



from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_validate(epoch, lr, weight_decay, model, device, train_loader, valid_loader, log_dir, encoder, freeze_encoder = False):
    # Initialize with WeightedCombinationLoss
    loss = WeightedCombinationLoss(ce_weight=1, dice_weight=0)

    metrics = [
        ut.metrics.IoU(threshold=0.5),
        ut.metrics.Accuracy(threshold=0.5),
        ut.metrics.Recall(threshold=0.5),
        ut.metrics.Fscore(threshold=0.5),
        ut.metrics.Precision(threshold=0.5)
    ]

    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        optimizer = torch.optim.AdamW([
            dict(params=model.decoder.parameters(), lr=lr, weight_decay=weight_decay)
        ])
    else:
        optimizer = torch.optim.AdamW([
            dict(params=model.parameters(), lr=lr, weight_decay=weight_decay)
        ])


    train_epoch = ut.train.TrainEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = ut.train.ValidEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True,
    )

    try:
        max_iou_score = 0
        no_improvement_count = 0


        for i in range(0, epoch + 1):

            logging.info(f'Epoch: {i}')
            logging.info(f'Epoch: {i}, Learning Rate: {optimizer.param_groups[0]["lr"]}')
            

            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            extra_logs = {"lr": optimizer.param_groups[0]["lr"], 'weight_decay': optimizer.param_groups[0]["weight_decay"]}

            wandb.log(wandb_epoch_log(train_logs, valid_logs, extra_logs))

            if max_iou_score < valid_logs['iou_score']:
                max_iou_score = valid_logs['iou_score']
                torch.save(model.state_dict(), os.path.join(log_dir, 'best_step_model.pth'))
                torch.save(optimizer.state_dict(), os.path.join(log_dir, 'best_optimizer.pth'))
                print("Model and optimizer are saved")
                no_improvement_count = 0 
            else:

                if i <35:

                    if i > 3:
                        no_improvement_count += 1

                    if no_improvement_count == 3:
                        model.load_state_dict(torch.load(os.path.join(log_dir, 'best_step_model.pth')))
                        optimizer.load_state_dict(torch.load(os.path.join(log_dir, 'best_optimizer.pth')))
                        train_epoch.optimizer = optimizer
                        train_epoch.model = model
                        valid_epoch.model = model
                        new_lr = optimizer.param_groups[0]["lr"] * 0.5  
                        new_weight_decay = weight_decay * 0.5  
                        
                    
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr
                            param_group['weight_decay'] = new_weight_decay
                        print("Loading the best model and optimizer due to no improvement.")
                        print(f"Learning rate decreased to {new_lr}, Weight decay decreased to {new_weight_decay}")
                        
                        no_improvement_count = 0
                if i == 35:
                    model.load_state_dict(torch.load(os.path.join(log_dir, 'best_step_model.pth')))
                    optimizer.load_state_dict(torch.load(os.path.join(log_dir, 'best_optimizer.pth')))
                    train_epoch.optimizer = optimizer
                    train_epoch.model = model
                    valid_epoch.model = model
                    new_lr = 1e-7
                    new_weight_decay = 1e-7
                    for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr
                            param_group['weight_decay'] = new_weight_decay
                            print("Last Training Part has started ")
                            print(f"Learning rate decreased to {new_lr}, Weight decay decreased to {new_weight_decay}")
                        



    except KeyboardInterrupt:
        print('Training interrupted.')

    model.load_state_dict(torch.load(os.path.join(log_dir, 'best_step_model.pth')))
    os.remove(os.path.join(log_dir, 'best_optimizer.pth'))

    print("Training completed.")
    return model


def test_model(cropped_res, stride, best_model, device, encoder, encoder_weight, log_dir, resolution, data):
    # test_images_dir
    # test_masks_dir
    # test_pred_masks_dir
    # test_pred_masks_probs_dir
    # supports only cropped case

    logging.info(f"Testing model: {log_dir}")

    try:
        predict_and_save_folder(input_folder="images/test/cropped_image", output_maskfolder="images/test/pred_masks", output_prob_folder="images/test/pred_masks_probs", encoder=encoder, encoder_weight=encoder_weight, best_model=best_model, device=device, resolution=resolution)
        logging.info("Prediction and saving completed successfully.")

        merge_cropped_images(2752, 2752, cropped_res=cropped_res, stride=stride, input_dir="images/test/pred_masks", output_dir=f"images/test/merged_pred_masks_{data}")
        logging.info("Merging cropped images completed successfully.")

        merge_cropped_arrays(2752, 2752, cropped_res=cropped_res, stride=stride, input_dir="images/test/pred_masks_probs", output_dir=f"images/test/merged_pred_probs_masks_{data}")
        logging.info("Merging cropped arrays completed successfully.")

        plot_save_mismatches(f"images/test/merged_pred_masks_{data}", f"images/test/mask/{data}", save_dir=log_dir)
        logging.info("Plotting and saving mismatches completed successfully.")

        auc_pr_result = auc_pr_folder_calculation(pred_mask_dir=f"images/test/merged_pred_probs_masks_{data}", test_mask_dir=f"images/test/mask/{data}", stride=stride)
        logging.info("AUC-PR calculation completed successfully.")

        metrics_merged = calculate_metrics(f"images/test/mask/{data}", f"images/test/merged_pred_masks_{data}")
        metrics_cropped = calculate_metrics(f"images/test/mask/cropped_{data}", f"images/test/pred_masks")
        logging.info("Metrics calculation completed successfully.")

        # Save results in a json file
        results = {'auc_pr': auc_pr_result, 'metrics_merged': metrics_merged, 'metrics_cropped': metrics_cropped}
        json_file_path = os.path.join(log_dir, 'results.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(results, json_file, indent=4)
        logging.info("Results saved successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

def test_model2(model, device, model_conf, dataset_conf, log_dir):
    logging.info(f"Testing model: {log_dir}")
    
    # First predict and save cropped image prediction masks
    predict_and_save_folder(input_folder=dataset_conf['test_image_dir_cropped'], output_maskfolder=log_dir+"pred_masks", output_prob_folder=log_dir+"pred_probs", encoder=model_conf['encoder'], encoder_weight=model_conf['encoder_weight'], best_model=model, device=device, resolution=dataset_conf['resolution'])
    logging.info("Prediction and saving completed successfully.")

    merge_cropped_images(3456, 3456, cropped_res=dataset_conf['crop_size'], stride=dataset_conf['stride'], input_dir=log_dir+"pred_masks", output_dir=log_dir+f"merged_pred_masks_{dataset_conf['data']}")
    logging.info("Merging cropped images completed successfully.")

    merge_cropped_arrays(3456, 3456, cropped_res=dataset_conf['crop_size'], stride=dataset_conf['stride'], input_dir=log_dir+"pred_probs", output_dir=log_dir+f"merged_pred_probs_{dataset_conf['data']}")
    logging.info("Merging cropped arrays completed successfully.")

    plot_save_mismatches(log_dir+f"merged_pred_masks_{dataset_conf['data']}", os.path.join(dataset_conf['test_mask_dir'],dataset_conf['data']), save_dir=log_dir)
    logging.info("Plotting and saving mismatches completed successfully.")

    #auc_pr_result = auc_pr_folder_calculation(pred_mask_dir=log_dir+f"merged_pred_probs_{dataset_conf['data']}", test_mask_dir=os.path.join(dataset_conf['test_mask_dir'],dataset_conf['data']), stride=dataset_conf['stride'])
    #logging.info("AUC-PR calculation completed successfully.")

    auc_pr_result_paper, precision, recall = auc_pr_paper_calculation(pred_mask_dir=log_dir+f"merged_pred_probs_{dataset_conf['data']}", test_mask_dir=os.path.join(dataset_conf['test_mask_dir'],dataset_conf['data']), stride=dataset_conf['stride'])
    logging.info("AUC-PR calculation according to paper completed successfully.")
    plot_pr_curve(precision, recall, save_dir=log_dir)
    auc_pr_result = auc_pr_result_paper

    
    metrics_merged = calculate_metrics(os.path.join(dataset_conf['test_mask_dir'],dataset_conf['data']), log_dir+f"merged_pred_masks_{dataset_conf['data']}")
    metrics_cropped = calculate_metrics(os.path.join(dataset_conf['test_mask_dir_cropped'],dataset_conf['data']), log_dir+"pred_masks")
    logging.info("Metrics calculation completed successfully.")

    wandb.log(wandb_final_log(auc_pr_result, metrics_merged, metrics_cropped))
    # Save results in a json file
    results = {'auc_pr': auc_pr_result, 'metrics_merged': metrics_merged, 'metrics_cropped': metrics_cropped}
    json_file_path = os.path.join(log_dir, 'results.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    logging.info("Results saved successfully.")
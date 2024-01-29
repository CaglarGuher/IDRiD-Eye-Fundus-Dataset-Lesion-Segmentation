
import json
import logging
import os
import torch
import wandb
from segmentation_models_pytorch import utils as ut


from Get_model_and_data import *
from mlops_utils import wandb_epoch_log, wandb_final_log
from utils import (auc_pr_folder_calculation, auc_pr_paper_calculation,
                   calculate_metrics, merge_cropped_arrays,
                   merge_cropped_images, plot_save_mismatches,
                   predict_and_save_folder,calculate_save_latest_pred_and_prob)
from visualiser import plot_pr_curve
from loss import WeightedCombinationLoss,FocalLoss



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
    loss = WeightedCombinationLoss(ce_weight = 1,dice_weight = 0)

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
            dict(params=model.decoder.parameters(), lr=lr,weight_decay = weight_decay)
        ])
    else:
        optimizer = torch.optim.Adam([
            dict(params=model.parameters(), lr=lr)
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

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    max_iou_score = 1

    for i in range(epoch):
        logging.info(f'Epoch: {i}')
        logging.info(f'Epoch: {i}, Learning Rate: {optimizer.param_groups[0]["lr"]}')

        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        #wandb.log(wandb_epoch_log(train_logs, valid_logs, {"lr": optimizer.param_groups[0]["lr"]}))

        if max_iou_score > valid_logs['weighted_combination_loss']:
            max_iou_score = valid_logs['weighted_combination_loss']
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
            print("Best model saved")

        scheduler.step(valid_logs['weighted_combination_loss'])  # Scheduler updates learning rate based on validation performance
    
    

    model.load_state_dict(torch.load(os.path.join(log_dir, 'best_model.pth')))
    print("Training completed.")
    return model


def test_model2(model, device, model_conf, dataset_conf, log_dir):

    logging.info(f"Testing model: {log_dir}")
    logging.info("Caglar process  started.")
    calculate_save_latest_pred_and_prob(dir_img=dataset_conf["test_image_dir"],out_pred=log_dir+f"pred_masks_caglar_{dataset_conf['data']}",out_prob=log_dir+f"pred_probs_caglar_{dataset_conf['data']}",model=model,device=device,stride=dataset_conf['stride'],encoder=model_conf['encoder'],encoder_weight=model_conf['encoder_weight'])
    logging.info("Caglar process finished successfully.") 
     
    # First predict and save cropped image prediction masks
    #predict_and_save_folder(input_folder=dataset_conf['test_image_dir_cropped'], output_maskfolder=log_dir+"pred_masks", output_prob_folder=log_dir+"pred_probs", encoder=model_conf['encoder'], encoder_weight=model_conf['encoder_weight'], best_model=model, device=device, resolution=dataset_conf['resolution'])
    #logging.info("Prediction and saving completed successfully.")

    
    #logging.info("calculate_save_latest_pred_and_prob completed successfully.")
    #merge_cropped_images(3456, 3456, cropped_res=dataset_conf['crop_size'], stride=dataset_conf['stride'], input_dir=log_dir+"pred_masks", output_dir=log_dir+f"merged_pred_masks_{dataset_conf['data']}")
    #logging.info("Merging cropped images completed successfully.")

    #merge_cropped_arrays(3456, 3456, cropped_res=dataset_conf['crop_size'], stride=dataset_conf['stride'], input_dir=log_dir+"pred_probs", output_dir=log_dir+f"merged_pred_probs_{dataset_conf['data']}")
    #logging.info("Merging cropped arrays completed successfully.")

    plot_save_mismatches(log_dir+f"pred_masks_caglar_{dataset_conf['data']}", os.path.join(dataset_conf['test_mask_dir'],dataset_conf['data']), save_dir=log_dir,mismatched_images="mismatched_images_caglar")
    logging.info("Plotting and saving mismatches completed successfully.")
    #plot_save_mismatches(log_dir+f"merged_pred_masks_{dataset_conf['data']}", os.path.join(dataset_conf['test_mask_dir'],dataset_conf['data']), save_dir=log_dir,mismatched_images="mismatched_images")
    #auc_pr_result = auc_pr_folder_calculation(pred_mask_dir=log_dir+f"merged_pred_probs_{dataset_conf['data']}", test_mask_dir=os.path.join(dataset_conf['test_mask_dir'],dataset_conf['data']), stride=dataset_conf['stride'])
    #logging.info("AUC-PR calculation completed successfully.")

    #auc_pr_result_paper, precision, recall = auc_pr_paper_calculation(pred_mask_dir=log_dir+f"merged_pred_probs_{dataset_conf['data']}", test_mask_dir=os.path.join(dataset_conf['test_mask_dir'],dataset_conf['data']), stride=dataset_conf['stride'])
    auc_pr_result_paper_caglar, precision_caglar, recall_caglar = auc_pr_paper_calculation(pred_mask_dir=log_dir+f"pred_probs_caglar_{dataset_conf['data']}", test_mask_dir=os.path.join(dataset_conf['test_mask_dir'],dataset_conf['data']), stride=dataset_conf['stride'])
    logging.info("AUC-PR calculation according to paper completed successfully.")
    #auc_pr_result = auc_pr_result_paper

    
    #metrics_merged = calculate_metrics(os.path.join(dataset_conf['test_mask_dir'],dataset_conf['data']), log_dir+f"merged_pred_masks_{dataset_conf['data']}")

    #metrics_cropped = calculate_metrics(os.path.join(dataset_conf['test_mask_dir_cropped'],dataset_conf['data']), log_dir+"pred_masks")

    metrics_caglar = calculate_metrics(os.path.join(dataset_conf['test_mask_dir'],dataset_conf['data']), log_dir+f"pred_masks_caglar_{dataset_conf['data']}")
    logging.info("Metrics calculation completed successfully.")

    #wandb.log(wandb_final_log(auc_pr_caglar=auc_pr_result_paper_caglar,metrics_caglar=metrics_caglar))
    # Save results in a json file
    results = { "auc_pr":auc_pr_result_paper_caglar,"metrics":metrics_caglar}
    json_file_path = os.path.join(log_dir, 'results.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    logging.info("Results saved successfully.")
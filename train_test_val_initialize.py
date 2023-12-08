from Get_model_and_data import *
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils as ut
import logging
import datetime
import shutil
import os
import json
from utils import *
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import StepLR



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


def train_validate(epoch, lr, weight_decay, model, device, train_loader, valid_loader, encoder, log_dir):
    loss= ut.losses.DiceLoss()

    metrics = [
        ut.metrics.IoU(threshold=0.5),
        ut.metrics.Accuracy(threshold=0.5),
        ut.metrics.Recall(threshold=0.5),
        ut.metrics.Fscore(threshold=0.5),
        ut.metrics.Precision(threshold=0.5)
    ]

    optimizer = torch.optim.AdamW([ 
        dict(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    ])

    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=1, factor=0.8, verbose=True)

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
        for i in range(0, epoch+1):
            logging.info(f'Epoch: {i}')
            logging.info(f'Epoch: {i}, Learning Rate: {optimizer.param_groups[0]["lr"]}')

            # Update the learning rate scheduler
   
            plateau_scheduler.step(max_iou_score)

            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)

            if max_iou_score  < valid_logs['iou_score']:
                max_iou_score =  valid_logs['iou_score']
                torch.save(model.state_dict(), os.path.join(log_dir,'best_step_model.pth'))
                print("Model is saved")
    except KeyboardInterrupt:
        print('Training interrupted.')


    model.load_state_dict(torch.load(os.path.join(log_dir,'best_step_model.pth')))
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

    merge_cropped_images(2752, 2752, cropped_res=dataset_conf['crop_size'], stride=dataset_conf['stride'], input_dir=log_dir+"pred_masks", output_dir=log_dir+f"merged_pred_masks_{dataset_conf['data']}")
    logging.info("Merging cropped images completed successfully.")

    merge_cropped_arrays(2752, 2752, cropped_res=dataset_conf['crop_size'], stride=dataset_conf['stride'], input_dir=log_dir+"pred_probs", output_dir=log_dir+f"merged_pred_probs_{dataset_conf['data']}")
    logging.info("Merging cropped arrays completed successfully.")

    plot_save_mismatches(log_dir+f"merged_pred_masks_{dataset_conf['data']}", os.path.join(dataset_conf['test_mask_dir'],dataset_conf['data']), save_dir=log_dir)
    logging.info("Plotting and saving mismatches completed successfully.")

    auc_pr_result = auc_pr_folder_calculation(pred_mask_dir=log_dir+f"merged_pred_probs_{dataset_conf['data']}", test_mask_dir=os.path.join(dataset_conf['test_mask_dir'],dataset_conf['data']), stride=dataset_conf['stride'])
    logging.info("AUC-PR calculation completed successfully.")
    
    metrics_merged = calculate_metrics(os.path.join(dataset_conf['test_mask_dir'],dataset_conf['data']), log_dir+f"merged_pred_masks_{dataset_conf['data']}")
    metrics_cropped = calculate_metrics(os.path.join(dataset_conf['test_mask_dir_cropped'],dataset_conf['data']), log_dir+"pred_masks")
    logging.info("Metrics calculation completed successfully.")

    # Save results in a json file
    results = {'auc_pr': auc_pr_result, 'metrics_merged': metrics_merged, 'metrics_cropped': metrics_cropped}
    json_file_path = os.path.join(log_dir, 'results.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    logging.info("Results saved successfully.")
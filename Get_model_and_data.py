import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from utils import get_preprocessing,get_augmentations
import segmentation_models_pytorch as smp
import logging
from natsort import natsorted
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class Global_Local_Dataset(BaseDataset):
    def __init__(self, array_folder_local, array_folder_global, gt_folder):
        self.array_fps_local = natsorted([os.path.join(array_folder_local, file) for file in os.listdir(array_folder_local)])
        self.array_fps_global = natsorted([os.path.join(array_folder_global, file) for file in os.listdir(array_folder_global)])
        self.gt_fps = natsorted([os.path.join(gt_folder, file) for file in os.listdir(gt_folder)])

    def __getitem__(self, idx):
        local_array = np.load(self.array_fps_local[idx])
        global_array = np.load(self.array_fps_global[idx])
        merged_array = np.stack((local_array, global_array), axis=0)  # Adjust concatenation axis as needed
        gt_image = cv2.imread(self.gt_fps[idx], cv2.IMREAD_GRAYSCALE)
        gt_mask = (gt_image > 1).astype('float')
        return merged_array, gt_mask
    
    def __len__(self):
        return len(self.array_fps_local)






class Dataset(BaseDataset):

    def __init__(
            self,
            images_dir,
            masks_dir,
            lesion_type,
            augmentation=None,
            preprocessing=None,
    ):
        self.masks_ids = natsorted(os.listdir(images_dir)) # every image must have a corresponding mask
        self.augmentation = augmentation
        
        self.masks_fps = []
        for mask_id in self.masks_ids:
                self.masks_fps.append([os.path.join(masks_dir, lesion, mask_id) for lesion in lesion_type]) # multi lesion support

        # self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.masks_ids ] # old single lesion version
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.masks_ids]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = []
        for mask_fp in self.masks_fps[i]:
            mask = cv2.imread(mask_fp, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 1).astype('float')
            masks.append(mask)
        mask = np.stack(masks)
        # reshape mask to w h c
        mask = np.transpose(mask, (1, 2, 0))
        # mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
        # mask = (mask > 1).astype('float')
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.masks_ids)





def get_train_val_data_and_model(encoder,
              encoder_weight,
              decoder,
              batch_size,
              train_image_dir,
              train_mask_dir,
              lesion_type,
              activation,
              resolution=0):

    logger.info(f'Creating model with encoder={encoder}, encoder_weight={encoder_weight}, decoder={decoder}')
    decoder_class = getattr(smp, decoder)

    # Create the model
    model = decoder_class(
        encoder_name=encoder,
        encoder_weights=encoder_weight,
        activation=activation,
        classes=len(lesion_type)
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weight)
    
    train_dataset = Dataset(
        train_image_dir,
        train_mask_dir,
        lesion_type,
        preprocessing=get_preprocessing(preprocessing_fn,resolution),
        augmentation=get_augmentations()
    )


    

    train_loader = DataLoader(train_dataset,batch_size = batch_size, shuffle=True)
 

    
    return model,train_loader

def get_test_data(encoder,
                encoder_weight,
                test_image_dir,
                test_mask_dir,
                lesion_type,
                resolution=0):

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weight)
    

    test_dataset = Dataset(
        test_image_dir,
        test_mask_dir,
        lesion_type,
        preprocessing=get_preprocessing(preprocessing_fn,resolution),
        )


    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False)

    
    return test_loader
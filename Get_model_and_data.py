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
class Dataset(BaseDataset):

    def __init__(
            self,
            images_dir,
            masks_dir,
            augmentation=None,
            preprocessing=None,
    ):
        self.image_ids = natsorted(os.listdir(images_dir))
        self.mask_ids = natsorted(os.listdir(masks_dir))
        self.augmentation = augmentation
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.image_ids]
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.mask_ids]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 1).astype('float')
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.image_ids)


def get_train_val_data_and_model(encoder,
              encoder_weight,
              decoder,
              batch_size,
              train_image_dir,
              train_mask_dir,
              activation,
              resolution=0):

    logger.info(f'Creating model with encoder={encoder}, encoder_weight={encoder_weight}, decoder={decoder}')
    decoder_class = getattr(smp, decoder)

    # Create the model
    model = decoder_class(
        encoder_name=encoder,
        encoder_weights=encoder_weight,
        activation=activation,
    
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weight)
    
    train_dataset = Dataset(
        train_image_dir,
        train_mask_dir,
        preprocessing=get_preprocessing(preprocessing_fn,resolution),
        augmentation=get_augmentations()
    )


    

    train_loader = DataLoader(train_dataset,batch_size = batch_size, shuffle=True)
 

    
    return model,train_loader

def get_test_data(encoder,
              encoder_weight,
              test_image_dir,
              test_mask_dir,
              resolution=0):

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weight)
    

    test_dataset = Dataset(
        test_image_dir,
        test_mask_dir,
        preprocessing=get_preprocessing(preprocessing_fn,resolution),
        )


    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False)

    
    return test_loader
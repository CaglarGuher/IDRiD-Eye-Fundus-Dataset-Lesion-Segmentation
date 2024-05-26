from os.path import join

from mlops_utils import (check_and_fix_masks_dir, make_log_dir, save_configs,
                          write_to_log)
from train_test_val_initialize import *
from utils import *
import wandb
import json
import os

torch.manual_seed(0)
device = "cuda:3"

log_dir = '/home/braincreator/daniel/ademgunesen/yeni/IDRiD-Eye-Fundus-Dataset-Lesion-Segmentation/out/2024_May_19-22_14_07/'
#load training config
with open(log_dir+'training_conf.json') as f:
    training_conf = json.load(f)
# load dataset config
with open(log_dir+'dataset_conf.json') as f:
    dataset_conf = json.load(f)
# load model config
with open(log_dir+'model_conf.json') as f:
    model_conf = json.load(f)

decoder_class = getattr(smp, model_conf['decoder'])

# Create the model
model = decoder_class(
    encoder_name=model_conf['encoder'],
    encoder_weights=model_conf['encoder_weight'],
    activation=model_conf['activation'],
    classes=len(dataset_conf['data'])
    )

# move model to device
model.to(device)

model.load_state_dict(torch.load(os.path.join(log_dir, 'best_model.pth'),map_location=torch.device(device)))
model.to(device)
test_model_LOCAL(model, device, model_conf, dataset_conf, log_dir)
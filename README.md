## Project Setup Guide
## Modeling Without Pbda Method
### Step 1: Download Dataset
- Download the dataset for the project from [this link](https://drive.google.com/drive/folders/1hDovmY747CMykCM-6u_WOz2gZxoYWDH_?usp=drive_link).

### Step 2: Clone Repository
- Open your terminal and run the following command to clone the project repository:

```python
git clone https://github.com/CaglarGuher/IDRiD-Eye-Fundus-Dataset-Lesion-Segmentation.git
```
### Step 3: Configure `task_runner.py`
- Navigate to the `task_runner.py` file in the cloned repository.
- Modify the following settings in the file:
- Change the dataset root to the folder you downloaded and named "square format".
- Choose whether to run the script with the preprocessed dataset (set to `True`) or test the preprocessed version (also set to `True`, but not both).
- Adjust the cropping size (e.g., 576x576) and stride (ensure it is divisible by the original dataset size, 3456x3456).
- Set the black ratio to determine the percentage of cropped images with black to delete (1 means delete every cropped image with black, which can improve model performance).
- Specify the lesion type to work with (e.g., microaneurysm: `ma`, hard exudate: `ex`, soft exudate: `se`, hemorrhage: `he`).
- Choose the encoder-decoder model to use (e.g., Unetplusplus with VGG19 as the encoder).
- Decide whether to freeze all encoder layers during training (set to `True` to freeze).
- Set the activation function (e.g., sigmoid, relu).
- Adjust other parameters like batch size, epochs, learning rate, and weight decay as needed.

### Step 4: Run the Code
- Once the configuration is complete, run the script by executing the following command in your terminal:

```python
python task_runner.py
```
### Step 5: Review Output
- After the script completes, check the `out` folder for the results.
- You'll find mismatched images, models, and prediction arrays.
- Mismatched images are displayed with green representing True Positives (TP), red representing Ground Truth, and blue representing False Positives (FP).

### Step 6: Additional Notes
- If you've run the code once with the same image type and cropping size, you can set `prepare_data_step` to `False` to save time.
- The loss function for the project is Binary Cross Entropy (BCE) loss, and the optimizer is Adam with weight decay.
## Modeling With Pbda Method
will be added soon





















## EXPLANATION OF THE DISEASE

Diabetic retinopathy (DR) is a leading cause of blindness in adults due to diabetes, affecting over 400 million people worldwide. Early diagnosis through fundus photography is crucial for effective treatment. However, there's inconsistency among specialists in diagnosing DR due to the lack of specific criteria for classifying its stages.

**Shows inconsistency among doctors in assessing DR stages.**

### MICROANEURYSMS

Microaneurysms are the earliest signs of DR, appearing as small, round, dark red dots. They can be up to 125 micrometers in size, smaller than the thickest blood vessels. See Figure 3 for an example.

### HEMORRHAGES

Hemorrhages result from the rupture of microaneurysms or blood vessels, causing bleeding. They are usually larger than microaneurysms and vary in shape. Unlike microaneurysms, they may not have sharp borders or colors. See Figure 4 for an example.

### EXUDATES

Exudates refer to the leakage of fluids, proteins, and other substances from retinal blood vessels. There are various types, including hard exudates and cotton wool spots. Hard exudates appear as small, yellow-white deposits, while cotton wool spots are cloud-like, fuzzy, white, or gray lesions observed in the retina. See Figure 5 for examples.

## DIABETIC RETINOPATHY STAGES

DR has two main stages: proliferative and non-proliferative, with the non-proliferative stage further divided into three. This makes a total of four stages:

1. **Mild NPDR (Non-Proliferative Diabetic Retinopathy):** Minor damage and swelling in the retinal blood vessels, with microaneurysms present. Vision problems are usually minimal, and treatment may not be necessary.
2. **Moderate NPDR:** More pronounced vessel damage, with increased narrowing and deposits on the retinal surface. Vision blurring and other symptoms may occur, and treatment involves monitoring and improved diabetes control.
3. **Severe NPDR:** Significant vessel damage leading to reduced eye nourishment. Bleeding and yellow spots on the retina become more prominent. Without treatment, vision problems can become permanent.
4. **PDR (Proliferative Diabetic Retinopathy):** The most severe stage, characterized by serious vessel damage and the formation of abnormal new vessels. This stage can lead to serious complications requiring urgent medical intervention.

## POPULAR SEGMENTATION MODELS FOR BIOMEDICAL IMAGES

1. **UNet:** Known for its encoder-decoder structure, capturing hierarchical features using convolutional and pooling layers. Skip connections preserve detailed information.
2. **DeepLab:** Uses dilated convolutions and atrous spatial pyramid pooling (ASPP) to capture multi-scale features. Variants like DeepLabv3 enhance its capabilities.
3. **FCN (Fully Convolutional Networks):** Composed of convolutional layers, preserving spatial information throughout the network.
4. **Mask R-CNN:** Originally for object detection, includes a mask prediction branch for pixel-level segmentation.
5. **UNet++:** Extends UNet with nested skip pathways for improved feature capture.
6. **Attention U-Net:** Incorporates attention mechanisms to focus on specific regions of the image.
7. **ResUNet:** Combines UNet with residual connections from ResNet, enabling deeper networks.

These models are widely used in biomedical imaging due to their effectiveness in segmenting images, including those related to diabetic retinopathy.

### RESULT COMPARISON FOR PREVIOUS STUDIES

Table 1 presents results of studies on the IDRiD dataset, showing AUPR for various lesion types.

| Method            | AUPR (EX) | AUPR (HE) | AUPR (MA) | AUPR (SE) | Average |
|-------------------|-----------|-----------|-----------|-----------|---------|
| L-Seg             | 79.45     | 63.74     | 46.27     | 71.13     | 65.15   |
| Local-Global UNets| 88.90     | 70.30     | 52.50     | 67.90     | 69.90   |
| VRT               | 71.27     | 68.04     | 49.51     | 69.95     | 64.69   |
| PATech            | 88.50     | 64.90     | 47.40     | -         | -       |
| iFLYTEK-MIG      | 87.40     | 55.88     | 50.17     | 65.88     | 64.84   |
| PBDA              | 86.43     | 71.53     | 53.41     | 73.07     | 71.11   |
| PBDA (UNet++)     | 81.04     | 64.08     | 49.17     | 68.88     | 65.48   |

### PROJECT

Preprocessing operations, shown in Figures 8 and 9, are used.

#### NOISE REDUCTION

The BSRGAN model is used for noise reduction, producing efficient results.

The examples for the different upscaling models along with the bsrgan is shown in below. The results for eye fundus images can be seen.

#### CROPPING AND SLIDING WINDOW TECHNIQUES

To address the limited number of annotated images in medical datasets like retinal images, a cropping strategy is used to resize the images into 576x576-pixel patches, balancing dataset size and information content. in order to crop the images by 576x576 the images are reshaped in to 3456x3456 with adding black borders as it can be seen in.

#### SLIDING WINDOW TECHNIQUE

The sliding window technique is then applied with a specified overlap to augment the dataset, ensuring lesion coverage while increasing the diversity of training samples.

| Lesion Location | Accuracy | Precision | Recall | F1 Score | IoU |
|-----------------|----------|-----------|--------|----------|-----|
| Central         | 0.80     | 0.58      | 0.43   | 0.49     | 0.34|
| Peripheral      | 0.81     | 0.57      | 0.39   | 0.46     | 0.33|

#### CENTER MERGE ALGORITHM

Patches created by sliding a window over the image were merged to form a complete prediction image. The center merge algorithm, shown in Figure 5.3, used only the center region of each patch for composition.

The model's edge detection is weaker due to partial lesions at the edges, which affects pixel adjacency and lesion structure. To address this, only the center part of the patches was used.

#### PBDA (POISSON BLEND DATA AUGMENTATION) METHOD

The PBDA method seamlessly integrates synthetic lesions into retinal images, enhancing dataset diversity. Small lesion images with masks are placed in various positions within the retinal images, and the Poisson blending method is used to ensure realistic integration. Random transformations are applied to the synthetic lesions to increase diversity and prevent overfitting. This method not only introduces lesions but also creates variations of retinal images, contributing to the model's robustness.

## Experimental Results

During the study, the methods explained in the previous sections were individually tested with isolated experiments. Additionally, each of the 4 lesion types was independently tested, and the results are presented. The effects of cropping, sliding window, preprocessing techniques, noise reduction techniques, the PBDA method, and the use of healthy images for each lesion are listed. The experimental results are presented using the following metrics: Accuracy, Recall, Precision, F1 Score, IoU, and AUC under the precision-recall curve.

### Results for Microaneurysm Class

#### Effect of Cropping

| Cropping Size | Accuracy | F1 Score | IoU | Precision | Recall | AUC PR |
|---------------|----------|----------|-----|-----------|--------|--------|
| 288           | 0.6707   | 0.4729   | 0.3096 | 0.4846 | 0.4616 | 0.4670 |
| 576           | 0.7331   | 0.4985   | 0.3432 | 0.55439| 0.4528 | 0.4948 |
| 1152          | 0.7670   | 0.4691   | 0.3077 | 0.5682  | 0.3995 | 0.3650 |

The highest-performing cropping size was 576x576 pixels, providing a balance between local information and the overall image.

#### Effect of Sliding Window

| Cropping Size | Sliding Size | Accuracy | F1 Score | IoU | Precision | Recall | AUC PR |
|---------------|--------------|----------|----------|-----|-----------|--------|--------|
| 576           | 576          | 0.7331   | 0.4985   | 0.3432 | 0.5543    | 0.4528 | 0.4948 |
| 576           | 288          | 0.7443   | 0.5068   | 0.3508 | 0.5703    | 0.4560 | 0.5120 |

Overlapping sliding windows during training and testing improved performance compared to non-overlapping windows.

#### Effect of Preprocessing Technique

| Preprocessing | Accuracy | F1 Score | IoU | Precision | Recall | AUC PR |
|---------------|----------|----------|-----|-----------|--------|--------|
| None          | 0.7443   | 0.5068   | 0.3508 | 0.5703    | 0.4560 | 0.5120 |
| Applied       | 0.7853   | 0.4983   | 0.3538 | 0.6072    | 0.4225 | 0.5239 |

The preprocessing technique improved the model's performance for microaneurysms.

#### Effect of Noise Reduction Technique

| Noise Reduction | Accuracy | F1 Score | IoU | Precision | Recall | AUC PR |
|-----------------|----------|----------|-----|-----------|--------|--------|
| None            | 0.7853   | 0.4983   | 0.3538 | 0.6072    | 0.4225 | 0.5239 |
| Applied         | 0.8253   | 0.4819   | 0.3586 | 0.6451    | 0.3845 | 0.5297 |

The noise reduction technique improved various performance metrics.

#### Effect of PBDA Method

| PBDA  | Accuracy | F1 Score | IoU | Precision | Recall | AUC PR |
|-------|----------|----------|-----|-----------|--------|--------|
| None  | 0.8253   | 0.4819   | 0.3538 | 0.6451    | 0.3845 | 0.5297 |
| Applied | 0.8331 | 0.4764   | 0.3238 | 0.6521    | 0.3752 | 0.5454 |

The PBDA method, which involved augmenting the dataset, resulted in a model with improved confidence in predictions.

#### Effect of Healthy Data

| Data Used | Accuracy | F1 Score | IoU | Precision | Recall | AUC PR |
|-----------|----------|----------|-----|-----------|--------|--------|
| PBDA      | 0.8331   | 0.4764   | 0.3238 | 0.6521    | 0.3752 | 0.5454 |
| PBDA + Healthy | 0.8376 | 0.4831 | 0.3298 | 0.6623    | 0.3802 | 0.5646 |

By adding healthy retina images to the dataset, the model's performance improved across all metrics.

### Results for Exudates Class

#### Effect of Cropping

Comparison between 576x576 and 1152x1152 pixel cropping sizes showed that 576x576 pixels provided better performance for exudate detection.

| Cropping Size | AUC PR | F1 Score | Accuracy | Precision | Recall | IoU |
|----------------|--------|----------|----------|------------|--------|-----|
| 576x576        | 0.8902 | 0.7998   | 0.9021   | 0.8719     | 0.7386 | 0.6664 |
| 1152x1152      | 0.8231 | 0.7399   | 0.8761   | 0.8258     | 0.6701 | 0.6015 |

#### Effect of Sliding Window

The sliding window technique improved the model's performance for both 576x576 and 288x288 pixel cropping sizes.

| Sliding | AUC PR | F1 Score | Accuracy | Precision | Recall | IoU |
|---------|--------|----------|----------|-----------|--------|-----|
| 576     | 0.8902 | 0.7998   | 0.9021   | 0.8719    | 0.7386 | 0.6664 |
#### Effect of Preprocessing Technique:
Preprocessing techniques improved the model's performance for exudate detection.

| Preprocessing | F1 Score | Accuracy | Precision | Recall | IoU | AUC PR |
| ------------- | -------- | -------- | ----------| ------ | --- | ------ |
| None          | 0.7998   | 0.9021   | 0.8719    | 0.7386 | 0.6664 | 0.8902 |
| Applied       | 0.8132   | 0.8823   | 0.8535    | 0.7766 | 0.6852 | 0.8972 |

#### Effect of Noise Reduction Technique:
The noise reduction technique had varying effects based on the cropping size, showing improvements for 1152x1152 but a decrease for 576x576.

| Noise Reduction | F1 Score | Accuracy | Precision | Recall | IoU | AUC PR |
| ---------------- | -------- | -------- | ----------| ------ | --- | ------ |
| None             | 0.7399   | 0.8761   | 0.8258    | 0.6701 | 0.6015 | 0.8231 |
| Applied          | 0.7645   | 0.8607   | 0.8162    | 0.7189 | 0.6341 | 0.8497 |

####Effect of PBDA Method:
The PBDA method improved the model's performance for detecting exudates.

| PBDA  | F1 Score | Accuracy | Precision | Recall | IoU | AUC PR |
| ----- | -------- | -------- | ----------| ------ | --- | ------ |
| None  | 0.7998   | 0.9021   | 0.8719    | 0.7386 | 0.6664 | 0.8902 |
| Applied | 0.8174  | 0.8800   | 0.8520    | 0.7855 | 0.7526 | 0.9019 |

####Effect of Healthy Data:
Including healthy data did not significantly improve the model's performance for exudate detection.

| Healthy Data | F1 Score | Accuracy | Precision | Recall | IoU | AUC PR |
| ------------ | -------- | -------- | ----------| ------ | --- | ------ |
| PBDA         | 0.8174   | 0.8800   | 0.8520    | 0.7855 | 0.7526 | 0.9019 |
| Healthy      | 0.8042   | 0.9060   | 0.8774    | 0.7423 | 0.7345 | 0.8959 |


import os
from PIL import Image
import numpy as np
import albumentations as albu
import cv2
from tqdm import tqdm
import random
import torch
import segmentation_models_pytorch as smp
from natsort import natsorted
from sklearn.metrics import jaccard_score
from sklearn.metrics import auc, precision_score, recall_score
import numpy as np
import shutil
from tqdm import tqdm

def to_tensor(x, **kwargs):
    try:
        
        if isinstance(x, np.ndarray):
            
            if len(x.shape) == 3:
                return x.transpose(2, 0, 1).astype('float32')
            else:
                return x.astype('float32')
        else:
            raise ValueError("Input is not a numpy array.")
    except Exception as e:
        print("Error in to_tensor:", e)
        return None
    


def get_preprocessing(preprocessing_fn,resolution,fn = True):

    if resolution > 10:    
        _transform = [
            
            albu.Lambda(image=preprocessing_fn),
            albu.Resize(resolution,resolution),
            albu.Lambda(image=to_tensor, mask=to_tensor),

        ]
    else:
        _transform = [
            
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=to_tensor, mask=to_tensor),
        ]
    
    return albu.Compose(_transform)

def delete_extra_images(folder1, folder2):
    #make the train and the groundtruths pair and delete the images if dont have gt 
   
    files_folder1 = os.listdir(folder1)
    for file_name in files_folder1:
        file_path_folder2 = os.path.join(folder2, file_name)
        if os.path.exists(file_path_folder2):
            pass
        else:
            file_path_folder1 = os.path.join(folder1, file_name)
            os.remove(file_path_folder1)
    files_folder2 = os.listdir(folder2)
    for file_name in files_folder2:
        file_path_folder1 = os.path.join(folder1, file_name)
        if os.path.exists(file_path_folder1):
            pass
        else:
            file_path_folder2 = os.path.join(folder2, file_name)
            os.remove(file_path_folder2)

def is_mostly_black(image_path, threshold=0.9):
    img = Image.open(image_path)
    width, height = img.size
    total_pixels = width * height

    # Convert the image to grayscale
    img_gray = img.convert('L')

    # Count the number of black pixels
    black_pixels = sum(1 for pixel in img_gray.getdata() if pixel == 0)

    # Check if the percentage of black pixels is above the threshold
    black_percentage = black_pixels / total_pixels
    return black_percentage >= threshold


def is_image_black(image_path, threshold=1):
    """
    Check if an image is completely black based on a threshold value.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return cv2.countNonZero(img) <= threshold



def delete_black_images(folder_path, deletion_percentage):
    """
    Delete a percentage of black images in a folder.
    """
    image_files = [f for f in os.listdir(folder_path)]
    image_files = natsorted(image_files)
    black_images = [f for f in image_files if is_image_black(os.path.join(folder_path, f))]
    black_images = natsorted(black_images)
    num_images_to_delete = int(len(black_images) * (deletion_percentage / 100))

    random.shuffle(black_images)

    for image_file in tqdm(black_images[:num_images_to_delete], desc='Processing images'):
        image_path = os.path.join(folder_path, image_file)
        os.remove(image_path)

def convert_tif_to_jpg(input_folder, output_folder):
    
    os.makedirs(output_folder, exist_ok=True)

   
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
           
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".jpg")

            # Open the TIFF image
            tif_image = Image.open(input_path)

           
            grayscale_image = tif_image.convert("L")

           
            grayscale_image.save(output_path, "JPEG")

            print(f"Converted: {filename}")


def predict_mask(best_model,imagepath,encoder,resolution,encoder_weight,device):
    image = cv2.imread(imagepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder,encoder_weight)
    sample = get_preprocessing(preprocessing_fn,resolution)(image=image)
    image = sample["image"]
    image = torch.from_numpy(image).to(device).unsqueeze(0)
    with torch.no_grad():
        predicted_masks = best_model.predict(image)
        predicted_masks = (predicted_masks.squeeze().cpu().numpy().round())
    predicted_mask_image = Image.fromarray((predicted_masks * 255).astype('uint8'))
    return predicted_mask_image


def predict_and_save_folder(input_folder, output_maskfolder,output_prob_folder, encoder, encoder_weight,best_model,device,resolution):
    
    os.makedirs(output_maskfolder, exist_ok=True)
    os.makedirs(output_prob_folder, exist_ok=True)

   
    image_files = [f for f in os.listdir(input_folder) ]
    image_files = natsorted(image_files)

    for image_file in tqdm(image_files,desc='Predicting masks'):
       
        image_path = os.path.join(input_folder, image_file)
        predicted_mask = predict_mask(imagepath=image_path, best_model=best_model,encoder=encoder, encoder_weight=encoder_weight,device=device,resolution=resolution)
        predicted_mask_prob = predict_mask_probs(best_model=best_model,imagepath=image_path,encoder=encoder,encoder_weight=encoder_weight,DEVICE=device,resolution=resolution)


        output_path_mask = os.path.join(output_maskfolder, image_file)
        predicted_mask.save(output_path_mask)
        output_path_probs = os.path.join(output_prob_folder, f"{image_file}_probs.npy")
        np.save(output_path_probs, predicted_mask_prob)



def vis_gt_predicted(image1,image2):

    red = np.array([0, 0, 255], dtype=np.uint8)  
    blue = np.array([255, 0, 0], dtype=np.uint8)  
    green = np.array([0, 255, 0], dtype=np.uint8)  

  

    red = np.clip(red , 0, 255).astype(np.uint8)
    blue = np.clip(blue , 0, 255).astype(np.uint8)
    green = np.clip(green , 0, 255).astype(np.uint8)

    
    non_black1 = (image1 > 1).all(axis=2)
    non_black2 = (image2 > 1).all(axis=2)

    #non_black = red
    image1[non_black1] = red

    #non_black = blue
    image2[non_black2] = blue

    # Merge 
    merged_image = cv2.addWeighted(image1, 0.5, image2, 0.5, 0)

 
    intersection = np.minimum(image1, image2)
    intersection[non_black1 | non_black2] = green

    return merged_image


def crop_save_mask_images(input_image_dir,output_image_dir,crop_size,stride):

    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
 
    input_image_files = [f for f in os.listdir(input_image_dir) ]
    input_image_files = natsorted(input_image_files)
    for input_file in tqdm(input_image_files,desc=f'Cropping {input_image_dir}'):
        input_path = os.path.join(input_image_dir, input_file)
        vis_image = Image.open(input_path)
        original_width, original_height = vis_image.size
        new_width = original_width - (original_width % stride)
        new_height = original_height - (original_height % stride)
        vis_image = vis_image.resize((new_width, new_height))
        original_width, original_height = vis_image.size

        small_images = []

        for y in range(0, original_height - crop_size + 1, stride):
            for x in range(0, original_width - crop_size + 1, stride):
                small_image = vis_image.crop((x, y, x + crop_size, y + crop_size))
                small_images.append(small_image)

        for i, small_image in enumerate(small_images):
            
            output_file_path = os.path.join(output_image_dir, f"{input_file}_{i + 1}.png")
            small_image.save(output_file_path)
    
#original input size
#new_width = original_width - (original_width % width),new_height = original_height - (original_height % height)
#cropped_imagesizes 
#new_width*new_height/croopewidth*cropped_height

def merge_cropped_images(original_width,original_height,cropped_res,stride,input_dir,output_dir):
    new_width = original_width - (original_width % stride)
    new_height = original_height - (original_height % stride)
    set_size = (new_width*new_height)/cropped_res**2
    filenames = os.listdir(input_dir)
    filenames = natsorted(filenames)
    os.makedirs(output_dir, exist_ok=True)
    for set_start in range(0, len(filenames), int(set_size)):
        new_image = Image.new('RGB', (new_width, new_height))
        for row in range(original_height // cropped_res):
            for col in range(original_width // cropped_res):
                index = set_start + row * (original_width // cropped_res) + col 
                if index < len(filenames):
                    paste_position = (col * cropped_res, row * cropped_res)    
                    cropped_image = Image.open(os.path.join(input_dir, filenames[index]))
                    new_image.paste(cropped_image, paste_position)
        # resize the image to the original size
        result_image = new_image.resize((original_width, original_height))
        result_image.save(os.path.join(output_dir, f'merged_{filenames[set_start]}.png'))

def merge_cropped_arrays(original_width, original_height, cropped_res, stride, input_dir, output_dir):
    new_width = original_width - (original_width % stride)
    new_height = original_height - (original_height % stride)
    set_size = (new_width * new_height) / cropped_res**2
    filenames = os.listdir(input_dir)
    filenames = natsorted(filenames)
    os.makedirs(output_dir, exist_ok=True)

    for set_start in range(0, len(filenames), int(set_size)):
        merged_prob_array = np.zeros((new_height, new_width), dtype=float)

        for row in range(original_height // cropped_res):
            for col in range(original_width // cropped_res):
                index = set_start + row * (original_width // cropped_res) + col
                if index < len(filenames):
                    prob_array = np.load(os.path.join(input_dir, filenames[index]))

                    # Calculate paste position in the merged array
                    paste_position_y = row * cropped_res
                    paste_position_x = col * cropped_res

                    # Paste the probability array into the merged array
                    merged_prob_array[paste_position_y:paste_position_y + cropped_res,
                                      paste_position_x:paste_position_x + cropped_res] = prob_array

        # Save the merged probability array
        output_path = os.path.join(output_dir, f'merged_{filenames[set_start]}')
        np.save(output_path, merged_prob_array)

def read_images_from_folder(folder_path):
    image_paths = [os.path.join(folder_path, filename) for filename in natsorted(os.listdir(folder_path))]
    images = [np.array(Image.open(image_path)) for image_path in image_paths]
    return images



def colorize_mismatches(ground_truth, prediction):
    """
    Colorizes mismatches between two binary images.

    Parameters:
    ground_truth (numpy.ndarray): A binary image representing the ground truth.
    prediction (numpy.ndarray): A binary image representing the prediction.

    Returns:
    PIL.Image.Image: A PIL image with colorized mismatches.
    """
    # Initialize an RGB image with the same dimensions, filled with black color
    height, width = ground_truth.shape
    colorized = np.zeros((height, width, 3), dtype=np.uint8)  # Black color

    # Green (1 in both ground truth and prediction)
    green_mask = (ground_truth == 1) & (prediction == 1)
    # print number of green pixels
    print(f"Number of green pixels: {np.sum(green_mask)}")
    colorized[green_mask] = [0, 255, 0]  # Green color

    # Red (1 in ground truth, 0 in prediction)
    red_mask = (ground_truth == 1) & (prediction == 0)
    colorized[red_mask] = [255, 0, 0]  # Red color

    # Yellow (0 in ground truth, 1 in prediction)
    yellow_mask = (ground_truth == 0) & (prediction == 1)
    colorized[yellow_mask] = [0, 0, 255]  # Yellow color

    # Convert to PIL Image
    return colorized

from sklearn.metrics import precision_recall_curve, auc

def calculate_auc_pr(y_true, y_scores):
    '''
    Calculate area under precision recall curve using scikit-learn.

    Args:
        y_true: Ground truth labels (binary numpy image).
        y_scores: Predicted probabilities (numpy image with values between 0 and 1).

    Returns:
        auc_pr: Area under precision recall curve.
    '''
    y_true_flat = y_true.flatten()
    y_scores_flat = y_scores.flatten()

    precision, recall, _ = precision_recall_curve(y_true_flat, y_scores_flat)
    auc_pr = auc(recall, precision)

    return auc_pr


def calculate_iou(image1, image2):
    # Convert images to binary masks assume grayscaled
    mask1 = (image1 > 0).astype(int)
    mask2 = (image2 > 0).astype(int)

    
    flat_mask1 = mask1.flatten()
    flat_mask2 = mask2.flatten()

    
    iou_score = jaccard_score(flat_mask1, flat_mask2)
    return iou_score

def calculate_iou_for_folders(folder1_path, folder2_path):
    images1 = read_images_from_folder(folder1_path)
    images2 = read_images_from_folder(folder2_path)

    iou_scores = []

    for image1, image2 in zip(images1, images2):
        iou_score = calculate_iou(image1, image2)
        iou_scores.append(iou_score)

    average_iou = np.mean(iou_scores)
    return average_iou

def predict_mask_probs(best_model, imagepath, encoder, encoder_weight,resolution ,DEVICE):
    image = cv2.imread(imagepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weight)
    sample = get_preprocessing(preprocessing_fn,resolution)(image=image)
    image = sample["image"]
    image = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    
    with torch.no_grad():
        predicted_masks = best_model.predict(image).squeeze().cpu().numpy().astype(np.float32)
        predicted_masks =predicted_masks.astype(np.float32)
    
    return predicted_masks

def detect_crop_save_image_mask(image_folder,gt_folder, output_img,output_grt,margin):
   #margin is how big the cropping size will be outside of the mask info
    os.makedirs(output_img, exist_ok=True)
    os.makedirs(output_grt, exist_ok=True)

    # Get a list of image filenames and ground truth filenames
    image_filenames = natsorted(os.listdir(image_folder))
    gt_filenames = natsorted(os.listdir(gt_folder))


    for image_filename, gt_filename in zip(image_filenames, gt_filenames):
        image_path = os.path.join(image_folder, image_filename)
        gt_path = os.path.join(gt_folder, gt_filename)
        original_image = cv2.imread(image_path)
        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        _, thresholded_image = cv2.threshold(gt_image, 5, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)

            # Expand the bounding box by the specified margin
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(original_image.shape[1] - x, w + 2 * margin)
            h = min(original_image.shape[0] - y, h + 2 * margin)

            # Extract the cropped region
            cropped_region = original_image[y:y+h, x:x+w]
            cropped_groundtruth = gt_image[y:y+h, x:x+w]
            # Save the cropped region
            output_path_0 = os.path.join(output_img, f"{os.path.splitext(image_filename)[0]}_{i}.jpg")
            output_path_1 = os.path.join(output_grt, f"{os.path.splitext(image_filename)[0]}_{i}.tif")
            cv2.imwrite(output_path_0, cropped_region)
            cv2.imwrite(output_path_1, cropped_groundtruth)
def find_non_black_coordinates(image):
    non_black_coordinates = np.column_stack(np.where(image > 20))
    return non_black_coordinates
def poisson_blend(small_img, small_mask, big_img, position):
    result = cv2.seamlessClone(small_img, big_img, small_mask, position, cv2.MIXED_CLONE)
    return result
def give_corr_mask(selected_image_file):    
    mask_filename = f"{os.path.splitext(selected_image_file)[0]}.tif"
    return mask_filename
def create_syntetic_image(ma_dir ="",he_dir="",ex_dir="",se_dir="",ma_img_dir="",
                          he_img_dir="",ex_img_dir="",se_img_dir="",output_mask_dir="",output_img_dir="",
                          ma_numb =(0,0),he_numb =(0,0),ex_numb =(0,0),se_numb =(0,0)):

    
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)

    mask_folders = dict(he_dir, ma_dir, se_dir, ex_dir)
    image_folders = dict(he_img_dir, ma_img_dir, se_img_dir, ex_img_dir)
    numb_lists = dict(he_numb,ma_numb,se_numb,ex_numb)
    original_image = cv2.imread("Healthy_Images/IDRiD_029.JPG")
    original_image_gray = cv2.imread("Healthy_Images/IDRiD_029.JPG", cv2.IMREAD_GRAYSCALE)
    output_image = original_image.copy()

    for mask_folder, img_folder,numb_list in zip(mask_folders, image_folders,numb_lists):
        num_masks = random.randint(numb_list)
        image_files = [f for f in os.listdir(img_folder)]
        random.shuffle(image_files)
        background = np.zeros_like(original_image_gray)
        non_black_coordinates = find_non_black_coordinates(original_image_gray)

        for i in range(0, num_masks):
            image_path = os.path.join(img_folder, image_files[i])
            mask = cv2.imread(f"{mask_folder}/{give_corr_mask(image_files[i])}")
            mask_gray = cv2.imread(f"{mask_folder}/{give_corr_mask(image_files[i])}", cv2.IMREAD_GRAYSCALE)
            corresponding_image = cv2.imread(image_path)
            mask[mask > 0] = 255
            ret, mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)

            # Create a kernel for dilation
            kernel = np.ones((30, 30), np.uint8)

            # Perform dilation on the mask
            mask = cv2.dilate(mask, kernel, iterations=1)
            background_height, background_width = 2848, 4288

            while True:
                coordinates = non_black_coordinates[np.random.choice(non_black_coordinates.shape[0], 1, replace=False)]
                y_position, x_position = coordinates[0][0], coordinates[0][1]

                if (
                        0 <= y_position < background_height - mask_gray.shape[0] and
                        0 <= x_position < background_width - mask_gray.shape[1]
                ):
                    try:
                        # Check if the pixel at the chosen position is not black
                        if  np.any(original_image_gray[y_position:y_position + mask_gray.shape[0],
                            x_position:x_position + mask_gray.shape[1]] == 0):
                            continue

                        # Place the mask_gray on the background
                        background[y_position:y_position + mask_gray.shape[0],
                        x_position:x_position + mask_gray.shape[1]] = mask_gray

                        # Calculate the position for poisson_blend
                        position = (x_position + (mask_gray.shape[1] // 2), y_position + (mask_gray.shape[0] // 2))

                        # Perform poisson_blend
                        output_image = poisson_blend(corresponding_image, mask, output_image, position)

                        # If successful, exit the loop
                        break
                    except Exception as e:
                        print(f"Error in poisson_blend: {e}")
                        continue

        cv2.imwrite(f"test_{mask_folder}.tif", background)

    cv2.imwrite("test.jpg", output_image)

def calculate_metrics(gt_folder, pred_folder,threshold=0.5):
    gt_files = os.listdir(gt_folder)
    gt_files = natsorted(gt_files)
    pred_files = os.listdir(pred_folder)
    pred_files = natsorted(pred_files)
    tp = fp = fn = 0

    for pred,mask  in zip(pred_files,gt_files):
    
        gt_path = os.path.join(gt_folder, mask)
        pred_path = os.path.join(pred_folder, pred)

        # Read ground truth and prediction images
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_img = (gt_img/255)>0
        
        # Binarize ground truth using a threshold (assuming it's not binary

        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        pred_img = (pred_img/255)
        # Binarize prediction using the specified threshold
        pred_img = (pred_img > threshold).astype(np.uint8)

        # Calculate intersection, union, true positive, false positive, and false negative
        intersection = np.logical_and(gt_img, pred_img)
        union = tp + fp + fn
        tp += np.sum(intersection)
        fp += np.sum(pred_img) - np.sum(intersection)
        fn += np.sum(gt_img) - np.sum(intersection)

    iou = tp / union if union > 0 else 0
    accuracy = (tp + fn) / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fscore = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"iou": iou, "accuracy": accuracy, "recall": recall, "fscore": fscore, "precision": precision}


def create_test_data(image_dir):
    original_images = os.listdir(f"{image_dir}/images")
    original_masks = os.listdir(f"{image_dir}/masks")
    original_images = natsorted(original_images)
    original_masks = natsorted(original_masks)
    for image in original_images:
        os.makedirs("images/test/image",exist_ok= True)
        image_path = os.path.join(image_dir, 'images', image)
        shutil.copy(image_path, "images/test/image")
    # Copy images to the validation director
    for mask_test in original_masks:
        
        gt_test = cv2.imread(f"{image_dir}/masks/{mask_test}",cv2.IMREAD_GRAYSCALE)
        
        ma_test = np.where(gt_test == 3,255,0)
        he_test = np.where(gt_test == 2,255,0)
        se_test = np.where(gt_test == 4,255,0)
        ex_test = np.where(gt_test == 1,255,0)
        os.makedirs("images/test/mask/ma",exist_ok=True)
        os.makedirs("images/test/mask/ex",exist_ok=True)
        os.makedirs("images/test/mask/se",exist_ok=True)
        os.makedirs("images/test/mask/he",exist_ok=True)
        cv2.imwrite(f"images/test/mask/ma/{mask_test}.png", ma_test)
        cv2.imwrite(f"images/test/mask/ex/{mask_test}.png", ex_test)
        cv2.imwrite(f"images/test/mask/se/{mask_test}.png", se_test)
        cv2.imwrite(f"images/test/mask/he/{mask_test}.png", he_test)

def create_train_val_data(image_dir,split_ratio):
    original_images = os.listdir(f"{image_dir}/images")
    original_masks = os.listdir(f"{image_dir}/masks")
    original_images = natsorted(original_images)
    original_masks = natsorted(original_masks)
    num_train = int(len(original_images) * split_ratio)

    # Randomly shuffle the images

    # Split the images into training and validation sets
    train_images = original_images[:num_train]
    train_masks = original_masks[:num_train]
    val_images = original_images[num_train:]
    val_masks = original_masks[num_train:]

    # Copy images to the training directory
    for image in train_images:
        os.makedirs("images/train/image",exist_ok= True)
        image_path = os.path.join(image_dir, 'images', image)
        shutil.copy(image_path, "images/train/image")
    # Copy images to the validation directory
    for image in val_images:
        os.makedirs("images/val/image",exist_ok= True)
        image_path = os.path.join(image_dir, 'images', image)
        shutil.copy(image_path, "images/val/image")
        
    for mask_train in train_masks:
        
        gt_train = cv2.imread(f"{image_dir}/masks/{mask_train}",cv2.IMREAD_GRAYSCALE)
        
        ma_train = np.where(gt_train == 3,255,0)
        he_train = np.where(gt_train == 2,255,0)
        se_train = np.where(gt_train == 4,255,0)
        ex_train = np.where(gt_train == 1,255,0)
        os.makedirs("images/train/mask/ma",exist_ok=True)
        os.makedirs("images/train/mask/ex",exist_ok=True)
        os.makedirs("images/train/mask/se",exist_ok=True)
        os.makedirs("images/train/mask/he",exist_ok=True)
        cv2.imwrite(f"images/train/mask/ma/{mask_train}.png", ma_train)
        cv2.imwrite(f"images/train/mask/ex/{mask_train}.png", ex_train)
        cv2.imwrite(f"images/train/mask/se/{mask_train}.png", se_train)
        cv2.imwrite(f"images/train/mask/he/{mask_train}.png", he_train)
    for mask_val in val_masks:
    
        gt_val = cv2.imread(f"{image_dir}/masks/{mask_val}",cv2.IMREAD_GRAYSCALE)
        
        ma_val = np.where(gt_val == 3,255,0)
        he_val = np.where(gt_val == 2,255,0)
        se_val = np.where(gt_val == 4,255,0)
        ex_val = np.where(gt_val == 1,255,0)
        os.makedirs("images/val/mask/ma",exist_ok=True)
        os.makedirs("images/val/mask/ex",exist_ok=True)
        os.makedirs("images/val/mask/se",exist_ok=True)
        os.makedirs("images/val/mask/he",exist_ok=True)
        cv2.imwrite(f"images/val/mask/ma/{mask_val}.png", ma_val)
        cv2.imwrite(f"images/val/mask/ex/{mask_val}.png", ex_val)
        cv2.imwrite(f"images/val/mask/se/{mask_val}.png", se_val)
        cv2.imwrite(f"images/val/mask/he/{mask_val}.png",he_val)

def initialize_crop_save(dataset_conf):
    crop_size = dataset_conf['crop_size']
    stride = dataset_conf['stride']
    lesion_list = ['ma','ex','se','he']
    

    crop_save_mask_images(dataset_conf['train_image_dir'],dataset_conf['train_image_dir_cropped'],crop_size,stride)
    crop_save_mask_images(dataset_conf['val_image_dir'],dataset_conf['val_image_dir_cropped'],crop_size,stride)
    crop_save_mask_images(dataset_conf['test_image_dir'],dataset_conf['test_image_dir_cropped'],crop_size,stride)

    for lesion in lesion_list:
        crop_save_mask_images(f"{dataset_conf['train_mask_dir']}/{lesion}",f"{dataset_conf['train_mask_dir_cropped']}/{lesion}",crop_size,stride)
        #crop_save_mask_images(f"{dataset_conf['val_mask_dir']}/{lesion}",f"{dataset_conf['val_mask_dir_cropped']}/{lesion}",crop_size,stride)
        crop_save_mask_images(f"{dataset_conf['test_mask_dir']}/{lesion}",f"{dataset_conf['test_mask_dir_cropped']}/{lesion}",crop_size,stride)



def auc_pr_folder_calculation(pred_mask_dir,test_mask_dir,stride):
    total_result = 0
    pred_items = os.listdir(pred_mask_dir)
    pred_items = natsorted(pred_items)
    
    test_items = os.listdir(test_mask_dir)
    test_items = natsorted(test_items)

    for pred,test in zip(pred_items,test_items):

        pred_mask = np.load(f"{pred_mask_dir}/{pred}")
        test_mask = cv2.imread(f"{test_mask_dir}/{test}",cv2.IMREAD_GRAYSCALE)
        new_width = test_mask.shape[:2][0] - (test_mask.shape[:2][0] % stride)
        new_height = test_mask.shape[:2][1] - (test_mask.shape[:2][1] % stride)
        test_mask = cv2.resize(test_mask, (new_width ,new_height))
        result = calculate_auc_pr(test_mask>1,pred_mask)
        print(result)
        total_result = total_result+result
    auc_pr_average_result = total_result/len(pred_items)
    return auc_pr_average_result

def plot_save_mismatches(dir1,dir2,save_dir):
    os.makedirs(f"{save_dir}/mismatched_images",exist_ok=True)
    images = os.listdir(dir1)
    masks = os.listdir(dir2)
    masks = natsorted(masks)
    images = natsorted(images)
    for image,mask in zip(images,masks):
        print(image,mask)
        image_1 = cv2.imread(f"{dir1}/{image}",cv2.IMREAD_GRAYSCALE)
        mask_1 = cv2.imread(f"{dir2}/{mask}",cv2.IMREAD_GRAYSCALE)
        # check if the image and the mask are the same size
        if image_1.shape == mask_1.shape:
            pass
        else:
            print(f"Image and mask {image} have different shapes: {image_1.shape} and {mask_1.shape}")
            continue
        # check if mask is binary
        if np.array_equal(np.unique(mask_1), np.array([0, 255])):
            pass
        else:
            print(f"Mask {mask} is not binary")
            continue
        cv2.imwrite(f"{save_dir}/mismatched_images/{image}.png",colorize_mismatches(image_1>1,mask_1>1))

def delete_black_masks(image_folder, mask_folder,threshold):
    # List all files in the image and mask folders
    image_files = natsorted(os.listdir(image_folder))
    mask_files = natsorted(os.listdir(mask_folder))

    # Pair image and mask files
    file_pairs = list(zip(image_files, mask_files))

    # Calculate the number of images to delete (80% of the total)
    num_images_to_delete = int(threshold * len(file_pairs))

    # Counter for deleted images
    deleted_images = 0

    for image_file, mask_file in file_pairs:
        image_path = os.path.join(image_folder, image_file)
        mask_path = os.path.join(mask_folder, mask_file)

        # Check if the mask image is completely black
        mask_image = Image.open(mask_path).convert("L")  # Convert to grayscale
        if not any(mask_image.getdata()):
            # Delete the corresponding image and mask only if the limit is not reached
            if deleted_images < num_images_to_delete:
                os.remove(image_path)
                os.remove(mask_path)
                deleted_images += 1
            else:
                break  # Exit the loop once the limit is reached

def copy_and_paste_folder(folder_path):
    # Get the parent directory and folder name
    parent_dir, folder_name = os.path.split(folder_path)

    # Create a new folder name with "_copy" suffix
    new_folder_name = f"{folder_name}_copy"

    # Create the path for the new folder
    new_folder_path = os.path.join(parent_dir, new_folder_name)

    try:
        # Copy the folder and its contents to the new location
        shutil.copytree(folder_path, new_folder_path)
        print(f"Folder copied successfully to {new_folder_path}")
    except Exception as e:
        print(f"Error copying folder: {e}")

        
def get_augmentations():
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(shift_limit = 0.1),
    ]
    return albu.Compose(train_transform)
# visualise images in a grid
# The images are linked so that they can be scrolled/zoomed together
# Their axes are shared so that they can be compared
from matplotlib import pyplot as plt
import os

def plot_images(imglist, figsize=(10,10), rows=1, cols=1, sharex=False, sharey=False):
    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=sharex, sharey=sharey)
    axes = axes.flatten()
    for img, ax in zip(imglist, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


""" img_num = 68
img1 = plt.imread(f'/home/adem/Downloads/Orjinal-20231207T222329Z-001/Orjinal/test/IDRiD_{img_num}.png')
img2 = plt.imread(f'/home/adem/Desktop/Thesis/IDRiD Dataset Collection/Adamlarin Format/labels/test/ma/IDRiD_{img_num}.png')
img3 = plt.imread(f'/home/adem/Desktop/Thesis/IDRiD Dataset Collection/Adamlarin Format/Preprocessed/test/IDRiD_{img_num}.png')
img4 = plt.imread(f'/home/adem/Desktop/Thesis/IDRiD Dataset Collection/Adamlarin Format/Denoised/all_4096/test/IDRiD_{img_num}.png')
img5 = plt.imread(f'/home/adem/Downloads/Yeni klasör (4)/mismatched_images/merged_IDRiD_{img_num}.png_1.png.png.png')
plot_images([img1, img2, img3, img5], figsize=(10,10), rows=2, cols=2, sharex=True, sharey=True) """

def plot_pr_curve(precision, recall, save_dir=None, figsize=(10,10)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'pr_curve.png'))
    # show non-blocking
    plt.show(block=False)
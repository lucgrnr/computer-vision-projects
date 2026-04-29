# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv (3.12.3)
#     language: python
#     name: python3
# ---

# %% [markdown]
# KUL H02A5a Computer Vision: Group Assignment 2
# ---------------------------------------------------------------
#
# <span style="color:red">**TODO**: Add your names below.</span>
#
# <span style="color:red">**TODO**: Change the notebook name by replacing the X with your actual group number.</span>
#
# Student names: <span style="color:red">name1, name2, ...</span>.
#
# In this group assignment your team will delve into some deep learning applications for computer vision. The assignment will be delivered in the same groups from *Group assignment 1* and you start from this template notebook. You can make use of the *Group assignment 2* forum/discussion board on Toledo if you have any questions.
#
# The notebook you submit for grading is the last notebook pinned as default and submitted to the competition prior to the deadline.
#
# Good luck and have fun!
#
# ---------------------------------------------------------------
# NOTES:
# * This notebook is just a template. Please keep the five main sections, but feel free to adjust further in any way you please!
# * Clearly indicate the improvements that you make! You can for instance use subsections like: *3.1. Improvement: applying loss function f instead of g*.
#

# %% [markdown]
# # Overview
# This assignment consists of *three main parts* for which we expect you to provide code and extensive documentation in the notebook, with a final discussion:
# * Image classification (Sect. 1)
# * Semantic segmentation (Sect. 2)
# * Adversarial attacks (Sect. 3)
# * Discussion (Sect. 4)

# %% [markdown]
# ## Deep learning resources
# If you did not yet explore this in *Group assignment 1 (Sect. 2)*, we recommend using the Pytorch or TensorFlow (and/or Keras) library for building deep learning models.

# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
import numpy as np
import pandas as pd
import os

# Uncomment the one you'll use
#import tensorflow as tf
import torch

from matplotlib import pyplot as plt

# %% [markdown]
# Add data paths:

# %%
DATA_PATH = ''  # for GitHub
# DATA_PATH = '/kaggle/input/competitions/kul-computer-vision-ga-2-2026/'  # for Kaggle

# %% [markdown]
# ## PASCAL VOC 2009
# For this project you will be using the [PASCAL VOC 2009](http://host.robots.ox.ac.uk/pascal/VOC/voc2009/index.html) dataset. This dataset consists of colour images of various scenes with different object classes (e.g. animal: *bird, cat, ...*; vehicle: *aeroplane, bicycle, ...*), totalling 20 classes.

# %%
# --- 1. Define Train and Test Folders ---
train_dir = os.path.join(DATA_PATH, "train")
test_dir  = os.path.join(DATA_PATH, "test")

# --- 2. Define CSV Paths ---
train_csv_path = os.path.join(train_dir, "train_set.csv")
test_csv_path  = os.path.join(test_dir, "test_set.csv")

# %%
# Loading the training data
train_df = pd.read_csv(train_csv_path, index_col="Id")
labels = train_df.columns

train_df["img"] = [np.load(os.path.join(train_dir, "img", "train_{}.npy".format(idx))) for idx, _ in train_df.iterrows()]
train_df["seg"] = [np.load(os.path.join(train_dir, "seg", "train_{}.npy".format(idx))) for idx, _ in train_df.iterrows()]
print("The training set contains {} examples.".format(len(train_df)))

# Show some examples
fig, axs = plt.subplots(2, 20, figsize=(10 * 20, 10 * 2))
for i, label in enumerate(labels):
    df = train_df.loc[train_df[label] == 1]
    axs[0, i].imshow(df.iloc[0]["img"], vmin=0, vmax=255)
    axs[0, i].set_title("\n".join(label for label in labels if df.iloc[0][label] == 1), fontsize=40)
    axs[0, i].axis("off")
    axs[1, i].imshow(df.iloc[0]["seg"], vmin=0, vmax=20)  # with the absolute color scale it will be clear that the arrays in the "seg" column are label maps (labels in [0, 20])
    axs[1, i].axis("off")
    
plt.show()

# The training dataframe contains for each image 20 columns with the ground truth classification labels and 20 column with the ground truth segmentation maps for each class
train_df.head(1)

# %%
# Loading the test data
test_df = pd.read_csv(test_csv_path, index_col="Id")
test_df["img"] = [np.load(os.path.join(test_dir, "img", "test_{}.npy".format(idx))) for idx, _ in test_df.iterrows()]
test_df["seg"] = [-1 * np.ones(img.shape[:2], dtype=np.int8) for img in test_df["img"]]
print("The test set contains {} examples.".format(len(test_df)))

# Show some examples
fig, axs = plt.subplots(2, 20, figsize=(10 * 20, 10 * 2))
for i in range(20):
    row = test_df.iloc[i]
    
    # Plot Image
    axs[0, i].imshow(row["img"], vmin=0, vmax=255)
    axs[0, i].set_title(f"Test Image {i}", fontsize=40)
    axs[0, i].axis("off")
    
    # Plot Segmentation Placeholder
    axs[1, i].imshow(row["seg"], vmin=0, vmax=20) 
    axs[1, i].axis("off")
    
plt.show()

# The test dataframe is similar to the training dataframe, but here the values are -1 --> your task is to fill in these as good as possible in Sect. 2 and Sect. 3; in Sect. 6 this dataframe is automatically transformed in the submission CSV!
test_df.head(1)


# %% [markdown]
# ## Your Kaggle submission
# Your filled test dataframe (during Sect. 2 and Sect. 3) must be converted to a submission.csv with two rows per example (one for classification and one for segmentation) and with only a single prediction column (the multi-class/label predictions running length encoded). You don't need to edit this section. Just make sure to call this function at the right position in this notebook.

# %%
def _rle_encode(img):
    """
    Kaggle requires RLE encoded predictions for computation of the Dice score (https://www.kaggle.com/lifa08/run-length-encode-and-decode)

    Parameters
    ----------
    img: np.ndarray - binary img array
    
    Returns
    -------
    rle: String - running length encoded version of img
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle = ' '.join(str(x) for x in runs)
    return rle

def generate_submission(df):
    """
    Make sure to call this function once after you completed Sect. 2 and Sect. 3! It transforms and writes your test dataframe into a submission.csv file.
    
    Parameters
    ----------
    df: pd.DataFrame - filled dataframe that needs to be converted
    
    Returns
    -------
    submission_df: pd.DataFrame - df in submission format.
    """
    df_dict = {"Id": [], "Predicted": []}
    for idx, _ in df.iterrows():
        df_dict["Id"].append(f"{idx}_classification")
        df_dict["Predicted"].append(_rle_encode(np.array(df.loc[idx, labels])))
        df_dict["Id"].append(f"{idx}_segmentation")
        df_dict["Predicted"].append(_rle_encode(np.array([df.loc[idx, "seg"] == j + 1 for j in range(len(labels))])))
    
    submission_df = pd.DataFrame(data=df_dict, dtype=str).set_index("Id")
    submission_df.to_csv("submission.csv")
    return submission_df


# %% [markdown]
# # 1. Image classification
# The goal here is simple: implement a classification model and train it to recognise all 20 classes (and/or background) using the training set and compete on the test set (by filling in the classification columns in the test dataframe).

# %%
if False:
    class RandomClassificationModel:
        """
        Random classification model: 
            - generates random labels for the inputs based on the class distribution observed during training
            - assumes an input can have multiple labels
        """
        def fit(self, X, y):
            """
            Adjusts the class ratio variable to the one observed in y. 

            Parameters
            ----------
            X: list of arrays - n x (height x width x 3)
            y: list of arrays - n x (nb_classes)

            Returns
            -------
            self
            """
            self.distribution = np.mean(y, axis=0)
            print("Setting class distribution to:\n{}".format("\n".join(f"{label}: {p}" for label, p in zip(labels, self.distribution))))
            return self
            
        def predict(self, X):
            """
            Predicts for each input a label.
            
            Parameters
            ----------
            X: list of arrays - n x (height x width x 3)
                
            Returns
            -------
            y_pred: list of arrays - n x (nb_classes)
            """
            np.random.seed(0)
            return [np.array([int(np.random.rand() < p) for p in self.distribution]) for _ in X]
        
        def __call__(self, X):
            return self.predict(X)
        
    model = RandomClassificationModel()
    model.fit(train_df["img"], train_df[labels])
    test_df.loc[:, labels] = model.predict(test_df["img"])
    test_df.head(1)

# %% [markdown]
# # 2. Semantic segmentation
# The goal here is to implement a segmentation model that labels every pixel in the image as belonging to one of the 20 classes (and/or background). Use the training set to train your model and compete on the test set (by filling in the segmentation column in the test dataframe).

# %% [markdown]
# ## 2.1 Inspect the data

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# --- 1. Look at the dataframe -------------------------------------------
print(f"Number of training examples: {len(train_df)}")
print(f"Columns (labels + img + seg): {list(train_df.columns)}")
print(f"Class label columns: {list(labels)}")
print(f"Number of class labels: {len(labels)}")
print()
print("First few rows of the label part:")
print(train_df[labels].head())

# %%
# --- 2. Inspect a single example ----------------------------------------
sample_img = train_df["img"].iloc[0]
sample_seg = train_df["seg"].iloc[0]

print("Image array:")
print(f"  shape: {sample_img.shape}")
print(f"  dtype: {sample_img.dtype}")
print(f"  min/max: {sample_img.min()} / {sample_img.max()}")
print()
print("Segmentation mask array:")
print(f"  shape: {sample_seg.shape}")
print(f"  dtype: {sample_seg.dtype}")
print(f"  min/max: {sample_seg.min()} / {sample_seg.max()}")
print(f"  unique values: {np.unique(sample_seg)}")

# --- Visualization ------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the image
axs[0].imshow(sample_img, vmin=0, vmax=255)
axs[0].set_title("Original Image", fontsize=16)
axs[0].axis("off")

# Plot the segmentation mask
# Using 'nipy_spectral' or 'jet' makes distinct integer classes easier to see
axs[1].imshow(sample_seg, vmin=0, vmax=20, cmap="nipy_spectral")
axs[1].set_title("Segmentation Mask", fontsize=16)
axs[1].axis("off")

plt.tight_layout()
plt.show()

# %%
# Checking that the size of the image and the mask is the same
mismatches = []
for idx, row in train_df.iterrows():
    img_h, img_w = row["img"].shape[:2]   # (H, W, 3) -> take H, W
    seg_h, seg_w = row["seg"].shape       # (H, W)
    if (img_h, img_w) != (seg_h, seg_w):
        mismatches.append((idx, row["img"].shape, row["seg"].shape))

print(f"Checked {len(train_df)} examples.")
print(f"Mismatches: {len(mismatches)}")
if mismatches:
    for m in mismatches[:10]:
        print(" ", m)

# %% [markdown]
# All pictures need to be the same size. Also unet expects a resolution divisible by 16. 

# %%
# Inspect image dimensions

from collections import Counter

shape_counter = Counter(arr.shape for arr in train_df["img"])

print(f"Number of unique image shapes: {len(shape_counter)}")
print(f"\nMost common shapes:")
for shape, count in shape_counter.most_common(10):
    print(f"  {shape}: {count} images ({100*count/len(train_df):.1f}%)")

# %%
# Inspect image dimensions

import numpy as np
import matplotlib.pyplot as plt

heights = np.array([arr.shape[0] for arr in train_df["img"]])
widths  = np.array([arr.shape[1] for arr in train_df["img"]])

print(f"Height — min {heights.min()}, max {heights.max()}, median {int(np.median(heights))}, mean {heights.mean():.0f}")
print(f"Width  — min {widths.min()},  max {widths.max()},  median {int(np.median(widths))},  mean {widths.mean():.0f}")

n_landscape = int((widths > heights).sum())
n_portrait  = int((heights > widths).sum())
n_square    = int((heights == widths).sum())
print(f"\nLandscape (W>H): {n_landscape}")
print(f"Portrait  (H>W): {n_portrait}")
print(f"Square    (H=W): {n_square}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(heights, bins=30); ax[0].set_title("Heights"); ax[0].set_xlabel("pixels")
ax[1].hist(widths,  bins=30); ax[1].set_title("Widths");  ax[1].set_xlabel("pixels")
plt.tight_layout(); plt.show()

plt.figure(figsize=(5, 5))
plt.scatter(widths, heights, alpha=0.3, s=10)
plt.xlabel("Width"); plt.ylabel("Height"); plt.title("Image dimensions")
plt.gca().set_aspect("equal"); plt.show()

# %%
short_sides = [min(arr.shape[0], arr.shape[1]) for arr in train_df["img"]]
print(f"Min short side: {min(short_sides)}")
print(f"Images with short side < 256: {sum(s < 256 for s in short_sides)}")
print(f"Images with short side < 320: {sum(s < 320 for s in short_sides)}")

# %%
# IMPORTANT: Look at class distribution in segmentation masks

from collections import Counter

counter = Counter()
for seg in train_df["seg"]:
    counter.update(seg.flatten().tolist())

total = sum(counter.values())

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

print(f"{'value':>5} {'class':<14} {'pixels':>14} {'%':>8}")
print("-" * 45)
for v in sorted(counter.keys()):
    name = VOC_CLASSES[v] if v < len(VOC_CLASSES) else f"<{v}>"
    pct = 100 * counter[v] / total
    print(f"{v:>5} {name:<14} {counter[v]:>14,} {pct:>7.3f}%")

# %% [markdown]
# Does every image have a mask?

# %%
n_pure_bg = sum(set(np.unique(seg).tolist()) == {0} for seg in train_df["seg"])
print(f"Images with only background pixels: {n_pure_bg}")

# %% [markdown]
# ## 2.2 Comments on the dataset
#
# We inspected the training set before designing the model to ground our
# design choices in the actual data.
#
# **Size.** 749 training examples — small. This drives several decisions
# later: model capacity, augmentation strength, expected overfitting risk.
#
# **Image format.** RGB images stored as `uint8` arrays with values in
# [0, 255]. Resolutions are variable (146 unique shapes), but the long
# side is almost always 500 pixels. The dataset is 80% landscape, 19%
# portrait, 1% square.
#
# **Mask format.** Single-channel 2D arrays (H × W) with the same spatial
# size as the corresponding image, dtype `uint8`. Each pixel value is a
# class index in [0, 20]: 0 = background, 1–20 = the 20 PASCAL VOC
# foreground classes. Importantly, **no void/ignore index** is present
# in this packaged version of the dataset, so the loss function does not
# need an `ignore_index`.
#
# **Crop size.** We will use 256×256 random crops during training:
# - divisible by 16 (required for U-Net's four downsampling steps),
# - fits inside almost every image (only 16 / 749 = 2.1% have a short
#   side smaller than 256, handled with zero-padding),
# - gives a reasonable batch size on a single GPU.
#
# **Class imbalance.** Background dominates at 77.7% of all pixels.
# Among foreground classes, person is by far the most common at 4.0%;
# all other classes lie between 0.3% and 1.7%. The imbalance ratio
# between background and the rarest class (bicycle) is roughly 250:1.
#
# This has two implications:
# 1. Pixel accuracy is meaningless as a metric (predicting all-background
#    gives 77.7%). The competition uses Dice score, which we will mirror
#    internally with per-class Dice and mean Dice.
# 2. We expect rare and structurally thin classes (bicycle, pottedplant,
#    aeroplane, cow) to underperform. The writeup will revisit this
#    prediction after training.
#
# Every image contains at least one foreground class, so no filtering
# is needed.

# %% [markdown]
# ## 2.3 Train / validation split

# %%
import albumentations as A
print(A.__version__)
import numpy as np

SEED = 42
rng = np.random.default_rng(SEED)

n = len(train_df)
indices = rng.permutation(n)
n_val = int(0.2 * n)
val_indices   = sorted(indices[:n_val].tolist())
train_indices = sorted(indices[n_val:].tolist())

print(f"Train: {len(train_indices)} examples")
print(f"Val:   {len(val_indices)} examples")
print(f"Overlap (should be 0): {len(set(train_indices) & set(val_indices))}")


# %%
def classes_present(seg_arrays):
    """Return set of class indices that appear at least once across the given masks."""
    seen = set()
    for seg in seg_arrays:
        seen.update(np.unique(seg).tolist())
    return seen

train_classes = classes_present(train_df["seg"].iloc[train_indices])
val_classes   = classes_present(train_df["seg"].iloc[val_indices])

print(f"Classes in train: {sorted(train_classes)}")
print(f"Classes in val:   {sorted(val_classes)}")
print(f"Missing from val: {sorted(train_classes - val_classes)}")

# %% [markdown]
# ## 2.4 Data augmentation

# %%
import albumentations as A
from albumentations.pytorch import ToTensorV2

CROP_SIZE = 256

train_transform = A.Compose([
    A.PadIfNeeded(min_height=CROP_SIZE, min_width=CROP_SIZE,
                  border_mode=0, fill=0, fill_mask=0),  # zero-pad small images
    A.RandomScale(scale_limit=(-0.5, 1.0), p=1.0),       # scale in [0.5, 2.0]
    A.PadIfNeeded(min_height=CROP_SIZE, min_width=CROP_SIZE,
                  border_mode=0, fill=0, fill_mask=0),  # pad again after scaling
    A.RandomCrop(height=CROP_SIZE, width=CROP_SIZE),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.PadIfNeeded(min_height=CROP_SIZE, min_width=CROP_SIZE,
                  border_mode=0, fill=0, fill_mask=0),
    A.CenterCrop(height=CROP_SIZE, width=CROP_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# %%
import torch
from torch.utils.data import Dataset

class VOCSegDataset(Dataset):
    def __init__(self, df, indices, transform):
        """
        df: the train_df with "img" and "seg" columns
        indices: which rows of df to use (train or val)
        transform: an Albumentations Compose pipeline
        """
        self.df = df
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row_idx = self.indices[i]
        img = self.df["img"].iloc[row_idx]   # (H, W, 3) uint8
        seg = self.df["seg"].iloc[row_idx]   # (H, W)    uint8

        out = self.transform(image=img, mask=seg)
        image_tensor = out["image"]                # float32, (3, H, W), normalized
        mask_tensor  = out["mask"].long()          # int64,   (H, W), class indices

        return image_tensor, mask_tensor


# %% [markdown]
# ## 2.5 From-scratch U-Net (segmentation)                                                                                     
#                                                                                                                                                         
#   We train a half-width U-Net (base channels 32, 5 levels: 32→64→128→256→512, ~7.8M params) **from scratch** on the 749-image training set. The         
#   objective is a sum of pixel-wise cross-entropy and a soft Dice loss over the 20 foreground classes — CE provides stable gradients early; Dice corrects
#    for the heavy background bias (background ≈ 78% of pixels).
#
#   Augmentation (Albumentations) handles the variable input resolutions: short side is normalised to ≥288, then a random scale/crop produces fixed
#   256×256 patches; horizontal flip and colour jitter expand the limited 749-image distribution. Validation runs on full images padded to 512×512.
#
#   Optimisation: AdamW (lr=1e-3, wd=1e-4) with cosine LR decay over 100 epochs, AMP mixed-precision. We track mean foreground Dice on a fixed 80/20
#   random split and checkpoint the best model to `best_unet.pt`

# %%
if False:
    class RandomSegmentationModel:
        """
        Random segmentation model: 
            - generates random label maps for the inputs based on the class distributions observed during training
            - every pixel in an input can only have one label
        """
        def fit(self, X, Y):
            """
            Adjusts the class ratio variable to the one observed in Y. 

            Parameters
            ----------
            X: list of arrays - n x (height x width x 3)
            Y: list of arrays - n x (height x width)

            Returns
            -------
            self
            """
            self.distribution = np.mean([[np.sum(Y_ == i) / Y_.size for i in range(len(labels) + 1)] for Y_ in Y], axis=0)
            print("Setting class distribution to:\nbackground: {}\n{}".format(self.distribution[0], "\n".join(f"{label}: {p}" for label, p in zip(labels, self.distribution[1:]))))
            return self
            
        def predict(self, X):
            """
            Predicts for each input a label map.
            
            Parameters
            ----------
            X: list of arrays - n x (height x width x 3)
                
            Returns
            -------
            Y_pred: list of arrays - n x (height x width)
            """
            np.random.seed(0)
            return [np.random.choice(np.arange(len(labels) + 1), size=X_.shape[:2], p=self.distribution) for X_ in X]
        
        def __call__(self, X):
            return self.predict(X)
        
    model = RandomSegmentationModel()
    model.fit(train_df["img"], train_df["seg"])
    test_df.loc[:, "seg"] = model.predict(test_df["img"])
    test_df.head(1)


# %% [markdown]
# ## 2.6 Transfer learning models for segmentation 

# %%
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

def get_transforms(model_name="mask2former"):
    """
    Returns optimal train and val transforms based on the target model.
    """
    # 1. Set optimal size for 8GB VRAM
    if model_name.lower() in ["deeplabv3", "segformer"]:
        CROP_SIZE = 512
    elif model_name.lower() == "mask2former":
        CROP_SIZE = 384
    else:
        CROP_SIZE = 512 

    # 2. PyTorch Standard Ignore Index
    IGNORE_INDEX = 255 
    
    # 3. ImageNet Mean Color for Padding (RGB)
    MEAN_COLOR = (124, 116, 104) 

    train_transform = A.Compose([
        # Optional: Prevent massive images from creating tiny crops
        A.LongestMaxSize(max_size=int(CROP_SIZE * 1.5), p=1.0),
        
        # Scale augmentation: (0.5x to 2.0x)
        # Masks automatically use nearest-neighbor interpolation to avoid blending class IDs
        A.RandomScale(scale_limit=(-0.5, 1.0), p=1.0),
        
        # Pad if the scaled image is smaller than our crop size
        A.PadIfNeeded(
            min_height=CROP_SIZE, 
            min_width=CROP_SIZE,
            border_mode=cv2.BORDER_CONSTANT, 
            fill=MEAN_COLOR,        # Pad images with neutral gray
            fill_mask=IGNORE_INDEX  # Pad masks with 255 (Loss ignores this)
        ),
        
        # Extract the exact tensor size
        A.RandomCrop(height=CROP_SIZE, width=CROP_SIZE),
        
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        # Safely fit the whole image inside the crop window without stretching
        A.LongestMaxSize(max_size=CROP_SIZE, p=1.0),
        
        # Pad the remaining space to make it exactly the CROP_SIZE
        A.PadIfNeeded(
            min_height=CROP_SIZE, 
            min_width=CROP_SIZE,
            border_mode=cv2.BORDER_CONSTANT, 
            fill=MEAN_COLOR,
            fill_mask=IGNORE_INDEX
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    print(f"Transforms built for {model_name.upper()} | Target Size: {CROP_SIZE}x{CROP_SIZE}")
    return train_transform, val_transform



# %% [markdown]
# ## Setup and Reusable Pipeline Functions
# We define a unified pipeline so any model (DeepLabV3+, SegFormer, etc.) 
# can be trained and evaluated using the exact same code.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Mask2FormerForUniversalSegmentation


from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from tqdm import tqdm
import gc

# 1. Setup Device & Common Variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# 2. Unified Loss Function (Fixed for 8GB Blackwell / CUDA 13.2)
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # CrossEntropy needs to ignore the padding index
        self.ce = nn.CrossEntropyLoss(ignore_index=255)
        
        # smp.DiceLoss also supports ignore_index
        # 'from_logits=True' is usually best if your model doesn't have a Softmax at the end
        self.dice = smp.losses.DiceLoss(
            mode="multiclass", 
            classes=21, 
            ignore_index=255, 
            from_logits=True 
        )
        
    def forward(self, y_pred, y_true):
        # Important: ensure y_true is Long for CE and lacks the channel dim
        if y_true.dim() == 4:
            y_true = y_true.squeeze(1)
            
        return self.ce(y_pred, y_true) + self.dice(y_pred, y_true)

criterion = BCEDiceLoss()

# 3. SegFormer Wrapper
# This forces SegFormer to output standard (B, C, H, W) tensors like SMP models
class SegFormerWrapper(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, x):
        # Extract logits and interpolate to input image size
        outputs = self.model(pixel_values=x).logits
        outputs = F.interpolate(outputs, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return outputs

class Mask2FormerWrapper(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, x):
        # 1. Forward pass through the HF model
        outputs = self.model(pixel_values=x)
        
        # 2. Extract Class Queries: (Batch, Num_Queries, Num_Classes)
        class_logits = outputs.class_queries_logits
        
        # 3. Extract Mask Queries: (Batch, Num_Queries, Height/4, Width/4)
        mask_logits = outputs.masks_queries_logits
        
        # 4. Upsample masks back to the original image resolution (e.g., 256x256)
        mask_logits = F.interpolate(
            mask_logits, 
            size=x.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        
        # 5. Project queries to spatial logits: (Batch, Classes, Height, Width)
        # We multiply the class probabilities by the spatial masks using Einstein Summation
        semantic_logits = torch.einsum("bqc,bqhw->bchw", class_logits, mask_logits)
        
        return semantic_logits

# --- REUSABLE PIPELINE FUNCTIONS ---

def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs, save_name, target_effective_batch=16):
    """Handles the training loop, mixed precision, automatic gradient accumulation, and saving."""
    
    # --- AUTOMATIC ACCUMULATION LOGIC ---
    physical_batch = train_loader.batch_size
    # Calculate how many steps needed to reach the target batch size (default 16)
    accumulation_steps = max(1, target_effective_batch // physical_batch)
    actual_effective_batch = physical_batch * accumulation_steps
    
    scaler = torch.amp.GradScaler('cuda')
    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    print(f"--- Starting Training for {save_name} ---")
    print(f"VRAM Physical Batch: {physical_batch} | Accumulation Steps: {accumulation_steps} | Math Effective Batch: {actual_effective_batch}")
    print("-" * 50)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        optimizer.zero_grad() # Move this outside the batch loop for accumulation
        
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss = loss / accumulation_steps # Scale loss for accumulation
                        
            scaler.scale(loss).backward()
            
            # Step only after 'accumulation_steps' batches
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Restore the true loss value for accurate logging
            train_loss += (loss.item() * accumulation_steps) * images.size(0)
            
        train_loss /= len(train_loader.dataset)
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                
        val_loss /= len(val_loader.dataset)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{save_name}.pt")
            print("  -> Best model saved!")
            
    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', color='blue')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', color='red', linestyle='--')
    plt.title(f'Loss Curve: {save_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()


def evaluate_metrics(model, val_loader, save_name, num_classes=21):
    """Calculates and prints detailed per-class IoU and Dice scores, ignoring 255 padding."""
    
    model.load_state_dict(torch.load(f"{save_name}.pt", weights_only=True))
    model.eval()
    
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Evaluating {save_name}"):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # Flatten and ensure long dtype
            mask_flat = masks.view(-1).long().cpu()
            pred_flat = preds.view(-1).long().cpu()
            
            # --- THE FIX: Mask out 255 ---
            valid_indices = (mask_flat != 255)
            mask_valid = mask_flat[valid_indices]
            pred_valid = pred_flat[valid_indices]
            
            conf_matrix += torch.bincount(
                num_classes * mask_valid + pred_valid, 
                minlength=num_classes**2
            ).reshape(num_classes, num_classes)

    tp = torch.diag(conf_matrix)
    fp = conf_matrix.sum(dim=0) - tp
    fn = conf_matrix.sum(dim=1) - tp
    
    iou = (tp.float() / (tp + fp + fn + 1e-12)).numpy()
    dice = ((2. * tp.float()) / (2. * tp + fp + fn + 1e-12)).numpy()
    
    print(f"\n--- Evaluation Results: {save_name} ---")
    print(f"{'Class ID':<10} {'Class Name':<15} {'IoU':<10} {'Dice':<10}")
    print("-" * 47)
    
    for i, name in enumerate(VOC_CLASSES):
        print(f"{i:<10} {name:<15} {iou[i]:.4f}     {dice[i]:.4f}")

    mean_iou = np.mean(iou)
    mean_dice_all = np.mean(dice)
    mean_dice_foreground = np.mean(dice[1:]) 

    print("-" * 47)
    print(f"Mean IoU (All):         {mean_iou:.4f}")
    print(f"Mean Dice (All):        {mean_dice_all:.4f}")
    print(f"Mean Foreground Dice:   {mean_dice_foreground:.4f}  <-- Key Metric\n")
    
    return mean_iou, mean_dice_foreground


def visualize_predictions(model, val_loader, save_name, num_examples=4):
    """Plots Input, Ground Truth, and Predictions with a dynamic legend."""
    model.load_state_dict(torch.load(f"{save_name}.pt", weights_only=True))
    model.eval()
    
    images, masks = next(iter(val_loader))
    images = images.to(device)
    
    with torch.no_grad():
        preds = torch.argmax(model(images), dim=1).cpu().numpy()
        
    images = images.cpu().numpy()
    masks_np = masks.cpu().numpy()
    
    # --- THE FIX: Map 255 back to 0 for plotting only ---
    masks_np[masks_np == 255] = 0 
    
    cmap = plt.get_cmap("nipy_spectral")
    
    fig, axs = plt.subplots(num_examples, 3, figsize=(15, 5 * num_examples))
    for i in range(num_examples):
        img = np.clip(np.transpose(images[i] * np.array([0.229, 0.224, 0.225]).reshape(3,1,1) + 
                                   np.array([0.485, 0.456, 0.406]).reshape(3,1,1), (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
        
        axs[i, 0].imshow(img); axs[i, 0].set_title("Input Image"); axs[i, 0].axis("off")
        axs[i, 1].imshow(masks_np[i], vmin=0, vmax=20, cmap="nipy_spectral"); axs[i, 1].set_title("Ground Truth"); axs[i, 1].axis("off")
        axs[i, 2].imshow(preds[i], vmin=0, vmax=20, cmap="nipy_spectral"); axs[i, 2].set_title(f"Prediction ({save_name})"); axs[i, 2].axis("off")
        
        legend_patches = [mpatches.Patch(color=cmap(c / 20.0), label=VOC_CLASSES[c]) for c in np.unique(preds[i])]
        axs[i, 2].legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
    plt.tight_layout()
    plt.show()

def generate_test_predictions(model, test_df, val_transform, save_name):
    """Runs inference on the test set and assigns it back to the dataframe."""
    model.load_state_dict(torch.load(f"{save_name}.pt", weights_only=True))
    model.eval()
    test_preds = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_df)), desc=f"Testing {save_name}"):
            tensor_img = val_transform(image=test_df.iloc[idx]["img"])["image"].unsqueeze(0).to(device)
            preds = torch.argmax(torch.softmax(model(tensor_img), dim=1), dim=1).squeeze(0).cpu().numpy().astype(np.int8)
            test_preds.append(preds)
            
    return test_preds

# %%
# --- DATALOADERS SETUP ---

# Assuming VOCSegDataset, train_df, val_df, train_transform, and val_transform 
# are already defined from your earlier data processing steps.

BATCH_SIZE = 4  # Adjust based on your VRAM

# Create Datasets
target_model = "deeplabv3" # change to "segformer" or "deeplabv3" when testing them
train_transform, val_transform = get_transforms(target_model)


train_dataset = VOCSegDataset(train_df, train_indices, train_transform)
val_dataset = VOCSegDataset(train_df, val_indices, val_transform)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=4,     # Adjust based on your CPU cores
    pin_memory=True    # Speeds up data transfer to the GPU
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True
)

print(f"DataLoaders ready: {len(train_loader)} training batches, {len(val_loader)} validation batches.")

# %% [markdown]
# ## Execute Pipeline: DeepLabV3+
# Using swapable backbones: mobilenet_v2, resnet34 or efficientnet-b3
# %%
torch.cuda.empty_cache(); gc.collect()

# 1. Initialize
model_deeplab = smp.DeepLabV3Plus(encoder_name="tu-tf_efficientnetv2_s", encoder_weights="imagenet", in_channels=3, classes=21).to(device)
optimizer_dl = torch.optim.AdamW(model_deeplab.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_dl = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_dl, T_max=40)

# 2. Run Pipeline
train_model(model_deeplab, train_loader, val_loader, optimizer_dl, scheduler_dl, epochs=40, save_name="best_deeplab")

# Store predictions to a specific column if needed
# test_df["seg_deeplab"] = generate_test_predictions(model_deeplab, test_df, val_transform, save_name="best_deeplab")


# %%
evaluate_metrics(model_deeplab, val_loader, save_name="best_deeplab")

# %%
#visualize_predictions(model_deeplab, val_loader, save_name="best_deeplab")

# %% [markdown]
# ## Execute Pipeline: SegFormer-B0
# Using MiT-b0 Backbone
# %%
torch.cuda.empty_cache(); gc.collect()

# %%
from transformers import SegformerForSemanticSegmentation
torch.cuda.empty_cache(); gc.collect()

# 1. Initialize using the Wrapper
hf_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b2", num_labels=21) # mit-b0 or mit-b2
model_segformer = SegFormerWrapper(hf_model).to(device)
# BO is fast, B2 is much slower

# SegFormer usually needs a slightly lower learning rate than CNNs
optimizer_sf = torch.optim.AdamW(model_segformer.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler_sf = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_sf, T_max=20)

# 2. Run Pipeline
train_model(model_segformer, train_loader, val_loader, optimizer_sf, scheduler_sf, epochs=20, save_name="best_segformer")

# %%
evaluate_metrics(model_segformer, val_loader, save_name="best_segformer")

# %%
#visualize_predictions(model_segformer, val_loader, save_name="best_segformer")

# %% [markdown]
# ## Mask2Former
#
# Test model with high resolution to improve performance

# %%

import torch
import gc
from transformers import (
    Mask2FormerConfig, 
    Mask2FormerForUniversalSegmentation, 
    Mask2FormerImageProcessor
)
import numpy as np

torch.cuda.empty_cache(); gc.collect()

# --- DATALOADERS SETUP ---

# Assuming VOCSegDataset, train_df, val_df, train_transform, and val_transform 
# are already defined from your earlier data processing steps.

BATCH_SIZE = 4  # Adjust based on your VRAM

# Create Datasets
target_model = "mask2former" # change to "segformer" or "deeplabv3" when testing them
train_transform, val_transform = get_transforms(target_model)

train_dataset = VOCSegDataset(train_df, train_indices, train_transform)
val_dataset = VOCSegDataset(train_df, val_indices, val_transform)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=4,     # Adjust based on your CPU cores
    pin_memory=True    # Speeds up data transfer to the GPU
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True
)

print(f"DataLoaders ready: {len(train_loader)} training batches, {len(val_loader)} validation batches.")




# --- 1. THE TARGET CONVERTER (The Secret Sauce) ---
def prepare_mask2former_targets(masks_tensor, ignore_index=255):
    """
    Converts a standard (B, H, W) semantic mask into the list of binary masks 
    and class labels that Mask2Former's native loss function requires.
    Runs entirely on the GPU for maximum speed.
    """
    batch_size = masks_tensor.shape[0]
    mask_labels = []
    class_labels = []
    
    for b in range(batch_size):
        mask = masks_tensor[b]
        
        # Find all unique classes in this specific image, ignoring the padding
        classes = torch.unique(mask)
        classes = classes[classes != ignore_index]
        
        b_masks = []
        b_classes = []
        
        for c in classes:
            b_masks.append(mask == c)
            b_classes.append(c)
            
        # Fallback: If an image is literally 100% padding/ignore (rare but possible during crops)
        # Mask2Former requires at least one target to not crash.
        if len(b_masks) == 0:
            b_masks.append(torch.zeros_like(mask, dtype=torch.bool))
            b_classes.append(torch.tensor(0, device=mask.device)) # Assumes 0 is background
            
        # HF expects float masks and long labels
        mask_labels.append(torch.stack(b_masks).to(torch.float32))
        class_labels.append(torch.stack(b_classes).to(torch.int64))
        
    return mask_labels, class_labels

# --- 2. NATIVE TRAINING LOOP ---
def train_mask2former_native(model, train_loader, val_loader, optimizer, scheduler, epochs, save_name, target_effective_batch=16):
    physical_batch = train_loader.batch_size
    accumulation_steps = max(1, target_effective_batch // physical_batch)
    
    scaler = torch.amp.GradScaler('cuda')
    best_val_loss = float("inf")
    
    print(f"--- Starting Native Mask2Former Training ---")
    print(f"Physical Batch: {physical_batch} | Accumulation: {accumulation_steps}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            
            # Convert targets to Hugging Face format dynamically
            mask_labels, class_labels = prepare_mask2former_targets(masks, ignore_index=255)
            
            with torch.amp.autocast('cuda'):
                # 💥 We pass the targets directly to the model. No Wrapper. No BCEDiceLoss.
                outputs = model(
                    pixel_values=images,
                    mask_labels=mask_labels,
                    class_labels=class_labels
                )
                
                # The model calculates its own Bipartite matching loss
                loss = outputs.loss / accumulation_steps
                        
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += (loss.item() * accumulation_steps) * images.size(0)
            
        train_loss /= len(train_loader.dataset)
        scheduler.step()
        
        # --- Native Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                mask_labels, class_labels = prepare_mask2former_targets(masks, ignore_index=255)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(pixel_values=images, mask_labels=mask_labels, class_labels=class_labels)
                    loss = outputs.loss
                val_loss += loss.item() * images.size(0)
                
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{save_name}.pt")
            print("  -> Best model saved!")


# --- 4. EXECUTION ---
model_id = "facebook/mask2former-swin-tiny-ade-semantic"
config = Mask2FormerConfig.from_pretrained(model_id)
config.num_queries = 20  
config.num_labels = 21

# Load RAW model. No wrapper!
hf_m2f = Mask2FormerForUniversalSegmentation.from_pretrained(
    model_id,
    config=config,
    ignore_mismatched_sizes=True 
).to(device)

base_lr = 1e-4
optimizer_m2f = torch.optim.AdamW([
    {'params': hf_m2f.model.pixel_level_module.encoder.parameters(), 'lr': base_lr * 0.1},
    {'params': hf_m2f.model.pixel_level_module.decoder.parameters(), 'lr': base_lr},
    {'params': hf_m2f.model.transformer_module.parameters(), 'lr': base_lr},
    {'params': hf_m2f.class_predictor.parameters(), 'lr': base_lr},
], weight_decay=1e-4)

scheduler_m2f = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_m2f, T_max=20)

train_mask2former_native(hf_m2f, train_loader, val_loader, optimizer_m2f, scheduler_m2f, epochs=20, save_name="native_mask2former")


# %%
# --- 3. NATIVE EVALUATION LOOP (Updated with Per-Class Details) ---
def evaluate_mask2former_native(model, val_loader, save_name, num_classes=21):
    model.load_state_dict(torch.load(f"{save_name}.pt", weights_only=True))
    model.eval()
    
    # We use the processor STRICTLY for converting queries back into a semantic map.
    # We disable resize/normalize because Albumentations already did that.
    processor = Mask2FormerImageProcessor(ignore_index=255, do_resize=False, do_rescale=False, do_normalize=False)
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Evaluating Native {save_name}"):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(pixel_values=images)
            
            # Tell the processor the size we want the output masks to be
            target_sizes = [masks.shape[1:] for _ in range(images.shape[0])]
            
            # The processor magically turns queries into (H, W) class predictions!
            predicted_masks_list = processor.post_process_semantic_segmentation(
                outputs, target_sizes=target_sizes
            )
            
            # Stack the list back into a standard (B, H, W) tensor
            preds = torch.stack(predicted_masks_list).to(device)
            
            mask_flat = masks.view(-1).long().cpu()
            pred_flat = preds.view(-1).long().cpu()
            
            valid_indices = (mask_flat != 255)
            mask_valid = mask_flat[valid_indices]
            pred_valid = pred_flat[valid_indices]
            
            conf_matrix += torch.bincount(
                num_classes * mask_valid + pred_valid, 
                minlength=num_classes**2
            ).reshape(num_classes, num_classes)

    # Calculate metrics from confusion matrix
    tp = torch.diag(conf_matrix)
    fp = conf_matrix.sum(dim=0) - tp
    fn = conf_matrix.sum(dim=1) - tp
    
    # Calculate IoU and Dice arrays
    iou = (tp.float() / (tp + fp + fn + 1e-12)).numpy()
    dice = ((2. * tp.float()) / (2. * tp + fp + fn + 1e-12)).numpy()
    
    # --- Print Detailed Results Table ---
    print(f"\n--- Evaluation Results: {save_name} ---")
    print(f"{'Class ID':<10} {'Class Name':<15} {'IoU':<10} {'Dice':<10}")
    print("-" * 47)
    
    # Assuming VOC_CLASSES is defined globally in your notebook
    for i, name in enumerate(VOC_CLASSES):
        print(f"{i:<10} {name:<15} {iou[i]:.4f}     {dice[i]:.4f}")

    # Summary Metrics
    mean_iou = np.mean(iou)
    mean_dice_all = np.mean(dice)
    mean_dice_foreground = np.mean(dice[1:]) # Exclude background (index 0)

    print("-" * 47)
    print(f"Mean IoU (All):         {mean_iou:.4f}")
    print(f"Mean Dice (All):        {mean_dice_all:.4f}")
    print(f"Mean Foreground Dice:   {mean_dice_foreground:.4f}  <-- Key Metric\n")
    
    return mean_iou, mean_dice_foreground

# Call the function
evaluate_mask2former_native(hf_m2f, val_loader, save_name="native_mask2former")

# %% [markdown]
# ## Submit to competition
# You don't need to edit this section. Just use it at the right position in the notebook. See the definition of this function in Sect. 1.3 for more details.

# %%
#generate_submission(test_df)

# %% [markdown]
# # 3. Adversarial attack
# For this part, your goal is to fool your classification and/or segmentation model, using an *adversarial attack*. More specifically, the goal is build a network to perturb test images in a way that (i) they look unperturbed to humans; but (ii) the original model classifies/segments these images in line with the perturbations.

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # 4. Discussion
# Finally, take some time to reflect on what you have learned during this assignment. Why did you make certain design choices? How are the models learning and performing? Reflect and produce an overall discussion with links to the lectures and "real world" computer vision.

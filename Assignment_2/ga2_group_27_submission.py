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
# **Team 27** | Fosteris Emmanouil · Paschalidis Marios · Greiner Luc · Reyes Salazar Jose · Digenis Stefanos
#
# In this group assignment your team will delve into some deep learning applications for computer vision. The assignment will be delivered in the same groups from *Group assignment 1* and you start from this template notebook. You can make use of the *Group assignment 2* forum/discussion board on Toledo if you have any questions.
#
# The notebook you submit for grading is the last notebook pinned as default and submitted to the competition prior to the deadline.
#
# Good luck and have fun!

# %% [markdown]
# # Overview
# This assignment consists of *three main parts* for which we expect you to provide code and extensive documentation in the notebook, with a final discussion:
# * Image classification (Sect. 1)
# * Semantic segmentation (Sect. 2)
# * Adversarial attacks (Sect. 3)
# * Discussion (Sect. 4)

# %% [markdown]
# ## Instructions: file path setup and package installation

# %%
# Directory
# Verify that the DATA_PATH variable is correctly set to the location of your dataset.
DATA_PATH = '' # for Github
#DATA_PATH = '/kaggle/input/kul-computer-vision-ga-2-2026' # for Kaggle

TRAIN = True # Set to False to load pre-trained weights instead of training
SMOKE = False # Set to True for quick plumbing checks

VRAM_GB = 8 # Set your available GPU VRAM in GB here
NUM_WORKERS = 6 # set equal no cores for Linux, 0 recommended for Windows


# Optimal values for low VRAM usages models
SCNN_BATCH_SIZE = 128
CLIP_BATCH_SIZE = 32

# Dynamic btach values based on VRAM
if VRAM_GB <= 8:
    # U-Net Effective 16
    UNET_BATCH_SIZE = 16 
    UNET_ACCUM_STEPS = 1
    # M2F Effective 16
    M2F_BATCH_SIZE = 4
    M2F_ACCUM_STEPS = 4
else: # 16GB+
    # U-Net Effective 32
    UNET_BATCH_SIZE = 32 
    UNET_ACCUM_STEPS = 1
    # M2F Effective 32
    M2F_BATCH_SIZE = 8
    M2F_ACCUM_STEPS = 2

# %% [markdown]
# ## Imports and deep learning resources

# %%
import os
import gc
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from tqdm import tqdm
from sklearn.metrics import average_precision_score

# We use torch, not tensorflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp
import open_clip
from transformers import (
    Mask2FormerConfig, 
    Mask2FormerForUniversalSegmentation, 
    Mask2FormerImageProcessor
)
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# Setup Device & Common Variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False # speedup for CNNs like U-Net.
        torch.backends.cudnn.benchmark = True # speedup for CNNs like U-Net.
    os.environ["PYTHONHASHSEED"] = str(seed)

# %% [markdown]
# ## Utility functions
#
# @TAs: There is no need to review in detail these functions. The more interesting pipelines are further down.

# %%

@torch.no_grad()
def visualize_classification(model, loader, val_df, num_examples=6):
    model.eval()
    # 1. Grab exactly one batch from the loader
    images, targets = next(iter(loader))
    
    # 2. Get predictions (using 0.5 threshold)
    logits = model(images.to(device))
    preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)
    targets_np = targets.numpy().astype(int)

    # 3. Plotting logic
    cols = 3
    rows = (num_examples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.5 * rows))
    axes = axes.flatten()

    for i in range(num_examples):
        # We take images from the batch, un-normalizing for display
        img = np.clip(np.transpose(images[i].numpy() * 
                     np.array([0.229, 0.224, 0.225]).reshape(3,1,1) + 
                     np.array([0.485, 0.456, 0.406]).reshape(3,1,1), (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
        
        # Get labels using global 'labels' list
        true_lbl = [labels[c] for c in range(len(labels)) if targets_np[i, c] == 1]
        pred_lbl = [labels[c] for c in range(len(labels)) if preds[i, c] == 1]

        axes[i].imshow(img)
        axes[i].set_title(f"True: {', '.join(true_lbl)}\nPred: {', '.join(pred_lbl)}", fontsize=9)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

def visualize_segmentation(model, val_loader, save_name, voc_classes, device='cuda', num_examples=5):
    """
    Standard visualization logic for semantic segmentation comparing Input, Ground Truth, and Prediction.
    """
    # 1. Load weights and set model to eval
    model.load_state_dict(torch.load(f"{save_name}.pt", weights_only=True, map_location=device))
    model.eval()
    
    # 2. Extract a single batch
    images, masks = next(iter(val_loader))
    
    # Ensure we don't try to plot more images than exist in the batch
    n = min(num_examples, images.size(0))
    
    # 3. Get predictions
    with torch.no_grad():
        outputs = model(images.to(device))
        if hasattr(outputs, 'class_queries_logits'):
            processor = Mask2FormerImageProcessor(ignore_index=255, do_resize=False, do_rescale=False, do_normalize=False)
            target_sizes = [masks[i].shape for i in range(n)]
            preds_list = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
            preds = torch.stack(preds_list).cpu().numpy()
        else:
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
    images = images.numpy()
    masks_np = masks.numpy()
    
    # Force padding/ignore index to background (0) for cleaner rendering
    preds[masks_np == 255] = 0 
    masks_np[masks_np == 255] = 0 
    
    # 4. Plotting setup
    cmap = plt.get_cmap("nipy_spectral")
    fig, axs = plt.subplots(n, 3, figsize=(15, 5 * n))
    
    # Handle matplotlib indexing when n=1 (1D array instead of 2D array)
    if n == 1:
        axs = np.expand_dims(axs, axis=0)
        
    for i in range(n):
        # Un-normalize the image back to standard RGB
        img = np.clip(np.transpose(images[i] * np.array([0.229, 0.224, 0.225]).reshape(3,1,1) + 
                                   np.array([0.485, 0.456, 0.406]).reshape(3,1,1), (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
        
        axs[i, 0].imshow(img)
        axs[i, 0].set_title("Input")
        axs[i, 0].axis("off")
        
        axs[i, 1].imshow(masks_np[i], vmin=0, vmax=20, cmap="nipy_spectral")
        axs[i, 1].set_title("Ground Truth")
        axs[i, 1].axis("off")
        
        axs[i, 2].imshow(preds[i], vmin=0, vmax=20, cmap="nipy_spectral")
        axs[i, 2].set_title(f"Prediction ({save_name})")
        axs[i, 2].axis("off")
        
        # Build dynamic legend based on classes present in the prediction
        legend_patches = [mpatches.Patch(color=cmap(c / 20.0), label=voc_classes[c]) for c in np.unique(preds[i])]
        axs[i, 2].legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
    plt.tight_layout()
    plt.show()

# Create a function to clean-up VRAM after each model is run
def clean_up_vram():
    """Cleans up VRAM by collecting garbage and emptying the CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# %% [markdown]
# # 0. Data preprocessing

# %% [markdown]
# ## 0.1 Load data: PASCAL VOC 2009

# %%
# Add class names
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# %%
# Loading the TRAINING data
train_dir = os.path.join(DATA_PATH, "train")
train_df = pd.read_csv(os.path.join(train_dir, "train_set.csv"), index_col="Id")
labels = train_df.columns

train_df["img"] = [np.load(os.path.join(train_dir, "img", f"train_{idx}.npy")) for idx, _ in train_df.iterrows()]
train_df["seg"] = [np.load(os.path.join(train_dir, "seg", f"train_{idx}.npy")) for idx, _ in train_df.iterrows()]
print("The training set contains {} examples.".format(len(train_df)))

# Show some examples
fig, axs = plt.subplots(2, 20, figsize=(10 * 20, 10 * 2))
for i, label in enumerate(labels):
    df = train_df.loc[train_df[label] == 1]
    if not df.empty:
        axs[0, i].imshow(df.iloc[0]["img"], vmin=0, vmax=255)
        axs[0, i].set_title("\n".join(lbl for lbl in labels if df.iloc[0][lbl] == 1), fontsize=40)
        axs[0, i].axis("off")
        axs[1, i].imshow(df.iloc[0]["seg"], vmin=0, vmax=20) 
        axs[1, i].axis("off")
    else:
        axs[0, i].set_title(f"No {label}", fontsize=40)
    
plt.show()

# %%
# Loading the TEST data
test_dir = os.path.join(DATA_PATH, "test")
test_df = pd.read_csv(os.path.join(test_dir, "test_set.csv"), index_col="Id")
test_df["img"] = [np.load(os.path.join(test_dir, "img", f"test_{idx}.npy")) for idx, _ in test_df.iterrows()]
test_df["seg"] = [-1 * np.ones(img.shape[:2], dtype=np.int8) for img in test_df["img"]]
print("The test set contains {} examples.".format(len(test_df)))

# Show some examples
fig, axs = plt.subplots(2, 20, figsize=(10 * 20, 10 * 2))
for i in range(min(20, len(test_df))):
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
# ## 0.2 Exploratory Data Analysis

# %% [markdown]
# ### 0.2.1 Class label frequency
#
# **Class Frequency:** 'person' dominates (~28%), while 'sheep'/'cow' are rare (~3-4%).

# %%
label_counts = train_df[labels].sum().sort_values(ascending=False)

CLASS_CMAP = plt.cm.tab20   # 20-distinct-colour palette for class plots
fig, ax = plt.subplots(figsize=(14, 4.5))
bars = ax.bar(label_counts.index, label_counts.values,
              color=[CLASS_CMAP(i / len(labels)) for i in range(len(labels))])
ax.set_ylabel("Number of images"); ax.set_xlabel("Class")
ax.set_title("Per-class image count (training set)")
ax.tick_params(axis="x", rotation=45)
for bar, v in zip(bars, label_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 3, str(int(v)),
            ha="center", va="bottom", fontsize=8)
plt.tight_layout(); plt.show()

print("Per-class counts (sorted, descending):")
for cls, cnt in label_counts.items():
    pct = 100 * cnt / len(train_df)
    print(f"  {cls:<15} {int(cnt):4d}  ({pct:.1f}%)")

# %% [markdown]
# ### 0.2.2 Labels per image
#
# **Label Cardinality:** Images typically carry only 1-3 of the 20 classes.

# %%
labels_per_image = train_df[labels].sum(axis=1)

print("Label cardinality (number of labels per image):")
print(labels_per_image.describe())
print()
card_counts = labels_per_image.value_counts().sort_index()
for k, v in card_counts.items():
    print(f"  {int(k)} label(s) : {v} images  ({100*v/len(train_df):.1f}%)")

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.bar(card_counts.index, card_counts.values, color="steelblue")
ax.set_xlabel("Labels per image"); ax.set_ylabel("Count")
ax.set_title("Label cardinality distribution")
ax.set_xticks(card_counts.index)
plt.tight_layout(); plt.show()

# Positive / negative (image,class) pair imbalance — drives the loss choice.
total_pairs = len(train_df) * len(labels)
pos_pairs   = int(train_df[labels].values.sum())
neg_pairs   = total_pairs - pos_pairs
print(f"\nPositive (image,class) pairs: {pos_pairs} / {total_pairs} = {100*pos_pairs/total_pairs:.2f}%")
print(f"Negative (image,class) pairs: {neg_pairs} / {total_pairs} = {100*neg_pairs/total_pairs:.2f}%")
print(f"Positive/Negative ratio: 1 : {neg_pairs/pos_pairs:.1f}")

# %% [markdown]
# ### 0.2.3 Pixel distribution across masks
#
# **Pixel Distribution:** Background dominates ~78% of the masks.

# %%
counter = Counter()
for seg in train_df["seg"]:
    counter.update(seg.flatten().tolist())

total = sum(counter.values())

print(f"{'value':>5} {'class':<14} {'pixels':>14} {'%':>8}")
print("-" * 45)
for v in sorted(counter.keys()):
    name = VOC_CLASSES[v] if v < len(VOC_CLASSES) else f"<{v}>"
    pct = 100 * counter[v] / total
    print(f"{v:>5} {name:<14} {counter[v]:>14,} {pct:>7.3f}%")

# %% [markdown]
# ## 0.3 Multi-label stratified train/validation split

# %%

# 1. Initialize the stratified splitter
# We use 5 splits (which gives an 80% train / 20% val split)
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 2. Extract our target labels as a numpy array for the stratifier
y_all = train_df[labels].values

# 3. Generate the train and validation indices for the first fold
# mskf.split expects (X, y). We can pass a dummy array for X of the same length as y.
for tr_idx, va_idx in mskf.split(np.zeros(len(y_all)), y_all):
    train_indices = tr_idx
    val_indices = va_idx
    break # We only need the first split for our train/val sets

# 4. Create the specific classification dataframes the loader is looking for
train_df_cls = train_df.iloc[train_indices].reset_index(drop=True)
val_df_cls = train_df.iloc[val_indices].reset_index(drop=True)

print(f"Successfully created 'train_df_cls' with {len(train_df_cls)} samples.")
print(f"Successfully created 'val_df_cls' with {len(val_df_cls)} samples.")


# %% [markdown]
# ## 0.4 Data Augmentation & DataLoaders
# We centralize Datasets, Transforms, and Dataloaders here to ensure consistency and modularity across the 4 specific models. Transforms are specifically scaled dynamically according to model VRAM requirements and architectural needs.

# %%

def calculate_dataset_stats(df):
    """
    Dynamically calculates the channel-wise mean and standard deviation 
    of the entire dataset.
    """
    print("Calculating dynamic dataset statistics...")
    
    # 1. Stack all images into a single array to vectorize the calculation.
    # We reshape to (-1, 3) to flatten all spatial dimensions, leaving just the color channels.
    all_pixels = np.concatenate([img.reshape(-1, 3) for img in df["img"]], axis=0)
    
    # 2. Calculate the mean and std per channel (R, G, B)
    # The output will be an array of shape (3,)
    raw_mean = np.mean(all_pixels, axis=0)
    raw_std = np.std(all_pixels, axis=0)
    
    # 3. Scale values for Albumentations Normalize (which expects values in [0, 1])
    scaled_mean = raw_mean / 255.0
    scaled_std = raw_std / 255.0
    
    # 4. Prepare the constant padding color (OpenCV expects integers)
    pad_color = tuple(np.round(raw_mean).astype(int))
    
    print(f"Calculated Mean (scaled): {scaled_mean}")
    print(f"Calculated Std (scaled):  {scaled_std}")
    print(f"Calculated Pad Color:     {pad_color}")
    
    return tuple(scaled_mean), tuple(scaled_std), pad_color

# Execute the calculation
IMG_MEAN, IMG_STD, MEAN_C = calculate_dataset_stats(train_df)

def get_transforms(model_name):
    """Returns dynamic transforms tailored to the specific model."""
    
    # 1. Determine which statistics to use based on the model type
    if model_name in ["clip", "mask2former"]:
        # Pre-trained models strictly require ImageNet statistics
        active_mean, active_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        # Models trained from scratch use your dynamically calculated dataset statistics
        active_mean, active_std = IMG_MEAN, IMG_STD

    # 2. Apply the chosen statistics (active_mean, active_std) to the transforms
    if model_name in ["smallcnn", "clip"]:
        crop_size = 224
        train_tf = A.Compose([
            A.RandomResizedCrop(size=(crop_size, crop_size), scale=(0.6, 1.0), ratio=(0.75, 1.3333)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.Normalize(mean=active_mean, std=active_std),
            ToTensorV2(),
        ])
        val_tf = A.Compose([
            A.Resize(crop_size, crop_size),
            A.Normalize(mean=active_mean, std=active_std),
            ToTensorV2(),
        ])
        
    elif model_name in ["unet", "mask2former"]:
        crop_size = 256
        
        # Use a slightly more robust M2F-style pipeline for both
        train_tf = A.Compose([
            A.LongestMaxSize(max_size=int(crop_size * 1.5), p=1.0),
            A.RandomScale(scale_limit=(-0.2, 0.5), p=1.0), 
            A.PadIfNeeded(min_height=crop_size, min_width=crop_size, border_mode=cv2.BORDER_CONSTANT, fill=MEAN_C, fill_mask=255),
            A.RandomCrop(height=crop_size, width=crop_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=active_mean, std=active_std), # This handles the difference!
            ToTensorV2(),
        ])
        
        val_tf = A.Compose([
            A.LongestMaxSize(max_size=crop_size, p=1.0),
            A.PadIfNeeded(min_height=crop_size, min_width=crop_size, border_mode=cv2.BORDER_CONSTANT, fill=MEAN_C, fill_mask=255),
            A.Normalize(mean=active_mean, std=active_std),
            ToTensorV2(),
        ])
        
    return train_tf, val_tf

# --- Dataset Classes ---
class VOCClassificationDataset(Dataset):
    def __init__(self, df, label_cols, transform=None):
        self.df = df
        self.label_cols = list(label_cols)
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = row["img"]
        y = row[self.label_cols].values.astype(np.float32)
        if self.transform: img = self.transform(image=img)["image"]
        return img, torch.from_numpy(y)

class VOCSegDataset(Dataset):
    def __init__(self, df, indices, transform):
        self.df = df
        self.indices = list(indices)
        self.transform = transform

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        row = self.df.iloc[self.indices[i]]
        img = row["img"]
        seg = row["seg"].astype(np.int32)
        out = self.transform(image=img, mask=seg)
        return out["image"], out["mask"].long()

# --- Loader Factory ---
def get_dataloaders(model_name, task="classification", batch_size=16, num_workers=0):
    train_tf, val_tf = get_transforms(model_name)
    if task == "classification":
        train_ds = VOCClassificationDataset(train_df_cls, labels, train_tf)
        val_ds = VOCClassificationDataset(val_df_cls, labels, val_tf)
    else:
        train_ds = VOCSegDataset(train_df, train_indices, train_tf)
        val_ds = VOCSegDataset(train_df, val_indices, val_tf)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=(task=="segmentation"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

# %% [markdown]
# ## 0.5 Your Kaggle submission
# Transforms your test dataframe into a submission.csv file using Run-Length Encoding.

# %%
def _rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle = ' '.join(str(x) for x in runs)
    return rle

def generate_submission(df):
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
#
# The goal here is simple: implement a classification model and train it to recognise all 20 classes (and/or background) using the training set and compete on the test set (by filling in the classification columns in the test dataframe).

# %% [markdown]
# ## 1.1 Benchmark: Random classification

# %%
class RandomClassificationModel:
    def fit(self, X, y):
        self.distribution = np.mean(y, axis=0)
        return self
        
    def predict(self, X):
        np.random.seed(0)
        return [np.array([int(np.random.rand() < p) for p in self.distribution]) for _ in X]
    
    def __call__(self, X):
        return self.predict(X)

model = RandomClassificationModel()
model.fit(train_df["img"], train_df[labels])
test_df.loc[:, labels] = model.predict(test_df["img"])

# %% [markdown]
# ## 1.2 Training from scratch: SmallCNN
#
# **Architecture:** A 5-block VGG-style CNN doubling channels (32 → 512). BatchNorm is critical for stabilizing the network when training from scratch on small datasets.
#
# **Loss:** `BCEWithLogitsLoss(reduction="mean")`. Treats the 20 outputs as independent Bernoulli trials to handle multiple labels per image appropriately.

# %% [markdown]
# ### 1.2.1 SmallCNN pipeline

# %%
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

class SmallCNN(nn.Module):
    def __init__(self, n_classes=20, in_ch=3):
        super().__init__()
        chs = [32, 64, 128, 256, 512]
        layers = []
        prev = in_ch
        for c in chs:
            layers += [
                nn.Conv2d(prev, c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            prev = c
        self.features = nn.Sequential(*layers)
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.head    = nn.Linear(chs[-1], n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.head(x)

@torch.no_grad()
def evaluate_cls(model, loader, loss_fn, device):
    """Evaluates the classification model and calculates loss and mAP."""
    model.eval()
    losses, all_logits, all_targets = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        losses.append(loss_fn(logits, yb).item())
        all_logits.append(logits.float().cpu())
        all_targets.append(yb.cpu())
    
    all_logits, all_targets = torch.cat(all_logits), torch.cat(all_targets)
    probs, targets_np = torch.sigmoid(all_logits).numpy(), all_targets.numpy()

    aps = [average_precision_score(targets_np[:, c], probs[:, c]) for c in range(targets_np.shape[1]) if targets_np[:, c].sum() > 0]
    return float(np.mean(losses)), (float(np.mean(aps)) if aps else float("nan")), all_logits, all_targets

def tune_thresholds(val_logits, val_targets):
    """Finds the optimal per-class probability threshold that maximizes F1/Dice."""
    probs = torch.sigmoid(val_logits).numpy()
    targets = val_targets.numpy()
    best_thresholds, best_dices = np.full(targets.shape[1], 0.5), np.zeros(targets.shape[1])
    
    for c in range(targets.shape[1]):
        y_true = targets[:, c]
        if y_true.sum() == 0: continue
        best_t, best_d = 0.5, 0.0
        for t in np.arange(0.1, 0.9, 0.05):
            y_pred = (probs[:, c] > t).astype(int)
            intersection = (y_true * y_pred).sum()
            dice = (2. * intersection + 1e-7) / (y_true.sum() + y_pred.sum() + 1e-7)
            if dice > best_d:
                best_d, best_t = dice, t
        best_thresholds[c], best_dices[c] = best_t, best_d
    return best_thresholds, best_dices

def predict_cls(model, dataset, thresholds, device='cuda', batch_size=16, num_workers=0):
    """Generates binary predictions using the classification model and tuned thresholds."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_probs = []
    with torch.no_grad():
        for xb in loader:
            all_probs.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    all_probs = np.concatenate(all_probs)
    return (all_probs > thresholds).astype(np.int8), all_probs

def train_scnn(model, train_loader, val_loader, epochs, optimizer, scheduler, loss_fn, device):
    """Trains the SmallCNN model using mixed precision and tracks the best mAP."""
    scaler = torch.amp.GradScaler('cuda')
    best_mAP, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(loss.item())

        scheduler.step()
        
        val_loss, val_mAP, last_logits, last_targets = evaluate_cls(model, val_loader, loss_fn, device=device)

        if val_mAP > best_mAP:
            best_mAP, best_state = val_mAP, model.state_dict()

        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}/{epochs} | train loss {np.mean(epoch_losses):.4f} | val loss {val_loss:.4f} | mAP {val_mAP:.4f}")

    return best_state, last_logits, last_targets

# %% [markdown]
# ### 1.2.2 Execute pipeline: SmallCNN

# %%
# 1. Dataloaders specifically tailored for SmallCNN constraints
train_loader_scnn, val_loader_scnn = get_dataloaders("smallcnn", task="classification", batch_size=SCNN_BATCH_SIZE)

EPOCHS_SMALLCNN = 50 if not SMOKE else 2
bce_loss_scnn = nn.BCEWithLogitsLoss(reduction="mean")
model_smallcnn = SmallCNN(n_classes=len(labels)).to(device)

if TRAIN:
    set_seed()
    # Optimizer and Scheduler remain focused on the SmallCNN parameters
    optimizer = torch.optim.AdamW(model_smallcnn.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_SMALLCNN)
    
    best_state_smallcnn, val_logits_smallcnn, val_targets_smallcnn = train_scnn(
        model_smallcnn, 
        train_loader_scnn, 
        val_loader_scnn, 
        epochs=EPOCHS_SMALLCNN, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        loss_fn=bce_loss_scnn,
        device=device
    )
    
    model_smallcnn.load_state_dict(best_state_smallcnn)
    # Saving with a specific scnn filename
    torch.save(best_state_smallcnn, "cls_smallcnn_bce.pt")
else:
    # Loading logic for pre-trained weights
    if Path("cls_smallcnn_bce.pt").exists():
        model_smallcnn.load_state_dict(torch.load("cls_smallcnn_bce.pt", map_location=device))
        
        _, _, val_logits_smallcnn, val_targets_smallcnn = evaluate_cls(
            model_smallcnn, 
            val_loader_scnn, 
            bce_loss_scnn,
            device=device
        )

# 2. Post-training threshold tuning
if 'val_logits_smallcnn' in locals():
    thresholds_smallcnn, dice_smallcnn = tune_thresholds(val_logits_smallcnn, val_targets_smallcnn)
    print(f"SmallCNN mean per-class Dice (val): {dice_smallcnn.mean():.4f}")

# %%
visualize_classification(model_smallcnn, val_loader_scnn, val_df_cls)

clean_up_vram()


# %% [markdown]
# ## 1.3 Transfer learning: CLIP
# **Architecture:** CLIP ViT-B/16 image encoder. We extract the raw 768-dim ViT features before the contrastive 512-dim projection to act as a robust semantic prior.
# **Training:** Two-phase execution. Phase 1 freezes the backbone, training only the classification head. Phase 2 unfreezes the last blocks using a lower learning rate.

# %% [markdown]
# ### 1.3.1 CLIP pipeline

# %%
class CLIPMultiLabel(nn.Module):
    def __init__(self, num_classes=20, freeze_backbone=True, head_dropout=0.3):
        super().__init__()
        self.backbone, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
        self.visual = self.backbone.visual
        if freeze_backbone:
            for p in self.visual.parameters(): p.requires_grad = False
        
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone.encode_image(x)
        return self.head(features.float()) 

    def unfreeze_last_n_blocks(self, n=4):
        resblocks = self.visual.transformer.resblocks
        for i in range(len(resblocks) - n, len(resblocks)):
            for p in resblocks[i].parameters(): p.requires_grad = True


def train_cls_clip(model, train_loader, val_loader, epochs, optimizer, scheduler, loss_fn, unfreeze_epoch=30):
    scaler = torch.amp.GradScaler('cuda')
    best_mAP, best_state = -1.0, None
    phase2_entered = False

    for epoch in range(1, epochs + 1):
        if epoch == unfreeze_epoch and not phase2_entered:
            phase2_entered = True
            model.unfreeze_last_n_blocks(n=4)
            optimizer = torch.optim.AdamW([
                {'params': [p for p in model.visual.parameters() if p.requires_grad], 'lr': 1e-5},
                {'params': model.head.parameters(), 'lr': 5e-4},
            ], weight_decay=1e-4)
            scaler = torch.amp.GradScaler('cuda') # Reset to avoid scale carry-over
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs - unfreeze_epoch))

        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(loss.item())

        scheduler.step()
        val_loss, val_mAP, last_logits, last_targets = evaluate_cls(model, val_loader, loss_fn, device=device)

        if val_mAP > best_mAP:
            best_mAP, best_state = val_mAP, model.state_dict()
            
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}/{epochs} | train loss {np.mean(epoch_losses):.4f} | val loss {val_loss:.4f} | mAP {val_mAP:.4f}")
            
    return best_state, last_logits, last_targets


# %% [markdown]
# ### 1.3.2 Execute pipeline: CLIP

# %%
# CLIP specific transforms and loaders
train_loader_clip, val_loader_clip = get_dataloaders("clip", task="classification", batch_size=SCNN_BATCH_SIZE)

EPOCHS_CLIP, UNFREEZE_CLIP = (75, 30) if not SMOKE else (2, 1)
bce_loss_clip = nn.BCEWithLogitsLoss(reduction="mean")
model_clip = CLIPMultiLabel(num_classes=len(labels), freeze_backbone=True).to(device)

if TRAIN:
    set_seed()
    opt_clip = torch.optim.AdamW(model_clip.head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched_clip = torch.optim.lr_scheduler.CosineAnnealingLR(opt_clip, T_max=UNFREEZE_CLIP)
    best_state_clip, val_logits_clip, val_targets_clip = train_cls_clip(
        model_clip, train_loader_clip, val_loader_clip, epochs=EPOCHS_CLIP, 
        optimizer=opt_clip, scheduler=sched_clip, loss_fn=bce_loss_clip, unfreeze_epoch=UNFREEZE_CLIP
    )
    model_clip.load_state_dict(best_state_clip)
    torch.save(best_state_clip, "clip_vitb16_multilabel.pt")
else:
    if Path("clip_vitb16_multilabel.pt").exists():
        model_clip.load_state_dict(torch.load("clip_vitb16_multilabel.pt", map_location=device))
        _, _, val_logits_clip, val_targets_clip = evaluate_cls(model_clip, val_loader_clip, bce_loss_clip)

if 'val_logits_clip' in locals():
    thresholds_clip, dice_clip = tune_thresholds(val_logits_clip, val_targets_clip)
    print("\n--- CLIP Validation Results ---\n")
    print(f"Mean Dice: {dice_clip.mean():.4f}", "\n")
    
    for i, label in enumerate(labels):
        print(f"{label:>15}: {dice_clip[i]:.4f}")

# %%
visualize_classification(model_clip, val_loader_clip, val_df_cls)

clean_up_vram()


# %% [markdown]
# ### 1.3.2 CLIP predictions for submission

# %%
# We apply test predictions using the best tuned CLIP Model threshold
class VOCClassificationDatasetTest(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        img = self.df.iloc[idx]["img"]
        if self.transform: img = self.transform(image=img)["image"]
        return img

if 'thresholds_clip' in locals():
    _, cls_val_tf = get_transforms("clip")
    test_ds_cls = VOCClassificationDatasetTest(test_df, transform=cls_val_tf)
    preds_test, probs_test = predict_cls(model_clip, test_ds_cls, thresholds_clip, batch_size=CLIP_BATCH_SIZE, num_workers=NUM_WORKERS)
    test_df.loc[:, labels] = preds_test
    print(f"Filled test_df with CLIP predictions. Shape: {preds_test.shape}")

# %% [markdown]
# # 2. Semantic segmentation
#
# The goal here is to implement a segmentation model that labels every pixel in the image as belonging to one of the 20 classes (and/or background). Use the training set to train your model and compete on the test set (by filling in the segmentation column in the test dataframe).

# %%
# Generic utilities for semantic segmentation tasks
class DiceLoss(nn.Module):
    def __init__(self, num_classes=21, smooth=1.0):
        super().__init__()
        self.num_classes, self.smooth = num_classes, smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets.clamp(0, 20), self.num_classes).permute(0, 3, 1, 2).float()
        
        # Mask out ignore index
        mask = (targets != 255).unsqueeze(1).float()
        probs, targets_oh = probs * mask, targets_oh * mask

        dims = (0, 2, 3)
        inter = (probs * targets_oh).sum(dims)
        dice = (2 * inter + self.smooth) / (probs.sum(dims) + targets_oh.sum(dims) + self.smooth)

        present = targets_oh.sum(dims) > 0 
        dice, present = dice[1:], present[1:] # ignore background
        return 1.0 - dice[present].mean() if present.any() else logits.sum() * 0.0 

class CEDiceLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=255)
        self.dice = DiceLoss()
        self.ce_w, self.dc_w = ce_weight, dice_weight

    def forward(self, logits, targets):
        return self.ce_w * self.ce(logits, targets) + self.dc_w * self.dice(logits, targets)

class SegmentationMetrics:
    def __init__(self, num_classes=21):
        self.num_classes = num_classes
        self.reset()
    def reset(self):
        self.confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)
    @torch.no_grad()
    def update(self, logits, targets):
        preds = logits.argmax(dim=1)
        self.update_preds(preds, targets)
        
    @torch.no_grad()
    def update_preds(self, preds, targets):
        preds = preds.flatten().cpu()
        targets = targets.flatten().cpu()
        valid = targets != 255
        idx = targets[valid] * self.num_classes + preds[valid]
        self.confusion += torch.bincount(idx, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
    def compute(self):
        cm = self.confusion.float()
        tp = cm.diagonal()
        fn, fp = cm.sum(dim=1) - tp, cm.sum(dim=0) - tp
        dice_per_class = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
        actual_present = (cm.sum(dim=1) > 0)
        mean_dice = dice_per_class[1:][actual_present[1:]].mean().item() if actual_present[1:].any() else 0.0
        return {"per_class_dice": dice_per_class.tolist(), "mean_dice": mean_dice}


# %% [markdown]
# ## 2.1 Benchmark: Random segmentation

# %%
class RandomSegmentationModel:
    def fit(self, X, Y):
        self.distribution = np.mean([[np.sum(Y_ == i) / Y_.size for i in range(len(labels) + 1)] for Y_ in Y], axis=0)
        return self
        
    def predict(self, X):
        np.random.seed(0)
        return [np.random.choice(np.arange(len(labels) + 1), size=X_.shape[:2], p=self.distribution) for X_ in X]
    
    def __call__(self, X):
        return self.predict(X)

# %% [markdown]
# ## 2.2 Training from scratch: U-Net
# **Architecture:** Standard U-Net (Ronneberger et al., 2015). Skip connections are essential for preserving fine spatial bounds from scratch when handling small, non-pretrained datasets.
#
# **Loss:** `CEDiceLoss`. A blend of pixel-wise CrossEntropy for stable early gradients, and Dice Loss to align with the competition's area-overlap evaluation metric.

# %% [markdown]
# ### 2.2.1 U-Net pipeline

# %%
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool, self.conv = nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)
    def forward(self, x, skip): return self.conv(torch.cat([skip, self.up(x)], dim=1))
    
    
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=21, base_channels=32):
        super().__init__()
        c = base_channels
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2, self.enc3, self.enc4 = Down(c, c*2), Down(c*2, c*4), Down(c*4, c*8)
        self.bottleneck = Down(c*8, c*16)
        
        self.dec4, self.dec3 = Up(c*16, c*8, c*8), Up(c*8, c*4, c*4)
        self.dec2, self.dec1 = Up(c*4, c*2, c*2), Up(c*2, c, c)
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b = self.bottleneck(s4)
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return self.head(d1)
    

@torch.no_grad()
def validate_unet(model, df, indices, loss_fn, multiple=16):
    model.eval()
    metrics = SegmentationMetrics()
    running_loss, n = 0.0, 0
    _, val_tf = get_transforms("unet")

    for i in indices:
        img_np = df["img"].iloc[i] 
        seg = df["seg"].iloc[i] 
        H, W = img_np.shape[:2]

        pad_h, pad_w = (multiple - H % multiple) % multiple, (multiple - W % multiple) % multiple
        x = img_np.astype(np.float32) / 255.0
        x = (x - np.array(IMG_MEAN)) / np.array(IMG_STD)
        x = np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)))
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(device)

        target = torch.from_numpy(seg.astype(np.int64)).unsqueeze(0).to(device)
        logits = model(x)[:, :, :H, :W] # crop padding off
        loss = loss_fn(logits, target)
        
        running_loss += loss.item()
        n += 1
        metrics.update(logits, target)

    return running_loss / n, metrics.compute()


# %% [markdown]
# ### 2.2.2 Execute pipeline: U-Net

# %%
train_loader_unet, val_loader_unet = get_dataloaders("unet", task="segmentation", batch_size=UNET_BATCH_SIZE)

model_unet = UNet(in_channels=3, num_classes=21, base_channels=64).to(device)
loss_fn = CEDiceLoss(ce_weight=1.0, dice_weight=5.0)

# Update below the number of epochs to limit training time.
# As a reference, the model achieves Dice = 0.2 only after 250 epochs.
if TRAIN:
    NUM_EPOCHS = 20 if not SMOKE else 2 
    optimizer = torch.optim.Adam(model_unet.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.amp.GradScaler('cuda')
    best_val_dice = -1.0
    accumulation_steps = UNET_ACCUM_STEPS
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model_unet.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_losses = []
        for i, (images, masks) in enumerate(train_loader_unet):
            images, masks = images.to(device), masks.to(device)
            
            with torch.amp.autocast('cuda'):
                loss = loss_fn(model_unet(images), masks) / accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader_unet):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            epoch_losses.append((loss * accumulation_steps).item())
        scheduler.step()
        
        if epoch % 5 == 0:
            val_loss, val_result = validate_unet(model_unet, train_df, val_indices, loss_fn)
            if val_result['mean_dice'] > best_val_dice:
                best_val_dice = val_result['mean_dice']
                torch.save(model_unet.state_dict(), "final_unet.pt")
            print(f"U-Net Epoch {epoch:3d}/{NUM_EPOCHS} | train loss {np.mean(epoch_losses):.4f} | val loss {val_loss:.4f} | Val Dice: {val_result['mean_dice']:.4f}")
            
else:
    if Path("final_unet.pt").exists():
        model_unet.load_state_dict(torch.load("final_unet.pt", map_location=device))
        val_loss, val_result = validate_unet(model_unet, train_df, val_indices, loss_fn)
        print(f"U-Net Loaded Val Dice: {val_result['mean_dice']:.4f}")

# %%
visualize_segmentation(model_unet, val_loader_unet, save_name="final_unet", voc_classes=VOC_CLASSES)

clean_up_vram()


# %% [markdown]
# ## 2.3 Transfer learning: Mask2Former
# **Architecture:** Utilizes Set Prediction and Bipartite (Hungarian) Matching, processing masks globally rather than relying solely on pixel-by-pixel cross-entropy.

# %% [markdown]
# ### 2.3.1 Mask2Former Pipeline (Bipartite Matching)

# %%
def prepare_mask2former_targets(masks_tensor, ignore_index=255):
    """Converts a (B, H, W) mask into the list of binary masks Mask2Former requires."""
    mask_labels, class_labels = [], []
    for mask in masks_tensor:
        classes = torch.unique(mask)
        classes = classes[classes != ignore_index]
        
        b_masks, b_classes = [], []
        for c in classes:
            b_masks.append(mask == c)
            b_classes.append(c)
            
        if not b_masks: # Fallback for pure padding crops
            b_masks.append(torch.zeros_like(mask, dtype=torch.bool))
            b_classes.append(torch.tensor(0, device=mask.device))
            
        mask_labels.append(torch.stack(b_masks).to(torch.float32))
        class_labels.append(torch.stack(b_classes).to(torch.int64))
    return mask_labels, class_labels

@torch.no_grad()
def evaluate_m2f(model, val_loader, device):
    model.eval()
    losses = []
    processor = Mask2FormerImageProcessor(ignore_index=255, do_resize=False, do_rescale=False, do_normalize=False)
    metrics = SegmentationMetrics(num_classes=21)
    
    for images, masks in val_loader:
        images, masks = images.to(device), masks.to(device)
        mask_labels, class_labels = prepare_mask2former_targets(masks)
        
        with torch.amp.autocast('cuda'):
            outputs = model(pixel_values=images, mask_labels=mask_labels, class_labels=class_labels)
            loss = outputs.loss
        losses.append(loss.item())
        
        target_sizes = [mask.shape for mask in masks]
        preds_list = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
        preds = torch.stack(preds_list)
        
        metrics.update_preds(preds, masks)

    return np.mean(losses), metrics.compute()

# %% [markdown]
# ### 2.3.2 Execute pipeline: Mask2Former

# %%
train_loader_m2f, val_loader_m2f = get_dataloaders("mask2former", task="segmentation", batch_size=M2F_BATCH_SIZE, num_workers=NUM_WORKERS)

config = Mask2FormerConfig.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic")
config.num_queries = 100
config.num_labels = 21

hf_m2f = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-tiny-ade-semantic", config=config, ignore_mismatched_sizes=True).to(device)

if TRAIN:
    M2F_EPOCHS = 30 if not SMOKE else 2
    base_lr = 1e-4
    optimizer_m2f = torch.optim.AdamW([
        {'params': hf_m2f.model.pixel_level_module.encoder.parameters(), 'lr': base_lr * 0.1},
        {'params': hf_m2f.model.pixel_level_module.decoder.parameters(), 'lr': base_lr},
        {'params': hf_m2f.model.transformer_module.parameters(), 'lr': base_lr},
    ], weight_decay=1e-4)

    scaler = torch.amp.GradScaler('cuda')
    best_m2f_dice = -1.0
    
    for epoch in range(1, M2F_EPOCHS + 1):
        hf_m2f.train()
        epoch_losses = []
        optimizer_m2f.zero_grad(set_to_none=True)
        for i, (images, masks) in enumerate(train_loader_m2f):
            images, masks = images.to(device), masks.to(device)
            mask_labels, class_labels = prepare_mask2former_targets(masks)
            
            with torch.amp.autocast('cuda'):
                outputs = hf_m2f(pixel_values=images, mask_labels=mask_labels, class_labels=class_labels)
                loss = outputs.loss / M2F_ACCUM_STEPS
            scaler.scale(loss).backward()
            
            if (i + 1) % M2F_ACCUM_STEPS == 0 or (i + 1) == len(train_loader_m2f):
                scaler.step(optimizer_m2f)
                scaler.update()
                optimizer_m2f.zero_grad(set_to_none=True)
                
            epoch_losses.append((loss * M2F_ACCUM_STEPS).item())
            
        if epoch % 5 == 0:
            val_loss, val_result = evaluate_m2f(hf_m2f, val_loader_m2f, device)
            if val_result['mean_dice'] > best_m2f_dice:
                best_m2f_dice = val_result['mean_dice']
                torch.save(hf_m2f.state_dict(), "best_mask2former.pt")
            print(f"Mask2Former Epoch {epoch:3d}/{M2F_EPOCHS} | train loss {np.mean(epoch_losses):.4f} | val loss {val_loss:.4f} | Val Dice: {val_result['mean_dice']:.4f}")
            
    if not Path("best_mask2former.pt").exists():
        torch.save(hf_m2f.state_dict(), "best_mask2former.pt")
else:
    if Path("best_mask2former.pt").exists():
        hf_m2f.load_state_dict(torch.load("best_mask2former.pt", map_location=device))

print("\n--- Mask2Former Validation Results ---\n")
_, m2f_val_result = evaluate_m2f(hf_m2f, val_loader_m2f, device)
print(f"{'Mean Dice':>15}: {m2f_val_result['mean_dice']:.4f}", "\n")
for i, cls_name in enumerate(VOC_CLASSES):
    print(f"{cls_name:>15}: {m2f_val_result['per_class_dice'][i]:.4f}")

# %%
visualize_segmentation(hf_m2f, val_loader_m2f, save_name="best_mask2former", voc_classes=VOC_CLASSES)

clean_up_vram()


# %% [markdown]
# # 3. Submitting best results
# We use Mask2Former for the final segmentation generation in the submission DataFrame.

# %%
class ImageDatasetTest(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        img = self.df.iloc[idx]["img"]
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, self.df.iloc[idx]["img"].shape[:2]

def generate_test_predictions_m2f(model, test_df, val_transform):
    """Runs inference on the test set utilizing Mask2Former's specific Post-Processor."""
    model.eval()
    processor = Mask2FormerImageProcessor(ignore_index=255, do_resize=False, do_rescale=False, do_normalize=False)
    
    test_ds = ImageDatasetTest(test_df, transform=val_transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    test_preds = []
    
    with torch.no_grad():
        for tensor_img, (orig_h, orig_w) in tqdm(test_loader, desc="Generating Submission"):
            tensor_img = tensor_img.to(device)
            outputs = model(pixel_values=tensor_img)
            
            # Post Process to original dimensions
            preds = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(orig_h.item(), orig_w.item())]
            )[0]
            test_preds.append(preds.cpu().numpy().astype(np.int8))
            
    return test_preds

# Apply predictions using native 384x384 transform wrapper scale for tiny Swin
_, seg_val_tf_m2f = get_transforms("mask2former") 
test_df["seg"] = generate_test_predictions_m2f(hf_m2f, test_df, seg_val_tf_m2f)

# Write to CSV
generate_submission(test_df)
print("Submission Generated Successfully.")

# %% [markdown]
# # 4. Adversarial attack
# For this part, your goal is to fool your classification and/or segmentation model, using an *adversarial attack*. More specifically, the goal is build a network to perturb test images in a way that (i) they look unperturbed to humans; but (ii) the original model classifies/segments these images in line with the perturbations.

# %% [markdown]
# # 5. Discussion
# Finally, take some time to reflect on what you have learned during this assignment. Why did you make certain design choices? How are the models learning and performing? Reflect and produce an overall discussion with links to the lectures and "real world" computer vision.

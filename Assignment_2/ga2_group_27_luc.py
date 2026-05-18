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

# %% [markdown]
# # Overview
# This assignment consists of *three main parts* for which we expect you to provide code and extensive documentation in the notebook, with a final discussion:
# * Image classification (Sect. 1)
# * Semantic segmentation (Sect. 2)
# * Adversarial attacks (Sect. 3)
# * Discussion (Sect. 4)

# %% [markdown]
# # 0. Setup and Global Imports
# Centralizing all imports and global variables ensures a clean namespace and 
# prevents dependency crashes if cells are executed out of order.

# %%
import os
import gc
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp
from transformers import (
    SegformerForSemanticSegmentation,
    Mask2FormerConfig, 
    Mask2FormerForUniversalSegmentation, 
    Mask2FormerImageProcessor
)

# Setup Device & Common Variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")


# Base Directory
DATA_PATH = ''
#DATA_PATH = '/kaggle/input/kul-computer-vision-ga-2-2026'

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# %% [markdown]
# ## PASCAL VOC 2009
# For this project you will be using the PASCAL VOC 2009 dataset. This dataset consists of colour images of various scenes with different object classes, totalling 20 classes plus background.

# %%
# Loading the training data
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
    axs[0, i].imshow(df.iloc[0]["img"], vmin=0, vmax=255)
    axs[0, i].set_title("\n".join(label for label in labels if df.iloc[0][label] == 1), fontsize=40)
    axs[0, i].axis("off")
    axs[1, i].imshow(df.iloc[0]["seg"], vmin=0, vmax=20) 
    axs[1, i].axis("off")
    
plt.show()

# %%
# Loading the test data
test_dir = os.path.join(DATA_PATH, "test")
test_df = pd.read_csv(os.path.join(test_dir, "test_set.csv"), index_col="Id")
test_df["img"] = [np.load(os.path.join(test_dir, "img", f"test_{idx}.npy")) for idx, _ in test_df.iterrows()]
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
# Your filled test dataframe must be converted to a submission.csv with two rows per example.

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
# The goal here is simple: implement a classification model and train it to recognise all 20 classes.

# %%
class RandomClassificationModel:
    def fit(self, X, y):
        self.distribution = np.mean(y, axis=0)
        print("Setting class distribution to:\n{}".format("\n".join(f"{label}: {p}" for label, p in zip(labels, self.distribution))))
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
# # 2. Semantic segmentation
# The goal here is to implement a segmentation model that labels every pixel in the image as belonging to one of the 20 classes (and/or background). 
#
# We test the performance of 3 models:
#
# ---
#
# 1. DeepLabV3+
#
# **Description:**  
# DeepLabV3+ is a high-performance CNN-based architecture that builds upon the concept of spatial pyramid pooling. It is designed to capture multi-scale context by using filters at multiple sampling rates. It is widely considered the "gold standard" for traditional deep learning segmentation due to its balance of boundary sharpness and category accuracy.
#
# **Key Components:**
# *   **Atrous Convolution:** Uses "dilated" filters to expand the receptive field without increasing the number of parameters.
# *   **ASPP (Atrous Spatial Pyramid Pooling):** Probes the incoming feature map with filters at multiple rates to capture objects at various scales.
# *   **Encoder-Decoder Structure:** A simple yet effective decoder refines object boundaries by gradually recovering spatial information.
#
# ---
#
# 2. SegFormer
#
# **Description:**  
# SegFormer is a "Transformer-only" framework that defies the traditional need for complex decoders or positional encoding. It is highly efficient and remarkably robust to different input resolutions. Unlike earlier Vision Transformers (ViT), SegFormer is designed specifically for dense prediction tasks like segmentation.
#
# **Key Components:**
# *   **Hierarchical Transformer Encoder:** Produces multi-scale features (high-resolution coarse features and low-resolution fine features) similar to a CNN, but using self-attention.
# *   **Positional-Encoding-Free:** Uses overlapping patch embeddings and convolutions to provide positional information, allowing the model to perform well on resolutions it wasn't trained on.
# *   **Lightweight MLP Decoder:** Because the encoder captures powerful global context, the decoder only needs a simple Multi-Layer Perceptron to aggregate information.
#
# ---
#
# 3. Mask2Former
#
# **Description:**  
# Mask2Former is a revolutionary "Universal" segmentation model. While the other two models predict a class for every pixel, Mask2Former treats segmentation as a **mask classification** problem. It identifies a set of regions (masks) and then assigns a label to each region. This allows a single architecture to handle semantic, instance, and panoptic segmentation.
#
# **Key Components:**
# *   **Pixel Decoder:** A convolutional branch that enhances image features to a high resolution.
# *   **Transformer Decoder:** Uses a set of "Object Queries" (learnable vectors) that interact with image features to "search" for objects.
# *   **Masked Attention:** Constrains the attention mechanism to the predicted mask area, leading to faster convergence and better detail.
#
# ---
#
# Architectural Comparison
#
# | Feature | DeepLabV3+ | SegFormer | Mask2Former |
# | :--- | :--- | :--- | :--- |
# | **Core Paradigm** | Convolutional (CNN) | Transformer (Self-Attention) | Query-based (Set Prediction) |
# | **Backbone Type** | EfficientNet, ResNet, etc. | MiT (Mix Transformer) | Swin, ResNet, etc. |
# | **Context Extraction** | Dilated Convolutions (ASPP) | Global Self-Attention | Object Queries & Masked Attention |
# | **Loss Logic** | Pixel-wise Cross Entropy | Pixel-wise Cross Entropy | Hungarian Matching (Set Loss) |
# | **Output Type** | Per-pixel class probability | Per-pixel class probability | A set of Binary Masks + Labels |
# | **Resolution Support**| Best at trained resolution | Highly flexible / Scale invariant | Best at high resolutions |
#
# ---
#
# Structural Summary
# *   **DeepLabV3+** is **local-to-global**: It uses small kernels to build up a global understanding. 
# *   **SegFormer** is **inherently global**: Every pixel can "talk" to every other pixel from the first layer, making it better at understanding the relationship between distant objects.
# *   **Mask2Former** is **object-centric**: Instead of asking "What class is this pixel?", it asks "Where is an object, and what is its shape?", which allows it to excel at distinguishing overlapping instances.

# %% [markdown]
# ## 2.1 Train / Validation Split

# %%
SEED = 42
rng = np.random.default_rng(SEED)

n = len(train_df)
indices = rng.permutation(n)
n_val = int(0.2 * n)
val_indices   = sorted(indices[:n_val].tolist())
train_indices = sorted(indices[n_val:].tolist())

print(f"Train: {len(train_indices)} examples | Val: {len(val_indices)} examples")

# %% [markdown]
# ## 2.2 Shared Data Pipeline Components
# These classes and functions handle image augmentations and dataset structuring.

# %%
class VOCSegDataset(Dataset):
    def __init__(self, df, indices, transform):
        self.df = df
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row_idx = self.indices[i]
        img = self.df["img"].iloc[row_idx]
        seg = self.df["seg"].iloc[row_idx]

        out = self.transform(image=img, mask=seg)
        return out["image"], out["mask"].long()

def get_transforms(model_name="mask2former"):
    """Returns optimal train and val transforms based on the target model's VRAM footprint."""
    CROP_SIZE = 384 if model_name.lower() == "mask2former" else 512 
    IGNORE_INDEX = 255 
    MEAN_COLOR = (124, 116, 104) 

    train_transform = A.Compose([
        A.LongestMaxSize(max_size=int(CROP_SIZE * 1.5), p=1.0),
        A.RandomScale(scale_limit=(-0.5, 1.0), p=1.0),
        A.PadIfNeeded(min_height=CROP_SIZE, min_width=CROP_SIZE, border_mode=cv2.BORDER_CONSTANT, fill=MEAN_COLOR, fill_mask=IGNORE_INDEX),
        A.RandomCrop(height=CROP_SIZE, width=CROP_SIZE),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.LongestMaxSize(max_size=CROP_SIZE, p=1.0),
        A.PadIfNeeded(min_height=CROP_SIZE, min_width=CROP_SIZE, border_mode=cv2.BORDER_CONSTANT, fill=MEAN_COLOR, fill_mask=IGNORE_INDEX),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform

def get_dataloaders(model_name, batch_size, num_workers=4):
    """Centralized helper to prevent code duplication when swapping models."""
    train_transform, val_transform = get_transforms(model_name)
    
    train_dataset = VOCSegDataset(train_df, train_indices, train_transform)
    val_dataset = VOCSegDataset(train_df, val_indices, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"[{model_name.upper()}] DataLoaders ready: {len(train_loader)} train batches, {len(val_loader)} val batches.")
    return train_loader, val_loader

# %% [markdown]
# ## 2.3 Standard Semantic Segmentation Pipeline
# Applies to DeepLabV3+ (CNN) and SegFormer (Transformer).

# %%
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=255)
        self.dice = smp.losses.DiceLoss(mode="multiclass", classes=21, ignore_index=255, from_logits=True)
        
    def forward(self, y_pred, y_true):
        if y_true.dim() == 4:
            y_true = y_true.squeeze(1)
        return self.ce(y_pred, y_true) + self.dice(y_pred, y_true)

criterion = BCEDiceLoss()

class SegFormerWrapper(nn.Module):
    """Forces SegFormer to output standard (B, C, H, W) tensors."""
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, x):
        outputs = self.model(pixel_values=x).logits
        return F.interpolate(outputs, size=x.shape[-2:], mode="bilinear", align_corners=False)

def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs, save_name, target_effective_batch=16):
    """Standard training loop with Automatic Gradient Accumulation and Loss Plotting."""
    physical_batch = train_loader.batch_size
    accumulation_steps = max(1, target_effective_batch // physical_batch)
    
    scaler = torch.amp.GradScaler('cuda')
    best_val_loss = float("inf")
    train_losses, val_losses = [], [] # --- Tracking Lists ---

    print(f"--- Starting Training for {save_name} ---")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad() 
        
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks) / accumulation_steps 
                        
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
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
                    loss = criterion(model(images), masks)
                val_loss += loss.item() * images.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        # --- Store losses for plotting ---
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{save_name}.pt")
            
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', color='blue')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', color='red', linestyle='--')
    plt.title(f'Loss Curve: {save_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()
    
    return train_losses, val_losses

def evaluate_metrics(model, val_loader, save_name, num_classes=21):
    """Calculates Detailed Per-Class and Global Metrics."""
    model.load_state_dict(torch.load(f"{save_name}.pt", weights_only=True))
    model.eval()
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Evaluating {save_name}"):
            preds = torch.argmax(model(images.to(device)), dim=1)
            
            mask_flat = masks.view(-1).long().cpu()
            pred_flat = preds.view(-1).long().cpu()
            
            valid_indices = (mask_flat != 255) # Ignore Padding
            
            conf_matrix += torch.bincount(
                num_classes * mask_flat[valid_indices] + pred_flat[valid_indices], 
                minlength=num_classes**2
            ).reshape(num_classes, num_classes)

    tp = torch.diag(conf_matrix)
    fp = conf_matrix.sum(dim=0) - tp
    fn = conf_matrix.sum(dim=1) - tp
    
    iou = (tp.float() / (tp + fp + fn + 1e-12)).numpy()
    dice = ((2. * tp.float()) / (2. * tp + fp + fn + 1e-12)).numpy()
    
    print(f"\n--- Evaluation Results: {save_name} ---")
    for i, name in enumerate(VOC_CLASSES):
        print(f"{i:<3} {name:<15} IoU: {iou[i]:.4f} | Dice: {dice[i]:.4f}")

    mean_iou = np.mean(iou)
    mean_dice_all = np.mean(dice)
    mean_dice_foreground = np.mean(dice[1:])

    print("-" * 47)
    print(f"Mean IoU (All):         {mean_iou:.4f}")
    print(f"Mean Dice (All):        {mean_dice_all:.4f}")
    print(f"Mean Foreground Dice:   {mean_dice_foreground:.4f}  <-- Key Metric\n")
    
    return mean_iou, mean_dice_foreground

def visualize_predictions(model, val_loader, save_name, num_examples=4):
    """Standard Visualization logic with Legend."""
    model.load_state_dict(torch.load(f"{save_name}.pt", weights_only=True))
    model.eval()
    
    images, masks = next(iter(val_loader))
    with torch.no_grad():
        preds = torch.argmax(model(images.to(device)), dim=1).cpu().numpy()
        
    images = images.numpy()
    masks_np = masks.numpy()
    preds[masks_np == 255] = 0 # Force padding predictions to background
    masks_np[masks_np == 255] = 0 # Map 255 back to 0 for rendering
    
    cmap = plt.get_cmap("nipy_spectral")
    fig, axs = plt.subplots(num_examples, 3, figsize=(15, 5 * num_examples))
    for i in range(num_examples):
        img = np.clip(np.transpose(images[i] * np.array([0.229, 0.224, 0.225]).reshape(3,1,1) + 
                                   np.array([0.485, 0.456, 0.406]).reshape(3,1,1), (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
        axs[i, 0].imshow(img); axs[i, 0].set_title("Input"); axs[i, 0].axis("off")
        axs[i, 1].imshow(masks_np[i], vmin=0, vmax=20, cmap="nipy_spectral"); axs[i, 1].set_title("Ground Truth"); axs[i, 1].axis("off")
        axs[i, 2].imshow(preds[i], vmin=0, vmax=20, cmap="nipy_spectral"); axs[i, 2].set_title(f"Prediction ({save_name})"); axs[i, 2].axis("off")
        
        legend_patches = [mpatches.Patch(color=cmap(c / 20.0), label=VOC_CLASSES[c]) for c in np.unique(preds[i])]
        axs[i, 2].legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 2.3.1 Execute Pipeline: DeepLabV3+

# %%
torch.cuda.empty_cache(); gc.collect()
train_loader, val_loader = get_dataloaders("deeplabv3", batch_size=4)

model_deeplab = smp.DeepLabV3Plus(encoder_name="tu-tf_efficientnetv2_s", encoder_weights="imagenet", in_channels=3, classes=21).to(device)
optimizer_dl = torch.optim.AdamW(model_deeplab.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_dl = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_dl, T_max=40)

train_hist, val_hist = train_model(model_deeplab, train_loader, val_loader, optimizer_dl, scheduler_dl, epochs=40, save_name="best_deeplab")

# %%
evaluate_metrics(model_deeplab, val_loader, save_name="best_deeplab")

# %%
visualize_predictions(model_deeplab, val_loader, save_name="best_deeplab")

# %%
del model_deeplab, optimizer_dl, train_loader, val_loader # Free VRAM
torch.cuda.empty_cache(); gc.collect()

# %% [markdown]
# ## 2.3.2 Execute Pipeline: SegFormer-B2

# %%
torch.cuda.empty_cache(); gc.collect()
train_loader, val_loader = get_dataloaders("segformer", batch_size=4)

hf_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b2", num_labels=21) 
model_segformer = SegFormerWrapper(hf_model).to(device)

optimizer_sf = torch.optim.AdamW(model_segformer.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler_sf = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_sf, T_max=12)

train_hist, val_hist = train_model(model_segformer, train_loader, val_loader, optimizer_sf, scheduler_sf, epochs=12, save_name="best_segformer")

# %%
evaluate_metrics(model_segformer, val_loader, save_name="best_segformer")

# %%
visualize_predictions(model_segformer, val_loader, save_name="best_segformer")

# %%
del model_segformer, optimizer_sf, train_loader, val_loader # Free VRAM
torch.cuda.empty_cache(); gc.collect()


# %% [markdown]
# ## 2.4 Native Mask2Former Pipeline (Bipartite Matching)
# Mask2Former utilizes Set Prediction and Hungarian Matching, requiring binary object masks.

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

def train_mask2former_native(model, train_loader, val_loader, optimizer, scheduler, epochs, save_name, target_effective_batch=16):
    accumulation_steps = max(1, target_effective_batch // train_loader.batch_size)
    scaler = torch.amp.GradScaler('cuda')
    best_val_loss = float("inf")
    train_losses, val_losses = [], [] # --- Tracking Lists ---
    
    print(f"--- Starting Native Mask2Former Training ---")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            mask_labels, class_labels = prepare_mask2former_targets(masks, ignore_index=255)
            
            with torch.amp.autocast('cuda'):
                outputs = model(pixel_values=images, mask_labels=mask_labels, class_labels=class_labels)
                loss = outputs.loss / accumulation_steps
                        
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += (loss.item() * accumulation_steps) * images.size(0)
            
        train_loss /= len(train_loader.dataset)
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                mask_labels, class_labels = prepare_mask2former_targets(masks, ignore_index=255)
                with torch.amp.autocast('cuda'):
                    val_loss += model(pixel_values=images, mask_labels=mask_labels, class_labels=class_labels).loss.item() * images.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        # --- Store losses for plotting ---
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{save_name}.pt")

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', color='blue')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', color='red', linestyle='--')
    plt.title(f'Loss Curve: {save_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()
    
    return train_losses, val_losses

def evaluate_mask2former_native(model, val_loader, save_name, num_classes=21):
    """Native Mask2Former Evaluation with Detailed Metrics."""
    model.load_state_dict(torch.load(f"{save_name}.pt", weights_only=True))
    model.eval()
    processor = Mask2FormerImageProcessor(ignore_index=255, do_resize=False, do_rescale=False, do_normalize=False)
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Evaluating {save_name}"):
            outputs = model(pixel_values=images.to(device))
            
            target_sizes = [masks.shape[1:] for _ in range(images.shape[0])]
            preds = torch.stack(processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)).to(device)
            
            mask_flat, pred_flat = masks.view(-1).long().cpu(), preds.view(-1).long().cpu()
            valid_indices = (mask_flat != 255)
            
            conf_matrix += torch.bincount(
                num_classes * mask_flat[valid_indices] + pred_flat[valid_indices], 
                minlength=num_classes**2
            ).reshape(num_classes, num_classes)

    tp = torch.diag(conf_matrix)
    fp = conf_matrix.sum(dim=0) - tp
    fn = conf_matrix.sum(dim=1) - tp
    
    iou = (tp.float() / (tp + fp + fn + 1e-12)).numpy()
    dice = ((2. * tp.float()) / (2. * tp + fp + fn + 1e-12)).numpy()
    
    print(f"\n--- Evaluation Results: {save_name} ---")
    for i, name in enumerate(VOC_CLASSES):
        print(f"{i:<3} {name:<15} IoU: {iou[i]:.4f} | Dice: {dice[i]:.4f}")

    mean_iou = np.mean(iou)
    mean_dice_all = np.mean(dice)
    mean_dice_foreground = np.mean(dice[1:])

    print("-" * 47)
    print(f"Mean IoU (All):         {mean_iou:.4f}")
    print(f"Mean Dice (All):        {mean_dice_all:.4f}")
    print(f"Mean Foreground Dice:   {mean_dice_foreground:.4f}  <-- Key Metric\n")
    
    return mean_iou, mean_dice_foreground

def visualize_mask2former_native(model, val_loader, save_name, num_examples=4):
    """Custom Visualizer for Mask2Former with dynamic batch size handling"""
    model.load_state_dict(torch.load(f"{save_name}.pt", weights_only=True))
    model.eval()
    processor = Mask2FormerImageProcessor(ignore_index=255, do_resize=False, do_rescale=False, do_normalize=False)
    
    images, masks = next(iter(val_loader))
    
    # Adjust num_examples to not exceed the actual batch size
    actual_batch_size = images.shape[0]
    num_to_show = min(num_examples, actual_batch_size)
    
    with torch.no_grad():
        outputs = model(pixel_values=images.to(device))
        target_sizes = [masks.shape[1:] for _ in range(images.shape[0])]
        preds = torch.stack(processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)).cpu().numpy()
        
    images, masks_np = images.numpy(), masks.numpy()
    preds[masks_np == 255] = 0 
    masks_np[masks_np == 255] = 0 
    
    cmap = plt.get_cmap("nipy_spectral")
    
    # Create the plot based on num_to_show
    fig, axs = plt.subplots(num_to_show, 3, figsize=(15, 5 * num_to_show))
    
    # Handle the case where num_to_show is 1 (Matplotlib removes the first axis dimension)
    if num_to_show == 1:
        axs = np.expand_dims(axs, axis=0)

    for i in range(num_to_show):
        img = np.clip(np.transpose(images[i] * np.array([0.229, 0.224, 0.225]).reshape(3,1,1) + 
                                   np.array([0.485, 0.456, 0.406]).reshape(3,1,1), (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
        axs[i, 0].imshow(img); axs[i, 0].axis("off"); axs[i, 0].set_title("Input")
        axs[i, 1].imshow(masks_np[i], vmin=0, vmax=20, cmap="nipy_spectral"); axs[i, 1].axis("off"); axs[i, 1].set_title("Ground Truth")
        axs[i, 2].imshow(preds[i], vmin=0, vmax=20, cmap="nipy_spectral"); axs[i, 2].axis("off"); axs[i, 2].set_title(f"Prediction ({save_name})")
        
        legend_patches = [mpatches.Patch(color=cmap(c / 20.0), label=VOC_CLASSES[c]) for c in np.unique(preds[i])]
        axs[i, 2].legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
    plt.show()

# %% [markdown]
# ## 2.4.1 Execute Pipeline: Mask2Former

# %%
torch.cuda.empty_cache(); gc.collect()
train_loader, val_loader = get_dataloaders("mask2former", batch_size=2)

model_id = "facebook/mask2former-swin-tiny-ade-semantic"
config = Mask2FormerConfig.from_pretrained(model_id)
config.num_queries = 20  
config.num_labels = 21

hf_m2f = Mask2FormerForUniversalSegmentation.from_pretrained(model_id, config=config, ignore_mismatched_sizes=True).to(device)

base_lr = 1e-4
optimizer_m2f = torch.optim.AdamW([
    {'params': hf_m2f.model.pixel_level_module.encoder.parameters(), 'lr': base_lr * 0.1},
    {'params': hf_m2f.model.pixel_level_module.decoder.parameters(), 'lr': base_lr},
    {'params': hf_m2f.model.transformer_module.parameters(), 'lr': base_lr},
    {'params': hf_m2f.class_predictor.parameters(), 'lr': base_lr},
], weight_decay=1e-4)

scheduler_m2f = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_m2f, T_max=12)

train_hist, val_hist = train_mask2former_native(hf_m2f, train_loader, val_loader, optimizer_m2f, scheduler_m2f, epochs=12, save_name="native_mask2former")

# %%
evaluate_mask2former_native(hf_m2f, val_loader, save_name="native_mask2former")

# %%
visualize_mask2former_native(hf_m2f, val_loader, save_name="native_mask2former")

# %%
del hf_m2f, optimizer_m2f, train_loader, val_loader
torch.cuda.empty_cache(); gc.collect()


# %% [markdown]
# ## 2.5 Generate Test Set Predictions for Submission
# Using the best performing model (SegFormer) to create the final competition test predictions.

# %%
def generate_test_predictions(model, test_df, val_transform, save_name):
    """Runs inference on the test set and assigns it back to the dataframe."""
    model.load_state_dict(torch.load(f"{save_name}.pt", weights_only=True))
    model.eval()
    test_preds = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_df)), desc=f"Testing {save_name}"):
            # val_transform expects a dict format for Albumentations
            tensor_img = val_transform(image=test_df.iloc[idx]["img"])["image"].unsqueeze(0).to(device)
            # Standard wrapper output: get argmax class predictions
            preds = torch.argmax(model(tensor_img), dim=1).squeeze(0).cpu().numpy().astype(np.int8)
            
            # Ensure the output matches the original test image size (H, W)
            orig_h, orig_w = test_df.iloc[idx]["img"].shape[:2]
            if preds.shape != (orig_h, orig_w):
                # Resize prediction back to original dimensions using nearest neighbor to preserve class IDs
                preds = cv2.resize(preds, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                
            test_preds.append(preds)
            
    return test_preds

# Make sure model_segformer is loaded in memory for this to work
# If you deleted it above, re-initialize it here:
hf_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b2", num_labels=21) 
model_segformer = SegFormerWrapper(hf_model).to(device)
_, val_transform = get_transforms("segformer")

test_df["seg"] = generate_test_predictions(model_segformer, test_df, val_transform, save_name="best_segformer")

# %% [markdown]
# ## Submit to competition
# This transforms and writes your test dataframe into a submission.csv file.

# %%
generate_submission(test_df)
print("Submission generated successfully!")


# %% [markdown]
# # 3. Adversarial attack
# Goal: Build an encoder-decoder network to perturb test images so they are misclassified 
# while remaining visually identical to humans.

# %%
def get_saliency_map(model, img_tensor, target_class):
    """Generates a saliency map showing pixel importance for a target class."""
    img_tensor = img_tensor.clone().detach().to(device).requires_grad_(True)
    outputs = model(img_tensor)
    score = outputs[:, target_class, :, :].sum()
    model.zero_grad()
    score.backward()
    saliency, _ = torch.max(torch.abs(img_tensor.grad.data), dim=1)
    return saliency.squeeze().cpu().numpy()

def find_class_samples(loader):
    """Scans the dataloader to find exactly one image for each of the 20 VOC object classes."""
    samples = {}
    target_classes = set(range(1, 21)) # Classes 1-20
    
    with torch.no_grad():
        for images, masks in loader:
            for i in range(images.size(0)):
                img = images[i]
                mask = masks[i]
                present_classes = torch.unique(mask).cpu().numpy()
                
                for cls_idx in present_classes:
                    if cls_idx in target_classes and cls_idx not in samples:
                        # Store to CPU to avoid VRAM bloat
                        samples[cls_idx] = (img.cpu(), mask.cpu())
                
                if len(samples) == 20:
                    return samples
    return samples

def prepare_mask2former_targets(masks_tensor, ignore_index=255):
    """Helper to convert standard masks to Mask2Former binary object targets."""
    mask_labels, class_labels = [], []
    for mask in masks_tensor:
        classes = torch.unique(mask)
        classes = classes[classes != ignore_index]
        
        b_masks, b_classes = [], []
        for c in classes:
            b_masks.append(mask == c)
            b_classes.append(c)
            
        if not b_masks: 
            b_masks.append(torch.zeros_like(mask, dtype=torch.bool))
            b_classes.append(torch.tensor(0, device=mask.device))
            
        mask_labels.append(torch.stack(b_masks).to(torch.float32))
        class_labels.append(torch.stack(b_classes).to(torch.int64))
    return mask_labels, class_labels

# %% [markdown]
# ## 3.2 The Adversarial Network (Generator)

# %%
class AdversarialGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1), nn.Tanh()
        )

    def forward(self, x, strength=0.05):
        return self.dec(self.enc(x)) * strength

def train_adversary(generator, victim, loader, model_type="segformer", target_class=1, epochs=5, strength=0.3):
    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
    victim.eval()
    for param in victim.parameters():
        param.requires_grad = False
        
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Training {model_type.upper()} Adversary Epoch {epoch+1}")
        for images, _ in pbar:
            images = images.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                # FIX 1: Use the dynamic strength passed to the function
                delta = generator(images, strength=strength) 
                
                # FIX 2: Align the clamping bounds with the evaluation bounds
                perturbed = torch.clamp(images + delta, -3.0, 3.0) 
                
                if model_type == "segformer":
                    outputs = victim(perturbed)
                    target = torch.full((images.shape[0], images.shape[2], images.shape[3]), 
                                        target_class, dtype=torch.long, device=device)
                    loss_task = F.cross_entropy(outputs, target)
                    
                elif model_type == "mask2former":
                    adv_mask = torch.full((images.shape[0], images.shape[2], images.shape[3]), 
                                          target_class, dtype=torch.long, device=device)
                    mask_labels, class_labels = prepare_mask2former_targets(adv_mask)
                    outputs = victim(pixel_values=perturbed, mask_labels=mask_labels, class_labels=class_labels)
                    loss_task = outputs.loss
                
                loss = loss_task + 0.5 * torch.mean(delta**2)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

# %% [markdown]
# ## 3.3 Execution and Visual Analysis

# %% [markdown]
# The reason the artifacts appear much harsher on Mask2Former than on SegFormer at the exact same $\epsilon$ level is tied to the Feature Resolution and the Backbone Sensitivity.
#
# Why Mask2Former looks "messier"
#
# * Backbone Architecture: SegFormer (MiT-B2) uses a more traditional hierarchical transformer that naturally smooths local pixel relationships. Mask2Former, especially if using a Swin-Transformer backbone, is highly sensitive to the Pixel Decoder stage. The adversary has to "scream" much louder in the high-frequency space to disrupt the query-matching process, leading to those jagged, high-contrast artifacts you see.
#
# * The Query Bottleneck: To fool Mask2Former, the perturbation doesn't just need to flip a pixel; it needs to fundamentally change the Query Embeddings. This often requires "structural" noise that clashes with the natural edges of the image, making it much more obvious to a human observer.Tuning the LevelsTo achieve a "visible but recognizable" result for Mask2Former—similar to what you saw with SegFormer—you should lower the epsilon for the extreme case. Mask2Former usually "breaks" structurally at a much lower threshold.

# %%
import matplotlib.gridspec as gridspec

def run_adversarial_suite(generator, victim, samples_dict, model_type="segformer", filename="adversarial_compact.png"):
    """
    Creates a single consolidated visualization with minimized vertical gaps.
    """
    num_classes = len(samples_dict)
    fig = plt.figure(figsize=(30, 3.5 * num_classes))
    outer_gs = gridspec.GridSpec(num_classes, 1, figure=fig, hspace=0.1)
    
    if model_type == "segformer":
        levels = {
            "Baseline": 0.0, 
            "Minor (Invisible)": 0.05, 
            "Extreme (Visible)": 1.0
        }
    else:
        levels = {
            "Baseline": 0.0, 
            "Minor (Invisible)": 0.05, 
            "Extreme (Visible)": 0.4
        }
    
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    
    width_ratios = [0.6, 1, 1, 1, 0.2, 1, 1, 1, 0.2, 1, 1, 1]
    col_mappings = [[1, 2, 3], [5, 6, 7], [9, 10, 11]]
    sorted_keys = sorted(samples_dict.keys())
    
    if model_type == "mask2former":
        processor = Mask2FormerImageProcessor(ignore_index=255, do_resize=False, do_rescale=False, do_normalize=False)
    
    for row_idx, cls_idx in enumerate(sorted_keys):
        img, mask = samples_dict[cls_idx]
        img_t = img.unsqueeze(0).to(device)
        class_name = VOC_CLASSES[cls_idx].upper()
        
        inner_gs = gridspec.GridSpecFromSubplotSpec(1, 12, subplot_spec=outer_gs[row_idx], 
                                                    width_ratios=width_ratios, wspace=0.1)
        
        # --- COLUMN 1: CLASS NAME ---
        ax_title = fig.add_subplot(inner_gs[0, 0])
        ax_title.text(0.5, 0.5, class_name, fontsize=16, fontweight='bold', 
                      va='center', ha='center')
        ax_title.axis('off')

        for i, (level_name, s) in enumerate(levels.items()):
            delta = generator(img_t, strength=s) if s > 0 else torch.zeros_like(img_t)
            perturbed = torch.clamp(img_t + delta, -3.0, 3.0)
            
            with torch.no_grad():
                if model_type == "segformer":
                    outputs = victim(perturbed)
                    pred = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()
                elif model_type == "mask2former":
                    outputs = victim(pixel_values=perturbed)
                    pred = processor.post_process_semantic_segmentation(outputs, target_sizes=[img_t.shape[2:]])[0].cpu().numpy()
                    
                    # --- CRITICAL ARCHITECTURE FIX ---
                    # Mask2Former outputs '21' for pixels with no confident object mask.
                    # In PASCAL VOC, we map these 'unassigned' pixels to Background (0).
                    pred[pred == 21] = 0

            # Map the padding regions to background (0)
            pred[mask.cpu().numpy() == 255] = 0

            vis_img = np.clip(perturbed.squeeze().detach().cpu().numpy().transpose(1, 2, 0) * std + mean, 0, 1)
            vis_delta = np.clip(delta.squeeze().detach().cpu().numpy().transpose(1, 2, 0) + 0.5, 0, 1)
            
            target_cols = col_mappings[i]
            labels = ["Perturbation", "Attack Image", "Prediction"]
            plot_data = [vis_delta, vis_img, pred]
            
            for sub_idx, col in enumerate(target_cols):
                ax = fig.add_subplot(inner_gs[0, col])
                if sub_idx == 2:
                    ax.imshow(plot_data[sub_idx], cmap='nipy_spectral', vmin=0, vmax=20)
                else:
                    ax.imshow(plot_data[sub_idx])
                
                # Group headers
                if row_idx == 0 and sub_idx == 1:
                    ax.text(0.5, 1.15, level_name, transform=ax.transAxes, 
                            fontsize=16, fontweight='bold', ha='center', 
                            bbox=dict(facecolor='lightgray', alpha=0.8, boxstyle='round,pad=0.3'))
                
                if row_idx == 0:
                    ax.set_title(labels[sub_idx], fontsize=10)
                
                ax.axis('off')

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.savefig(filename, dpi=120, bbox_inches='tight')
    plt.show()


# %%
# ==========================================
# 1. SegFormer Attack Pipeline
# ==========================================
print("\n" + "="*50 + "\nINITIATING SEGFORMER ATTACK\n" + "="*50)
torch.cuda.empty_cache(); gc.collect()

adv_gen_sf = AdversarialGenerator().to(device)
hf_sf = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b2", num_labels=21)
victim_sf = SegFormerWrapper(hf_sf).to(device)
if os.path.exists("best_segformer.pt"):
    victim_sf.load_state_dict(torch.load("best_segformer.pt", weights_only=True, map_location=device))

train_loader_sf, val_loader_sf = get_dataloaders("segformer", batch_size=2)
print("Gathering class-representative images for SegFormer...")
samples_sf = find_class_samples(val_loader_sf)

if samples_sf:
    train_adversary(adv_gen_sf, victim_sf, train_loader_sf, model_type="segformer")

# %%
if samples_sf:
    run_adversarial_suite(adv_gen_sf, victim_sf, samples_sf, model_type="segformer", filename="adversarial_segformer.png")

# Free memory before starting Mask2Former
torch.cuda.empty_cache(); gc.collect()

# %%
# ==========================================
# 2. Mask2Former Attack Pipeline
# ==========================================
print("\n" + "="*50 + "\nINITIATING MASK2FORMER ATTACK\n" + "="*50)
torch.cuda.empty_cache(); gc.collect()

adv_gen_m2f = AdversarialGenerator().to(device)
model_id = "facebook/mask2former-swin-tiny-ade-semantic"
config = Mask2FormerConfig.from_pretrained(model_id)
config.num_queries = 20  
config.num_labels = 21
victim_m2f = Mask2FormerForUniversalSegmentation.from_pretrained(model_id, config=config, ignore_mismatched_sizes=True).to(device)

if os.path.exists("native_mask2former.pt"):
    victim_m2f.load_state_dict(torch.load("native_mask2former.pt", weights_only=True, map_location=device))

# Use batch_size 1 for Mask2Former Attack to prevent RTX 5060 OOM
train_loader_m2f, val_loader_m2f = get_dataloaders("mask2former", batch_size=1) 
print("Gathering class-representative images for Mask2Former...")
samples_m2f = find_class_samples(val_loader_m2f)

if samples_m2f:
    train_adversary(adv_gen_m2f, victim_m2f, train_loader_m2f, model_type="mask2former")

# %%
if samples_m2f:
    run_adversarial_suite(adv_gen_m2f, victim_m2f, samples_m2f, model_type="mask2former", filename="adversarial_mask2former.png")

# Final Cleanup
torch.cuda.empty_cache(); gc.collect()

# ISSUES WITH MASK2FORMER VISUALISATIONS

# %% [markdown]
# ## 3.4 Targeted attacks: force the model to segment a class across every image

# %%
def create_targeted_attack(victim_model, train_loader, target_name, model_type="segformer", epochs=5, strength=0.3):
    generator = AdversarialGenerator().to(device)
    
    if target_name not in VOC_CLASSES:
        raise ValueError(f"Target class '{target_name}' not found in VOC_CLASSES.")
    target_class_idx = VOC_CLASSES.index(target_name)
    
    train_adversary(
        generator=generator, 
        victim=victim_model, 
        loader=train_loader, 
        model_type=model_type, 
        target_class=target_class_idx, 
        epochs=epochs,
        strength=strength # Pass the strength down
    )
    
    return generator


# %%
import matplotlib.patches as mpatches

def verify_universal_attack(generator, victim, target_name, val_loader=None, samples_dict=None, model_type="segformer", filename="universal_attack.png"):
    """
    Visualizes the attack pipeline using a provided samples_dict, or automatically pulls a batch from val_loader.
    """
    generator.eval()
    victim.eval()
    
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    cmap = plt.get_cmap('nipy_spectral')
    
    if model_type == "mask2former":
        processor = Mask2FormerImageProcessor(ignore_index=255, do_resize=False, do_rescale=False, do_normalize=False)
        
    # --- FIX: Conditionally handle the inputs ---
    if samples_dict is None:
        if val_loader is None:
            raise ValueError("You must provide either 'val_loader' or 'samples_dict'.")
        # Extract one real validation batch and map it to a temporary dict format
        images_batch, masks_batch = next(iter(val_loader))
        samples_dict = {i: (images_batch[i], masks_batch[i]) for i in range(images_batch.size(0))}
    # -------------------------------------------
        
    num_images = len(samples_dict)
    fig, axs = plt.subplots(num_images, 5, figsize=(25, 5 * num_images))
    if num_images == 1:
        axs = np.expand_dims(axs, axis=0)
        
    fig.suptitle(f"Universal Targeted Attack: FORCE TO '{target_name.upper()}' ({model_type.upper()})", fontsize=26, fontweight='bold', y=1.01)

    attack_strength = 1.0 if model_type == "segformer" else 0.4
    
    for row_idx, cls_idx in enumerate(sorted(samples_dict.keys())):
        img_cpu, mask_cpu = samples_dict[cls_idx]
        img = img_cpu.unsqueeze(0).to(device)
        mask = mask_cpu.numpy()
        
        # Fallback for dynamic batch extraction where class ID isn't in VOC_CLASSES range
        class_original = VOC_CLASSES[cls_idx].capitalize() if cls_idx < len(VOC_CLASSES) else f"Image {cls_idx}"
        
        # ==========================================
        # 1. Clean Prediction
        # ==========================================
        with torch.no_grad():
            if model_type == "segformer":
                out_c = victim(img)
                pred_c = torch.argmax(out_c, dim=1).cpu().numpy()[0]
            elif model_type == "mask2former":
                out_c = victim(pixel_values=img)
                pred_c = torch.stack(processor.post_process_semantic_segmentation(out_c, target_sizes=[img.shape[2:]])).cpu().numpy()[0]
                pred_c[pred_c == 21] = 0
        
        # ==========================================
        # 2. Attack Execution
        # ==========================================
        delta = generator(img, strength=attack_strength)
        perturbed = torch.clamp(img + delta, -3.0, 3.0)
        
        # ==========================================
        # 3. Attacked Prediction
        # ==========================================
        with torch.no_grad():
            if model_type == "segformer":
                out_a = victim(perturbed)
                pred_a = torch.argmax(out_a, dim=1).cpu().numpy()[0]
            elif model_type == "mask2former":
                out_a = victim(pixel_values=perturbed)
                pred_a = torch.stack(processor.post_process_semantic_segmentation(out_a, target_sizes=[img.shape[2:]])).cpu().numpy()[0]
                pred_a[pred_a == 21] = 0

        # ==========================================
        # 4. Format Visuals
        # ==========================================
        vis_orig = np.clip(img.squeeze().detach().cpu().numpy().transpose(1, 2, 0) * std + mean, 0, 1)
        vis_attack = np.clip(perturbed.squeeze().detach().cpu().numpy().transpose(1, 2, 0) * std + mean, 0, 1)
        vis_delta = np.clip(delta.squeeze().detach().cpu().numpy().transpose(1, 2, 0) + 0.5, 0, 1)
        
        # Handle Padding
        pred_c[mask == 255] = 0
        pred_a[mask == 255] = 0
        
        # --- Column 1: Original Image ---
        axs[row_idx, 0].imshow(vis_orig)
        axs[row_idx, 0].set_title(f"1. Original ({class_original})", fontsize=14)
        axs[row_idx, 0].axis('off')
        
        # --- Column 2: Unperturbed Prediction ---
        axs[row_idx, 1].imshow(pred_c, cmap='nipy_spectral', vmin=0, vmax=20)
        axs[row_idx, 1].set_title("2. Clean Prediction", fontsize=14)
        axs[row_idx, 1].axis('off')
        u_c = np.unique(pred_c)
        axs[row_idx, 1].legend(handles=[mpatches.Patch(color=cmap(c/20.), label=VOC_CLASSES[c]) for c in u_c if c < len(VOC_CLASSES)], loc='best', fontsize=9, framealpha=0.7)
        
        # --- Column 3: Perturbed Image ---
        axs[row_idx, 2].imshow(vis_attack)
        axs[row_idx, 2].set_title(f"3. Perturbed Image (ε={attack_strength})", fontsize=14)
        axs[row_idx, 2].axis('off')
        
        # --- Column 4: Perturbation Signal ---
        axs[row_idx, 3].imshow(vis_delta)
        axs[row_idx, 3].set_title("4. Perturbation Signal", fontsize=14)
        axs[row_idx, 3].axis('off')
        
        # --- Column 5: Attacked Prediction ---
        axs[row_idx, 4].imshow(pred_a, cmap='nipy_spectral', vmin=0, vmax=20)
        axs[row_idx, 4].set_title("5. Attacked Prediction", fontsize=14)
        axs[row_idx, 4].axis('off')
        u_a = np.unique(pred_a)
        axs[row_idx, 4].legend(handles=[mpatches.Patch(color=cmap(c/20.), label=VOC_CLASSES[c]) for c in u_a if c < len(VOC_CLASSES)], loc='best', fontsize=9, framealpha=0.7)
        
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches='tight')
    plt.show()


# %%
# 1. Choose your target class
TARGET_CLASS_NAME = "sofa"  # Try "cat", "train", "sofa", etc.

# 2. Train the specific attack (Using SegFormer as an example)
class_generator = create_targeted_attack(
    victim_model=victim_sf, 
    train_loader=train_loader_sf, 
    target_name=TARGET_CLASS_NAME, 
    model_type="segformer",
    epochs=5  # Increase to 10 if the model is resisting the attack
)

# %%
# 3. Verify it works on a random batch of images
verify_universal_attack(
    generator=class_generator, 
    victim=victim_sf, 
    target_name=TARGET_CLASS_NAME,
    val_loader=val_loader_sf,      # Passed explicitly
    model_type="segformer"
)


# %%
def evaluate_attack_success(generator, victim, val_loader, target_name, model_type="segformer", eval_batches=10):
    """
    Evaluates the numerical success of a targeted adversarial attack.
    
    Args:
        eval_batches: How many batches to evaluate (to save time in the loop). 
                      Set to len(val_loader) for full dataset evaluation.
    """
    generator.eval()
    victim.eval()
    
    target_idx = VOC_CLASSES.index(target_name)
    attack_strength = 1.0 if model_type == "segformer" else 0.4
    
    if model_type == "mask2former":
        processor = Mask2FormerImageProcessor(ignore_index=255, do_resize=False, do_rescale=False, do_normalize=False)
        
    total_eligible_pixels = 0
    successful_attack_pixels = 0
    conf_matrix = torch.zeros(21, 21, dtype=torch.int64)
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            if i >= eval_batches:
                break
                
            images = images.to(device)
            
            # 1. Generate Attack
            delta = generator(images, strength=attack_strength)
            perturbed = torch.clamp(images + delta, -3.0, 3.0)
            
            # 2. Get Attacked Predictions
            if model_type == "segformer":
                outputs = victim(perturbed)
                preds = torch.argmax(outputs, dim=1).cpu()
            elif model_type == "mask2former":
                outputs = victim(pixel_values=perturbed)
                target_sizes = [images.shape[2:] for _ in range(images.shape[0])]
                preds = torch.stack(processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)).cpu()
                preds[preds == 21] = 0 # Handle unassigned pixels
                
            mask_flat = masks.view(-1).long()
            pred_flat = preds.view(-1).long()
            
            # 3. Calculate Attack Success Rate (ASR)
            valid_indices = (mask_flat != 255)
            # We only count pixels that were NOT originally the target class
            eligible_indices = valid_indices & (mask_flat != target_idx)
            
            total_eligible_pixels += eligible_indices.sum().item()
            successful_attack_pixels += (pred_flat[eligible_indices] == target_idx).sum().item()
            
            # 4. Calculate Degraded mIoU (Compared to Ground Truth)
            conf_matrix += torch.bincount(
                21 * mask_flat[valid_indices] + pred_flat[valid_indices], 
                minlength=21**2
            ).reshape(21, 21)

    # Final Metrics
    asr = (successful_attack_pixels / max(total_eligible_pixels, 1)) * 100
    
    tp = torch.diag(conf_matrix)
    fp = conf_matrix.sum(dim=0) - tp
    fn = conf_matrix.sum(dim=1) - tp
    iou = (tp.float() / (tp + fp + fn + 1e-12)).numpy()
    mean_iou = np.mean(iou)
    
    return asr, mean_iou


# %%
# %%
# ==========================================
# Loop: Universal Attack for Every Class
# ==========================================
import gc

# 1. Gather the 20 class-representative images globally
print("Gathering class-representative images from validation set...")
class_samples_dict = find_class_samples(val_loader_sf)
print(f"Found images for {len(class_samples_dict)}/20 classes.\n")

# Ensure memory is completely clean
torch.cuda.empty_cache()
gc.collect()

# Dictionary to store our numerical results
attack_results = {"Class": [], "ASR (%)": [], "Degraded mIoU": []}

# %%
# ==========================================
# Loop: Universal Attack for Every Class
# ==========================================
import gc

# 1. Gather the 20 class-representative images globally
print("Gathering class-representative images from validation set...")
class_samples_dict = find_class_samples(val_loader_sf)
print(f"Found images for {len(class_samples_dict)}/20 classes.\n")

# Ensure memory is completely clean
torch.cuda.empty_cache()
gc.collect()

# Dictionary to store our numerical results
attack_results = {"Class": [], "ASR (%)": [], "Degraded mIoU": []}

# --- YOUR UPDATED LOOP CODE GOES HERE ---
# We skip index 0 ("background") to only target foreground objects
for target_class in VOC_CLASSES[1:]:
    print(f"\n{'='*80}")
    print(f" Executing Universal Targeted Attack: {target_class.upper()}")
    print(f"{'='*80}\n")
    
    # Define the exact strength we will use for both training and evaluation
    current_attack_strength = 1.0 if "segformer" == "segformer" else 0.4
    
    # 2. Train the specific attack 
    class_generator = create_targeted_attack(
        victim_model=victim_sf, 
        train_loader=val_loader_sf, # FIXED: Train on val_loader (static images)
        target_name=target_class, 
        model_type="segformer",
        epochs=5,
        strength=current_attack_strength # FIXED: Synchronized training strength
    )

    # 3. Visualize the attack on the curated 20 images
    verify_universal_attack(
        generator=class_generator, 
        victim=victim_sf, 
        val_loader=None, 
        samples_dict=class_samples_dict, 
        target_name=target_class,
        model_type="segformer",
        filename=f"universal_attack_{target_class}.png"
    )
    
    # 4. Numerically Evaluate the Attack 
    print(f"Evaluating numerical success for {target_class.upper()}...")
    asr, mean_iou = evaluate_attack_success(
        generator=class_generator,
        victim=victim_sf,
        val_loader=val_loader_sf,
        target_name=target_class,
        model_type="segformer",
        eval_batches=10 
    )
    
    print(f"  -> Attack Success Rate (ASR): {asr:.2f}% of pixels forced to '{target_class}'")
    print(f"  -> Degraded Model mIoU:       {mean_iou:.4f}")
    
    # Save results for the final table
    attack_results["Class"].append(target_class)
    attack_results["ASR (%)"].append(asr)
    attack_results["Degraded mIoU"].append(mean_iou)
    
    # ==========================================
    # CRITICAL MEMORY CLEANUP
    # ==========================================
    del class_generator
    plt.close('all')
    torch.cuda.empty_cache()
    gc.collect()

# ==========================================
# FINAL SUMMARY REPORT
# ==========================================
import pandas as pd

print("\n" + "="*50)
print(" FINAL UNIVERSAL ATTACK SUMMARY REPORT")
print("="*50)
results_df = pd.DataFrame(attack_results)
# Sort by Attack Success Rate to see which classes were easiest to force
results_df = results_df.sort_values(by="ASR (%)", ascending=False).reset_index(drop=True)
print(results_df.to_string(index=False))

# %% [markdown]
#
#  FINAL UNIVERSAL ATTACK SUMMARY REPORT
#
#       Class   ASR (%)  Degraded mIoU
#       sofa 60.864354       0.025786
#       boat  3.696168       0.080277
#       tvmonitor  2.539727       0.138534
#       car  1.833709       0.144566
#       chair  0.559239       0.287705
#       pottedplant  0.485041       0.197031
#       cow  0.481038       0.186725
#       dog  0.369092       0.112621
#       sheep  0.360145       0.138348
#       aeroplane  0.329089       0.103639
#       bottle  0.129366       0.075379
#       diningtable  0.086211       0.107468
#       bird  0.063304       0.121929
#       bus  0.020970       0.099946
#       train  0.004041       0.095339
#       person  0.000513       0.077922
#       bicycle  0.000000       0.046085
#       cat  0.000000       0.154126
#       horse  0.000000       0.038225
#       motorbike  0.000000       0.043387

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # 4. Discussion
# Finally, take some time to reflect on what you have learned during this assignment. Why did you make certain design choices? How are the models learning and performing? Reflect and produce an overall discussion with links to the lectures and "real world" computer vision.

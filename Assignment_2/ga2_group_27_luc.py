# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
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

# Base Kaggle Directory
DATA_PATH = '/kaggle/input/kul-computer-vision-ga-2-2025'

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
    """Standard training loop with Automatic Gradient Accumulation."""
    physical_batch = train_loader.batch_size
    accumulation_steps = max(1, target_effective_batch // physical_batch)
    
    scaler = torch.amp.GradScaler('cuda')
    best_val_loss = float("inf")
    train_losses, val_losses = [], []

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
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{save_name}.pt")
            
    return train_losses, val_losses

def evaluate_metrics(model, val_loader, save_name, num_classes=21):
    """Calculates Detailed Per-Class Metrics."""
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

    print(f"\nMean Foreground Dice:   {np.mean(dice[1:]):.4f}\n")

def visualize_predictions(model, val_loader, save_name, num_examples=4):
    """Standard Visualization logic."""
    model.load_state_dict(torch.load(f"{save_name}.pt", weights_only=True))
    model.eval()
    
    images, masks = next(iter(val_loader))
    with torch.no_grad():
        preds = torch.argmax(model(images.to(device)), dim=1).cpu().numpy()
        
    images = images.numpy()
    masks_np = masks.numpy()
    
    # Force predictions in the padded areas to be 0 (Background)
    preds[masks_np == 255] = 0 
    
    masks_np[masks_np == 255] = 0 # Map 255 back to 0 for rendering
    
    cmap = plt.get_cmap("nipy_spectral")
    fig, axs = plt.subplots(num_examples, 3, figsize=(15, 5 * num_examples))
    for i in range(num_examples):
        img = np.clip(np.transpose(images[i] * np.array([0.229, 0.224, 0.225]).reshape(3,1,1) + 
                                   np.array([0.485, 0.456, 0.406]).reshape(3,1,1), (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
        axs[i, 0].imshow(img); axs[i, 0].set_title("Input"); axs[i, 0].axis("off")
        axs[i, 1].imshow(masks_np[i], vmin=0, vmax=20, cmap="nipy_spectral"); axs[i, 1].set_title("Ground Truth"); axs[i, 1].axis("off")
        axs[i, 2].imshow(preds[i], vmin=0, vmax=20, cmap="nipy_spectral"); axs[i, 2].set_title("Prediction"); axs[i, 2].axis("off")
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

train_model(model_deeplab, train_loader, val_loader, optimizer_dl, scheduler_dl, epochs=40, save_name="best_deeplab")
evaluate_metrics(model_deeplab, val_loader, save_name="best_deeplab")
visualize_predictions(model_deeplab, val_loader, save_name="best_deeplab")

del model_deeplab, optimizer_dl, train_loader, val_loader # Free VRAM

# %% [markdown]
# ## 2.3.2 Execute Pipeline: SegFormer-B2

# %%
torch.cuda.empty_cache(); gc.collect()
train_loader, val_loader = get_dataloaders("segformer", batch_size=4)

hf_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b2", num_labels=21) 
model_segformer = SegFormerWrapper(hf_model).to(device)

optimizer_sf = torch.optim.AdamW(model_segformer.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler_sf = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_sf, T_max=20)

train_model(model_segformer, train_loader, val_loader, optimizer_sf, scheduler_sf, epochs=20, save_name="best_segformer")
evaluate_metrics(model_segformer, val_loader, save_name="best_segformer")
visualize_predictions(model_segformer, val_loader, save_name="best_segformer")

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
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{save_name}.pt")

def evaluate_mask2former_native(model, val_loader, save_name, num_classes=21):
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
    
    dice = ((2. * tp.float()) / (2. * tp + fp + fn + 1e-12)).numpy()
    print(f"\nMean Foreground Dice: {np.mean(dice[1:]):.4f}\n")

def visualize_mask2former_native(model, val_loader, save_name, num_examples=4):
    """Custom Visualizer for Mask2Former Query Decoding"""
    model.load_state_dict(torch.load(f"{save_name}.pt", weights_only=True))
    model.eval()
    processor = Mask2FormerImageProcessor(ignore_index=255, do_resize=False, do_rescale=False, do_normalize=False)
    
    images, masks = next(iter(val_loader))
    with torch.no_grad():
        outputs = model(pixel_values=images.to(device))
        target_sizes = [masks.shape[1:] for _ in range(images.shape[0])]
        preds = torch.stack(processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)).cpu().numpy()
        
    images, masks_np = images.numpy(), masks.numpy()
    
    # Force predictions in the padded areas to be 0 (Background)
    preds[masks_np == 255] = 0 
    
    masks_np[masks_np == 255] = 0 # Map 255 back to 0 for rendering
    
    cmap = plt.get_cmap("nipy_spectral")
    fig, axs = plt.subplots(num_examples, 3, figsize=(15, 5 * num_examples))
    for i in range(num_examples):
        img = np.clip(np.transpose(images[i] * np.array([0.229, 0.224, 0.225]).reshape(3,1,1) + 
                                   np.array([0.485, 0.456, 0.406]).reshape(3,1,1), (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
        axs[i, 0].imshow(img); axs[i, 0].axis("off")
        axs[i, 1].imshow(masks_np[i], vmin=0, vmax=20, cmap="nipy_spectral"); axs[i, 1].axis("off")
        axs[i, 2].imshow(preds[i], vmin=0, vmax=20, cmap="nipy_spectral"); axs[i, 2].axis("off")
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

scheduler_m2f = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_m2f, T_max=30)

train_mask2former_native(hf_m2f, train_loader, val_loader, optimizer_m2f, scheduler_m2f, epochs=30, save_name="native_mask2former")
evaluate_mask2former_native(hf_m2f, val_loader, save_name="native_mask2former")
visualize_mask2former_native(hf_m2f, val_loader, save_name="native_mask2former")

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
# For this part, your goal is to fool your classification and/or segmentation model, using an *adversarial attack*. More specifically, the goal is build a network to perturb test images in a way that (i) they look unperturbed to humans; but (ii) the original model classifies/segments these images in line with the perturbations.

# %%


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # 4. Discussion
# Finally, take some time to reflect on what you have learned during this assignment. Why did you make certain design choices? How are the models learning and performing? Reflect and produce an overall discussion with links to the lectures and "real world" computer vision.

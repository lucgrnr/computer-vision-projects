import pandas as pd
import numpy as np
import cv2
import torch
import os
from tqdm import tqdm
from sklearn.preprocessing import normalize
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

from face_detection import DNNFaceDetector

FACE_SIZE = (160, 160)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)

def get_single_embedding(face_crop):
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (160, 160))
    face_tensor = preprocess(face_resized).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = resnet(face_tensor).cpu().numpy()[0]
    return normalize(emb.reshape(1, -1), norm='l2')[0]

def main():
    print("Extracting training data from annotations...")
    label_to_int = {
        "Michael_Cera": 0,
        "Jesse_Eisenberg": 1,
        "Mila_Kunis": 2,
        "Sarah_Hyland": 3,
        "Not_A_Face": 4
    }
    df = pd.read_csv("annotations.csv")
    DATASET_FOLDER = "datasets/train/"
    extracted_faces = []
    labels = []
    
    for _, row in df.iterrows():
        orig_file = row['original_file']     
        crop_path = row['saved_crop_path']
        label_str = row['manual_label']
        orig_path = os.path.join(DATASET_FOLDER, orig_file)
        if os.path.exists(orig_path) and pd.notna(crop_path) and os.path.exists(crop_path):
            crop_bgr = cv2.imread(crop_path)
            if crop_bgr is not None:
                extracted_faces.append(crop_bgr)
                labels.append(label_to_int[label_str])
            
    def augment(images, labels, num_variants=2):
        import random
        aug_X, aug_y = [], []
        if len(images) == 0:
            return np.array([]), np.array([])
        h, w = images[0].shape[:2]
        center = (w // 2, h // 2)
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for img, label in zip(images, labels):
            aug_X.append(img)
            aug_y.append(label)
            for _ in range(num_variants):
                aug_img = img.copy()
                if random.choice([True, False]):
                    aug_img = cv2.flip(aug_img, 1)
                light_choice = random.choice(["clahe", "dark", "light", "none"])
                aug_img_uint8 = np.clip(aug_img, 0, 255).astype(np.uint8)
                if light_choice == "clahe":
                    if len(aug_img_uint8.shape) == 3:
                        lab = cv2.cvtColor(aug_img_uint8, cv2.COLOR_RGB2LAB)
                        l, a, b = cv2.split(lab)
                        cl = clahe_obj.apply(l)
                        limg = cv2.merge((cl, a, b))
                        aug_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
                    else:
                        aug_img = clahe_obj.apply(aug_img_uint8)
                elif light_choice in ["dark", "light"]:
                    gamma = 0.7 if light_choice == "dark" else 1.3
                    invGamma = 1.0 / gamma
                    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                    aug_img = cv2.LUT(aug_img_uint8, table)
                geo_choice = random.choice(["rotate", "scale", "none"])
                if geo_choice == "rotate":
                    angle = random.choice([-8, -4, 4, 8])
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                elif geo_choice == "scale":
                    scale = random.choice([0.9, 1.1])
                    M = cv2.getRotationMatrix2D(center, 0, scale)
                    aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                if random.choice([True, False]):
                    box_w = random.randint(int(w * 0.15), int(w * 0.35))
                    box_h = random.randint(int(h * 0.15), int(h * 0.35))
                    x1 = random.randint(0, w - box_w)
                    y1 = random.randint(0, h - box_h)
                    if random.choice(["black", "noise"]) == "black":
                        aug_img[y1:y1+box_h, x1:x1+box_w] = 0
                    else:
                        if len(aug_img.shape) == 3:
                            noise = np.random.randint(0, 256, (box_h, box_w, 3), dtype=np.uint8)
                        else:
                            noise = np.random.randint(0, 256, (box_h, box_w), dtype=np.uint8)
                        aug_img[y1:y1+box_h, x1:x1+box_w] = noise
                if random.random() < 0.40:
                    k_size = random.choice([3, 5, 7, 9])
                    aug_img = cv2.GaussianBlur(aug_img, (k_size, k_size), 0)
                aug_X.append(aug_img)
                aug_y.append(label)
        return np.array(aug_X), np.array(aug_y)

    X_train_final, y_train_final = augment(extracted_faces, labels)
    
    # Extract WITH same setting for training
    train_embs = []
    for crop in tqdm(X_train_final, desc="Extracting Train Embeddings"):
        train_embs.append(get_single_embedding(crop))
        
    np.save('final_data/X_train_emb_no_tta.npy', np.array(train_embs))
    np.save('final_data/y_train_no_tta.npy', y_train_final)
    print("Train embeddings saved (NO TTA)!")

if __name__ == '__main__':
    main()

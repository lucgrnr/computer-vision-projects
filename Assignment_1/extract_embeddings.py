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
face_detector = DNNFaceDetector(path="models", face_size=FACE_SIZE)

def get_single_embedding(face_crop):
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (160, 160))
    face_tensor = preprocess(face_resized).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = resnet(face_tensor).cpu().numpy()[0]
    return normalize(emb.reshape(1, -1), norm='l2')[0]

def get_tta_embedding(face_crop):
    augmentations = [
        face_crop,
        cv2.flip(face_crop, 1)
    ]
    augmentations = [aug.astype(np.uint8) if aug.dtype != np.uint8 else aug for aug in augmentations]
    embs = []
    for aug in augmentations:
        embs.append(get_single_embedding(aug))
    averaged_emb = np.mean(embs, axis=0).reshape(1, -1)
    return normalize(averaged_emb, norm='l2')[0]

def main():
    print("Loading test dataset logic (memory safe)...")
    test = pd.read_csv('datasets/test_set.csv', index_col=0)
    
    all_image_embeddings = [None] * len(test)
    
    for idx, row in tqdm(test.iterrows(), total=len(test), desc="Extracting Test Embeddings"):
        img_bgr = np.load(f'datasets/test/test_{idx}.npy', allow_pickle=False)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        faces = face_detector.detect_faces(img_rgb)
        crops = []
        if faces:
            for (x, y, w, h, conf, sharp) in faces:
                crop = img_bgr[y:y+h, x:x+w]
                crops.append(cv2.resize(crop, face_detector.face_size, interpolation=cv2.INTER_AREA))
        
        img_embs = []
        for crop in crops:
            # Strictly NO-TTA to match dnn_try.py perfectly
            emb = get_single_embedding(crop)
            img_embs.append(emb)
            
        all_image_embeddings[idx] = img_embs

    np.save('final_data/X_test_embs_no_tta.npy', np.array(all_image_embeddings, dtype=object), allow_pickle=True)
    print("Test embeddings saved!")
    
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
            
    # We augment train set
    sys_path = os.path.abspath('.')
    import sys
    if sys_path not in sys.path: sys.path.append(sys_path)
    from dnn_try import augment
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

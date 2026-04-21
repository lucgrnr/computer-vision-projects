import pandas as pd
import numpy as np
import cv2
import torch
import os
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from face_detection import DNNFaceDetector

import random

# Force single thread for torch to not oversubscribe
torch.set_num_threads(1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)
    face_detector = DNNFaceDetector(path="models", face_size=(160, 160))
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    def extract_facenet_features_batch(faces_array):
        processed_tensors = []
        for face_crop_bgr in faces_array:
            face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (160, 160))
            face_tensor = preprocess(face_resized)
            processed_tensors.append(face_tensor)
            
        batch_tensor = torch.stack(processed_tensors).to(device)
        with torch.no_grad():
            embeddings = resnet(batch_tensor)
        return embeddings.cpu().numpy()
        
    print("Loading test logic (memory safe, simple eval)...")
    test = pd.read_csv('datasets/test_set.csv', index_col=0)
    # limit to 400 for speed evaluation of bugfix
    test = test.head(300)
    y_test_true = np.load('final_data/y_test.npy')[:len(test)]
    
    label_to_int = {"Michael_Cera": 0, "Jesse_Eisenberg": 1, "Mila_Kunis": 2, "Sarah_Hyland": 3, "Not_A_Face": 4}
    df = pd.read_csv("annotations.csv")
    DATASET_FOLDER = "datasets/train/"
    extracted_faces = []
    labels = []
    
    for _, row in df.iterrows():
        orig_file = row['original_file']     
        crop_path = row['saved_crop_path']
        if pd.notna(crop_path) and os.path.exists(crop_path):
            crop_bgr = cv2.imread(crop_path)
            if crop_bgr is not None:
                extracted_faces.append(crop_bgr)
                labels.append(label_to_int[row['manual_label']])
                
    X_train_final = np.array(extracted_faces)
    y_train_final = np.array(labels)
    
    # Simple augment for train
    import sys; sys.path.append('.')
    from dnn_try import augment
    X_train_final, y_train_final = augment(X_train_final, y_train_final)
    
    all_embeddings = extract_facenet_features_batch(X_train_final)

    # BUGFIX: Normalize embeddings BEFORE training
    all_embeddings_norm = normalize(all_embeddings, norm='l2')
    
    clf = SVC(kernel='rbf', C=3, gamma='scale', probability=True, class_weight='balanced', random_state=42)
    clf.fit(all_embeddings_norm, y_train_final)
    
    y_pred = []
    
    # We will just evaluate accuracy
    for idx, row in tqdm(test.iterrows(), total=len(test), desc="Testing"):
        img_bgr = np.load(f'datasets/test/test_{idx}.npy', allow_pickle=False)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        faces = face_detector.detect_faces(img_rgb)
        
        if not faces:
            y_pred.append(4)
            continue
            
        best_prob = -1
        best_pred = 4
        
        for (x, y, w, h, conf, sharp) in faces:
            crop = img_bgr[y:y+h, x:x+w]
            # No TTA for speed, just single embedding
            emb = extract_facenet_features_batch([crop])[0]
            emb_norm = normalize(emb.reshape(1, -1), norm='l2')
            
            probs = clf.predict_proba(emb_norm)[0]
            max_cls = np.argmax(probs)
            if probs[max_cls] > best_prob and max_cls != 4:
                best_prob = probs[max_cls]
                best_pred = max_cls
                
        # threshold
        if best_prob >= 0.3:
            y_pred.append(best_pred)
        else:
            y_pred.append(4)
            
    # Accuracy on first 300
    acc = accuracy_score(y_test_true, y_pred)
    print("Fixed Logic Accuracy (First 300 images):", acc)
    
if __name__ == "__main__":
    main()

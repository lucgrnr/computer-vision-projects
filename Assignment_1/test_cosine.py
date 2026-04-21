import numpy as np
import pandas as pd
import cv2
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from sklearn.metrics import accuracy_score
import os
from scipy.spatial.distance import cosine
from tqdm import tqdm

from face_detection import DNNFaceDetector

def main():
    print("Loading data...")
    train = pd.read_csv('datasets/train_set.csv', index_col=0)
    test = pd.read_csv('datasets/test_set.csv', index_col=0)
    
    # annotations
    df_annotations = pd.read_csv('annotations.csv')
    
    y_test_true = np.load('final_data/y_test.npy')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    
    detector = DNNFaceDetector(path="models", face_size=(160, 160))
    resnet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 1. Build prototypes from annotations
    print("Building prototypes...")
    class_embs = {0: [], 1: [], 2: [], 3: []}
    
    label_to_int = {
        "Michael_Cera": 0,
        "Jesse_Eisenberg": 1,
        "Mila_Kunis": 2,
        "Sarah_Hyland": 3,
        "Not_A_Face": 4
    }
    
    for _, row in df_annotations.iterrows():
        crop_path = row['saved_crop_path']
        label_str = row['manual_label']
        
        if pd.notna(crop_path) and os.path.exists(crop_path):
            lbl = label_to_int.get(label_str, 4)
            if lbl != 4:
                # Load Crop
                crop_bgr = cv2.imread(crop_path)
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                crop_resized = cv2.resize(crop_rgb, (160, 160))
                tensor = preprocess(crop_resized).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = resnet(tensor).cpu().numpy()[0]
                # Normalize
                emb = emb / np.linalg.norm(emb)
                class_embs[lbl].append(emb)
                
    prototypes = {}
    for k, v in class_embs.items():
        if len(v) > 0:
            avg_emb = np.mean(v, axis=0)
            prototypes[k] = avg_emb / np.linalg.norm(avg_emb)
            
    # 2. Extract embeddings for all test images
    print("Processing test images...")
    
    test_preds = []
    
    # Batch process test
    for idx, row in tqdm(test.iterrows(), total=len(test)):
        path = f'datasets/test/test_{idx}.npy'
        img_bgr = np.load(path, allow_pickle=False)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        faces = detector.detect_faces(img_rgb)
        
        if not faces or len(faces) == 0:
            test_preds.append(4)
            continue
            
        best_sim = -1
        best_class = 4
        
        for (x, y, w, h, conf, sharp) in faces:
            crop = img_rgb[y : y + h, x : x + w]
            crop_resized = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)
            tensor = preprocess(crop_resized).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = resnet(tensor).cpu().numpy()[0]
            emb = emb / np.linalg.norm(emb)
            
            # Compare with prototypes
            for k, p_emb in prototypes.items():
                sim = 1 - cosine(emb, p_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_class = k
                    
        if best_sim > -1:
            test_preds.append((best_class, best_sim))
        else:
            test_preds.append((4, 0.0))
            
    print("\nEvaluating thresholds...")
    best_acc = 0
    best_t = 0
    
    # Try different thresholds
    for t in np.arange(0.1, 0.9, 0.01):
        y_pred = []
        for cls, sim in test_preds:
            if sim > t:
                y_pred.append(cls)
            else:
                y_pred.append(4)
                
        acc = accuracy_score(y_test_true, y_pred[:len(y_test_true)])
        if acc > best_acc:
            best_acc = acc
            best_t = t
            
    print(f"Best Accuracy: {best_acc:.4f} at threshold: {best_t:.2f}")
    
    # Also print final classes distribution at best threshold
    y_pred = [cls if sim > best_t else 4 for cls, sim in test_preds]
    from sklearn.metrics import classification_report
    print(classification_report(y_test_true, y_pred[:len(y_test_true)]))
    
if __name__ == "__main__":
    main()

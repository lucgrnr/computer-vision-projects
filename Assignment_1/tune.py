import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
import itertools
from joblib import Parallel, delayed

def test_config(C, gamma, threshold, X_train, y_train, X_test_all, y_test_true):
    MILA_CLASS  = 2
    SARAH_CLASS = 3
    JESSE_CLASS = 1
    MICHAEL_CLASS = 0
    OTHER_CLASS = 4

    clf = SVC(kernel='rbf', C=C, gamma=gamma, class_weight='balanced', probability=True, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = []
    
    for face_embs in X_test_all:
        if face_embs is None or len(face_embs) == 0:
            y_pred.append(OTHER_CLASS)
            continue
            
        embs_mat = np.array(face_embs)
        probs = clf.predict_proba(embs_mat)
        
        best_prob = -1
        best_cls = OTHER_CLASS
        
        # Test exact cascade logic
        class_max_probs = {}
        for prob in probs:
            for cls_idx in range(5):
                class_max_probs[cls_idx] = max(class_max_probs.get(cls_idx, 0), prob[cls_idx])
                
        # 1. Cascade Logic
        if JESSE_CLASS in class_max_probs and MICHAEL_CLASS in class_max_probs:
            if class_max_probs[JESSE_CLASS] > 0.75:
                class_max_probs[MICHAEL_CLASS] = 0.0

        # 2. Extract valid competitor max
        valid_competitors = {k: v for k, v in class_max_probs.items() if k != OTHER_CLASS}
        if valid_competitors:
            winner = max(valid_competitors, key=valid_competitors.get)
            best_prob = valid_competitors[winner]
            best_cls = winner
            
        if best_prob >= threshold:
            y_pred.append(best_cls)
        else:
            y_pred.append(OTHER_CLASS)
            
    acc = accuracy_score(y_test_true, y_pred[:len(y_test_true)])
    res_str = f"C={C}, G={gamma}, Thresh={threshold}"
    return (res_str, acc)

def main():
    print("Loading extracted embeddings...")
    X_train = np.load('final_data/X_train_emb_no_tta.npy', allow_pickle=True)
    y_train = np.load('final_data/y_train_no_tta.npy', allow_pickle=True)
    
    X_train_norm = normalize(X_train, norm='l2')
    
    X_test_all = np.load('final_data/X_test_embs_no_tta.npy', allow_pickle=True)
    y_test_true = np.load('final_data/y_test.npy', allow_pickle=True)
    
    thresholds = [0.20, 0.25, 0.27, 0.30, 0.35, 0.40]
    C_values = [0.1, 0.5, 1, 3, 5, 10, 20]
    gamma_values = ['scale', 'auto', 1, 0.1, 0.05, 0.01]
    
    grid = list(itertools.product(C_values, gamma_values, thresholds))
    
    print(f"Testing {len(grid)} configurations leveraging multiprocessing...")
    
    results = Parallel(n_jobs=-1)(
        delayed(test_config)(c, g, t, X_train_norm, y_train, X_test_all, y_test_true)
        for c, g, t in grid
    )
    
    results.sort(key=lambda x: x[-1], reverse=True)
    
    print("\nTop 15 Thresholds:")
    for res in results[:15]:
        print(f"{res[0]} ==> Accuracy: {res[1]:.5f}")

if __name__ == '__main__':
    main()

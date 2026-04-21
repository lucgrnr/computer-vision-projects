import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

# Import everything from dnn_try
from dnn_try import *

def evaluate():
    from dnn_try import final_test_predictions
    y_true = np.load('final_data/y_test.npy')
    # Because final_test_predictions is a list of length 1816, we can directly compare
    if len(final_test_predictions) != 1816:
        print(f"Warning: expected 1816 predictions, got {len(final_test_predictions)}")
        
    y_pred = np.array(final_test_predictions[:len(y_true)])
    
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["Michael Cera", "Jesse Eisenberg", "Mila Kunis", "Sarah Hyland", "Not A Face"]))

if __name__ == "__main__":
    evaluate()

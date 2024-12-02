import myutils
from datetime import datetime
import sys
import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

mode = "sql"

if len(sys.argv) > 1:
    mode = sys.argv[1]

model_path = f'model/LSTM_model_{mode}.h5'
print(f"Loading model from: {model_path}")
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found.")
    sys.exit(1)

model = load_model(
    model_path,
    custom_objects={'f1_loss': myutils.f1_loss, 'f1': myutils.f1},
    compile=False  
)
print("Model loaded successfully.")

data_x_path = f'data/{mode}_dataset_finaltest_X'
data_y_path = f'data/{mode}_dataset_finaltest_Y'
print(f"Loading test data from: {data_x_path} and {data_y_path}")
if not os.path.exists(data_x_path) or not os.path.exists(data_y_path):
    print(f"Error: Test data files not found in path '{data_x_path}' or '{data_y_path}'.")
    sys.exit(1)

with open(data_x_path, 'rb') as fp:
    FinaltestX = pickle.load(fp)
with open(data_y_path, 'rb') as fp:
    FinaltestY = pickle.load(fp)
print("Test data loaded successfully.")

if len(FinaltestX) != len(FinaltestY):
    print(f"Error: Mismatch in number of samples: X({len(FinaltestX)}) != Y({len(FinaltestY)})")
    sys.exit(1)

valid_data = [
    (x, y) for x, y in zip(FinaltestX, FinaltestY)
    if isinstance(x, (list, np.ndarray)) and len(x) > 0
]
if len(valid_data) == 0:
    print("Error: No valid data found in the test set.")
    sys.exit(1)

FinaltestX, FinaltestY = zip(*valid_data)
max_length = 200
X_finaltest = pad_sequences(FinaltestX, maxlen=max_length, padding='post', truncating='post')
y_finaltest = np.array(FinaltestY)

y_finaltest = np.where(y_finaltest == 0, 1, 0)

print(f"{len(X_finaltest)} samples in the final test set.")
csum = np.sum(y_finaltest)
print(f"Percentage of vulnerable samples: {csum / len(X_finaltest) * 100:.2f}%")
print(f"Absolute number of vulnerable samples: {csum}")

print("Starting predictions...")
yhat_probs = model.predict(X_finaltest, verbose=0)
yhat_classes = (yhat_probs > 0.5).astype(int).flatten()  

if len(yhat_classes) != len(y_finaltest):
    print(f"Error: Prediction length ({len(yhat_classes)}) does not match label length ({len(y_finaltest)}).")
    sys.exit(1)

accuracy = accuracy_score(y_finaltest, yhat_classes)
precision = precision_score(y_finaltest, yhat_classes)
recall = recall_score(y_finaltest, yhat_classes)
F1Score = f1_score(y_finaltest, yhat_classes)

print("\nEvaluation Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {F1Score:.4f}")

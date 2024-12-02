import myutils
from datetime import datetime
import sys
import json
from keras.models import load_model
from keras.optimizers import Adam
from gensim.models import Word2Vec, KeyedVectors
import tensorflow as tf

threshold1 = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
threshold2 = [0.9999, 0.999, 0.99, 0.9, 0.5, 0.1, 0.01, 0.001, 0.0001]

mode = sys.argv[1] if len(sys.argv) > 1 else "sql"
nr = sys.argv[2] if len(sys.argv) > 2 else "1"
fine = sys.argv[3] if len(sys.argv) > 3 else ""

threshold = threshold2 if fine == "fine" else threshold1

mincount = 10
iterationen = 100
s = 200
w2v = f"word2vec_withString{mincount}-{iterationen}-{s}"
w2vmodel = f"w2v/{w2v}.model"

try:
    w2v_model = Word2Vec.load(w2vmodel)
except FileNotFoundError:
    print(f"Error: Word2Vec model not found at {w2vmodel}")
    sys.exit(1)

step = 5
fulllength = 200

try:
    model = load_model(f'model/CNNResNet_model_{mode}.h5', compile=False)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=myutils.f1_loss, metrics=[myutils.f1])
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

with open(f'data/plain_{mode}', 'r') as infile:
    data = json.load(infile)

print("Finished loading data")

identifying = myutils.getIdentifiers(mode, nr)
info = myutils.getFromDataset(identifying, data)
sourcefull = info[0]
commentareas = myutils.findComments(sourcefull)

myutils.getblocksVisual(mode, sourcefull, [], commentareas, fulllength, step, nr, w2v_model, model, threshold, "")

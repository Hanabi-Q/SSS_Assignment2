import myutils
import sys
import os.path
import json
from datetime import datetime
import random
import pickle
import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Add, Input, Flatten, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence
from keras import backend as K
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.utils import class_weight
import tensorflow as tf
from gensim.models import Word2Vec, KeyedVectors
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

mode = "remote_code_execution"

if (len(sys.argv) > 1):
    mode = sys.argv[1]

progress = 0
count = 0


restriction = [20000, 5, 6, 10]  
step = 5  
fulllength = 200  

mode2 = str(step) + "_" + str(fulllength)


mincount = 10  
iterationen = 100  
s = 200  
w = "withString"  

w2v = "word2vec_" + w + str(mincount) + "-" + str(iterationen) + "-" + str(s)
w2vmodel = "w2v/" + w2v + ".model"


if not (os.path.isfile(w2vmodel)):
    print("word2vec model is still being created...")
    sys.exit()

w2v_model = Word2Vec.load(w2vmodel)
word_vectors = w2v_model.wv

with open('data/plain_' + mode, 'r') as infile:
    data = json.load(infile)

now = datetime.now()  
nowformat = now.strftime("%H:%M")
print("finished loading. ", nowformat)

allblocks = []

for r in data:
    progress = progress + 1

    for c in data[r]:

        if "files" in data[r][c]:

            for f in data[r][c]["files"]:

                if not "source" in data[r][c]["files"][f]:
                    continue

                if "source" in data[r][c]["files"][f]:
                    sourcecode = data[r][c]["files"][f]["source"]

                    allbadparts = []

                    for change in data[r][c]["files"][f]["changes"]:

                        badparts = change["badparts"]
                        count = count + len(badparts)

                        for bad in badparts:
                            pos = myutils.findposition(bad, sourcecode)
                            if not -1 in pos:
                                allbadparts.append(bad)

                    if (len(allbadparts) > 0):
                        positions = myutils.findpositions(allbadparts, sourcecode)

                        blocks = myutils.getblocks(sourcecode, positions, step, fulllength)

                        for b in blocks:
                            allblocks.append(b)

keys = []

for i in range(len(allblocks)):
    keys.append(i)
random.shuffle(keys)

cutoff = round(0.7 * len(keys))  
cutoff2 = round(0.85 * len(keys))  

keystrain = keys[:cutoff]
keystest = keys[cutoff:cutoff2]
keysfinaltest = keys[cutoff2:]

print("cutoff " + str(cutoff))
print("cutoff2 " + str(cutoff2))

with open('data/' + mode + '_dataset_keystrain', 'wb') as fp:
    pickle.dump(keystrain, fp)
with open('data/' + mode + '_dataset_keystest', 'wb') as fp:
    pickle.dump(keystest, fp)
with open('data/' + mode + '_dataset_keysfinaltest', 'wb') as fp:
    pickle.dump(keysfinaltest, fp)

TrainX = []
TrainY = []
ValidateX = []
ValidateY = []
FinaltestX = []
FinaltestY = []

print("Creating training dataset... (" + mode + ")")
for k in keystrain:
    block = allblocks[k]
    code = block[0]
    token = myutils.getTokens(code)  
    vectorlist = []
    for t in token:  
        if t in word_vectors.key_to_index and t != " ":
            vector = w2v_model.wv[t]
            vectorlist.append(vector.tolist())
    TrainX.append(vectorlist) 
    TrainY.append(block[1])  

print("Creating validation dataset...")
for k in keystest:
    block = allblocks[k]
    code = block[0]
    token = myutils.getTokens(code)  
    vectorlist = []
    for t in token:  
        if t in word_vectors.key_to_index and t != " ":
            vector = w2v_model.wv[t]
            vectorlist.append(vector.tolist())
    ValidateX.append(vectorlist)  
    ValidateY.append(block[1])  

print("Creating finaltest dataset...")
for k in keysfinaltest:
    block = allblocks[k]
    code = block[0]
    token = myutils.getTokens(code)  
    vectorlist = []
    for t in token:  
        if t in word_vectors.key_to_index and t != " ":
            vector = w2v_model.wv[t]
            vectorlist.append(vector.tolist())
    FinaltestX.append(vectorlist)  
    FinaltestY.append(block[1])  

print("Train length: " + str(len(TrainX)))
print("Test length: " + str(len(ValidateX)))
print("Finaltesting length: " + str(len(FinaltestX)))
now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print("time: ", nowformat)

with open('data/' + mode + '_dataset_finaltest_X', 'wb') as fp:
    pickle.dump(FinaltestX, fp)
with open('data/' + mode + '_dataset_finaltest_Y', 'wb') as fp:
    pickle.dump(FinaltestY, fp)

def pad_sequences_2d(sequences, maxlen, vector_dim):
    padded = numpy.zeros((len(sequences), maxlen, vector_dim), dtype=numpy.float32)
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            padded[i] = seq[:maxlen]
        else:
            padded[i, :len(seq)] = seq
    return padded


def residual_block(input_tensor, filters, kernel_size, strides=1):
    x = Conv1D(filters, kernel_size, strides=strides, padding="same")(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)

    x = Conv1D(filters, kernel_size, strides=1, padding="same")(x)
    x = BatchNormalization()(x)

    shortcut = Conv1D(filters, 1, strides=strides, padding="same")(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x

def build_resnet_model(input_shape, num_classes=1):
    inputs = Input(shape=input_shape)

    x = Conv1D(64, 3, padding="same", strides=1)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = residual_block(x, filters=64, kernel_size=3)
    x = MaxPooling1D(pool_size=2)(x)

    x = residual_block(x, filters=128, kernel_size=3)
    x = MaxPooling1D(pool_size=2)(x)

    x = residual_block(x, filters=256, kernel_size=3)
    x = MaxPooling1D(pool_size=2)(x)

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs, outputs)


    return model


max_length = 200 
vector_dim = s
input_shape = (max_length, vector_dim)

X_train = pad_sequences_2d(TrainX, max_length, vector_dim)
X_test = pad_sequences_2d(ValidateX, max_length, vector_dim)
X_finaltest = pad_sequences_2d(FinaltestX, max_length, vector_dim)

y_train = numpy.array(TrainY)
y_test = numpy.array(ValidateY)
y_finaltest = numpy.array(FinaltestY)

print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
print(f"X_test shape: {X_test.shape}, dtype: {X_test.dtype}")
print(f"y_test shape: {y_test.shape}, dtype: {y_test.dtype}")
print(f"X_finaltest shape: {X_finaltest.shape}, dtype: {X_finaltest.dtype}")
print(f"y_finaltest shape: {y_finaltest.shape}, dtype: {y_finaltest.dtype}")


for i in range(len(y_train)):
    if y_train[i] == 0:
        y_train[i] = 1
    else:
        y_train[i] = 0

for i in range(len(y_test)):
    if y_test[i] == 0:
        y_test[i] = 1
    else:
        y_test[i] = 0

for i in range(len(y_finaltest)):
    if y_finaltest[i] == 0:
        y_finaltest[i] = 1
    else:
        y_finaltest[i] = 0

now = datetime.now()  
nowformat = now.strftime("%H:%M")
print("numpy array done. ", nowformat)

print(str(len(X_train)) + " samples in the training set.")
print(str(len(X_test)) + " samples in the validation set.")
print(str(len(X_finaltest)) + " samples in the final test set.")

csum = 0
for a in y_train:
    csum = csum + a
print("percentage of vulnerable samples: " + str(int((csum / len(X_train)) * 10000) / 100) + "%")

testvul = 0
for y in y_test:
    if y == 1:
        testvul = testvul + 1
print("absolute amount of vulnerable samples in test set: " + str(testvul))

max_length = fulllength

dropout = 0.2
neurons = 100
optimizer = "adam"
epochs = 20
batchsize = 128

now = datetime.now()  
nowformat = now.strftime("%H:%M")
print("Starting CNNResNet: ", nowformat)

print("Dropout: " + str(dropout))
print("Neurons: " + str(neurons))
print("Optimizer: " + optimizer)
print("Epochs: " + str(epochs))
print("Batch Size: " + str(batchsize))
print("max length: " + str(max_length))


model = build_resnet_model(input_shape=input_shape)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=myutils.f1_loss,
    metrics=[myutils.f1]
)
model.summary()


now = datetime.now()  
nowformat = now.strftime("%H:%M")
print("Compiled CNNResNet: ", nowformat)


class_weights_array = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=numpy.unique(y_train),
    y=y_train
)


class_weights = {cls: weight for cls, weight in zip(numpy.unique(y_train), class_weights_array)}

history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batchsize,
    class_weight=class_weights,
    validation_data=(X_test, y_test)
)

print("saving CNNResNet model " + mode + ". ", nowformat)
model.save('model/CNNResNet_model_' + mode + '.h5')  
print("\n\n")


for dataset in ["train", "test", "finaltest"]:
    print("Now predicting on " + dataset + " set (" + str(dropout) + " dropout)")

    if dataset == "train":
        yhat_classes = (model.predict(X_train, verbose=0) > 0.5).astype("int32")
        accuracy = accuracy_score(y_train, yhat_classes)
        precision = precision_score(y_train, yhat_classes)
        recall = recall_score(y_train, yhat_classes)
        F1Score = f1_score(y_train, yhat_classes)

    if dataset == "test":
        yhat_classes = (model.predict(X_test, verbose=0) > 0.5).astype("int32")
        accuracy = accuracy_score(y_test, yhat_classes)
        precision = precision_score(y_test, yhat_classes)
        recall = recall_score(y_test, yhat_classes)
        F1Score = f1_score(y_test, yhat_classes)

    if dataset == "finaltest":
        yhat_classes = (model.predict(X_finaltest, verbose=0) > 0.5).astype("int32")
        accuracy = accuracy_score(y_finaltest, yhat_classes)
        precision = precision_score(y_finaltest, yhat_classes)
        recall = recall_score(y_finaltest, yhat_classes)
        F1Score = f1_score(y_finaltest, yhat_classes)

    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print('F1 score: %f' % F1Score)
    print("\n")

now = datetime.now()  
nowformat = now.strftime("%H:%M")
'''
print("saving CNNResNet model " + mode + ". ", nowformat)
model.save('model/CNNResNet_model_' + mode + '.h5')  # creates a HDF5 file 'my_model.h5'
print("\n\n")
'''
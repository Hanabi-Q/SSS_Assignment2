import nltk
from gensim.models import Word2Vec
import os.path
import pickle
import sys
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

all_words = []

mode = "withString"
if len(sys.argv) > 1:
    mode = sys.argv[1]

print("Loading " + mode)
with open(f'w2v/pythontraining_{mode}_X', 'r') as file:
    pythondata = file.read().lower().replace('\n', ' ')

print("Length of the training file: " + str(len(pythondata)) + ".")
print("It contains " + str(pythondata.count(" ")) + " individual code tokens.")

if os.path.isfile(f'data/pythontraining_processed_{mode}'):
    with open(f'data/pythontraining_processed_{mode}', 'rb') as fp:
        all_words = pickle.load(fp)
    print("Loaded processed model.")
else:
    print("Now processing...")
    processed = pythondata
    all_sentences = nltk.sent_tokenize(processed)
    all_words = [nltk.word_tokenize(sent) for sent in all_sentences]
    with open(f'data/pythontraining_processed_{mode}', 'wb') as fp:
        pickle.dump(all_words, fp)
    print("Processed.\n")

for mincount in [10]:
    for iterationen in [100]:
        for s in [200]:
            print("\n\n" + mode + f" W2V model with min count {mincount}, {iterationen} iterations, size {s}")
            fname = f"w2v/word2vec_{mode}{mincount}-{iterationen}-{s}.model"
            wv_file = f"w2v/word2vec_{mode}{mincount}-{iterationen}-{s}.model.wv"

            if os.path.isfile(fname) and os.path.isfile(wv_file + ".vectors.npy"):
                print("Model and KeyedVectors already exist.")
                continue

            print("Calculating model...")
            model = Word2Vec(all_words, vector_size=s, min_count=mincount, epochs=iterationen, workers=12)

            print(f"Vocabulary size: {len(model.wv.index_to_key)}")

            model.save(fname)
            print(f"Model saved as {fname}")

            model.wv.save(wv_file)
            print(f"KeyedVectors saved as {wv_file}")

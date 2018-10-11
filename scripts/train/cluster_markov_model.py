import pickle
from sklearn.externals import joblib
import numpy as np


def main():
    parent_texts = pickle.load(open("preclean/parent_texts.pkl", "rb"))
    child_texts = pickle.load(open("preclean/children_texts.pkl", "rb"))
    cluster_model = joblib.load(open("models/cluster_comments.pkl", "rb"))
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    tfidf_parent_texts = tokenizer.texts_to_matrix(parent_texts,
                                                    mode='tfidf')
    children_tokened = tokenizer.texts_to_sequences(child_texts)
    predicts = cluster_model.predict(tfidf_parent_texts)
    cluster_markov_chain = {k : np.arange(0, 10000) for k in range(256)}
    for cluster, child in zip(predicts, children_tokened):
        for word in child:
            if word != 0:
                cluster_markov_chain[cluster][word] += 1
    p_chain = {k : (lambda d: d / d.sum())(v) for k, v in cluster_markov_chain.items()}
    pickle.dump(p_chain, open("models/cluster_markov_model.pkl", "wb"))

if __name__ == "__main__":
    main()

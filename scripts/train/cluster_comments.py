import pickle
from sklearn.cluster import KMeans
from sklearn.externals import joblib

def main():
    comment_texts = pickle.load(open("preclean/comment_texts.pkl", "rb"))
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    tfidf_comment_texts = tokenizer.texts_to_matrix(comment_texts,
                                                    mode='tfidf')

    model = KMeans(n_clusters=256)
    model.fit(tfidf_comment_texts)
    joblib.dump(model, 'models/cluster_comments.pkl')

if __name__ == "__main__":
    main()

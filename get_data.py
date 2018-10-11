import praw
import nltk
import numpy as np

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from nltk.sentiment import SentimentIntensityAnalyzer

# get a random submission from a given subreddit
def get_random_submission(subreddit_name, reddit):
    post = reddit.subreddit(subreddit_name).random()
    return post

# returns the text and the sentiment for each comment and
# for each comment, if the comment has a parent returns
# the parent and the comment as a child
def get_comments_and_parents(post):
    post.comments.replace_more(limit=None)
    comments = post.comments.list()
    vader_analyzer = SentimentIntensityAnalyzer()
    parents = []
    parents_scores = []
    rtn_comments = []
    scores = []
    for comment in comments:
        comment_parent = comment.parent()
        comment_scores = vader_analyzer.polarity_scores(comment.body)
        comment_scores_lst = [comment_scores["neg"], comment_scores["neu"],
                              comment_scores["pos"], comment_scores["compound"]]
        scores += [comment_scores_lst]
        try:
            parents += [comment_parent.body]
            parent_scores = vader_analyzer.polarity_scores(comment_parent.body)
            parent_scores_lst = [parent_scores["neg"], parent_scores["neu"],
                                parent_scores["pos"], parent_scores["compound"]]
            parents_scores += [parent_scores_lst]
            rtn_comments += [comment.body]
        except AttributeError:
            pass
    return ([comment.body for comment in comments], scores,
            [comment.score for comment in comments]),(parents, parents_scores, rtn_comments)


# tokenizes and pads the comment data and vectorizes the upvote score data.
def prepare_score_data(comments, scores, labels, max_comment_length, vocab_size=10000, tokenizer=None):
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(comments)
    token_comments = tokenizer.texts_to_sequences(comments)
    pad_comments = sequence.pad_sequences(token_comments, maxlen=max_comment_length)
    return pad_comments, np.array(scores), np.array(labels), tokenizer


# tokenizes and pads the parent data and creates a array of the
# first word in the child comment
def prepare_next_word_data(parents, parents_scores, comments, texts,
                           vocab_size=10000, max_comment_length=150, tokenizer=None):
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(texts)
    token_parents = tokenizer.texts_to_sequences(parents)
    token_child = tokenizer.texts_to_sequences(comments)
    pad_parents = sequence.pad_sequences(token_parents, maxlen=max_comment_length)
    labels = []
    for child in token_child:
        if len(child) >= 1:
            labels.append([child[0]])
        else:
            labels.append([0])
    return pad_parents, np.array(parents_scores), np.array(labels), tokenizer


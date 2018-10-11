import pickle
import numpy as np
from keras import layers, models


def make_upvote_model(score_data_shape, text_data_shape, neurons, vocab_size):
    score_input = layers.Input(shape=score_data_shape,
                               dtype='float32', name="score_input")
    text_input = layers.Input(shape=text_data_shape,
                              dtype='float32', name="text_input")
    embedding = layers.Embedding(vocab_size, neurons[0])(text_input)
    text_lstm = layers.LSTM(neurons[1])(embedding)
    concate = layers.concatenate([score_input, text_lstm])
    dense = layers.Dense(neurons[2], activation="relu")(concate)
    main_output = layers.Dense(1, activation="relu",
                               name="main_output")(dense)
    model = models.Model(inputs=[score_input, text_input],
                         outputs=[main_output])
    return model


def compile_model(model, opt, loss, metrics):
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=metrics
    )
    return model


def main():
    texts = np.load("cleaned/comment_texts.npy")
    scores = np.load("cleaned/comment_sentiment_scores.npy")
    upvotes = np.load("cleaned/comment_upvotes.npy")

    epochs = 3
    neurons = (16, 32, 64)
    model = make_upvote_model(scores.shape[1:],
                              texts.shape[1:], neurons, 10000)
    model = compile_model(model, "adam", "mse", ["mse"])
    hst = model.fit([scores, texts], [upvotes],
                epochs=epochs, batch_size=32, validation_split=0.1,
                shuffle=True, verbose=True).history

    model.save("models/predict_upvotes.h5")

if __name__ == "__main__":
    main()

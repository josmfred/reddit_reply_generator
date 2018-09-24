import numpy as np
from keras import layers, optimizers, Model
from keras.utils.np_utils import to_categorical


def make_word_predict_model(score_data_shape, text_data_shape,
                            neurons, vocab_size):
    score_input = layers.Input(shape=score_data_shape,
                               dtype='float32', name="score_input")
    text_input = layers.Input(shape=text_data_shape,
                              dtype='int32', name="text_input")
    embedding = layers.Embedding(vocab_size,
                                 neurons[0])(text_input)
    text_lstm = layers.LSTM(neurons[1])(embedding)
    concate = layers.concatenate([score_input, text_lstm])
    dense = layers.Dense(neurons[2], activation="relu")(concate)
    first_output = layers.Dense(vocab_size, activation="softmax",
                                name="first_output")(dense)
    model = Model(inputs=[score_input, text_input],
                  outputs=[first_output])
    return model


def compile_model(model, opt, loss, metrics):
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=metrics
    )
    return model


def main():
    first_word = np.load("child_first_word.npy")
    parents_scores = np.load("parent_sentiment_scores.npy")
    parents = np.load("parent_texts.npy")

    neurons = (32, 32, 64)
    model = make_word_predict_model(parents_scores.shape[1:],
                                    parents.shape[1:], neurons, 10000)
    compile_model(model, "adam", "categorical_crossentropy", [])

    first_word_cat = to_categorical(first_word, num_classes = 10000)

    model.fit([parents_scores, parents], [first_word_cat],
              epochs=2, validation_split=0.1)

    model.save("predict_first_word.h5")


if __name__ == "__main__":
    main()

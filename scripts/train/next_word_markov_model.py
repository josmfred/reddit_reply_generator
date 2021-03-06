import pickle
import numpy as np


# Creates a model that for each pair of words, stores, for each word,
# the count of the word given that follows the pair.
def train_next_word_markov_chain(texts, vocab_size=10000):
    chain = {}
    for text in texts:
        for pair in zip(zip(text, text[1:]), np.append(text[2:], -1)):
            if pair != ((0, 0), 0):
                chain[pair[0]] = chain.get(pair[0], {})
                chain[pair[0]][pair[1]] = chain[pair[0]].get(pair[1], 0) + 1
    # divides each link in the chain by the links total to create a probability
    # distribution for each pair in the bigram chain. I am sorry this is so bad.
    gen_p_chain = {key : {k: v / total for total in (sum(link.values()),)
                    for k, v in link.items()} for key, link in chain.items()}
    # count_to_p = (lambda word : word / sum(word))
    # gen_p_chain = {k : (count_to_p)(d) for k, d in chain.items()}
    return gen_p_chain


# Creates the dictionary of the form {pair -> {word : count}},
# and then converts this into a dictionary of the form,
# {pair -> {word: probability}}. We then save this model
def main():
    comment_texts = np.load("cleaned/comment_texts.npy")
    generator_chain = train_next_word_markov_chain(comment_texts)
    pickle.dump(generator_chain,
            open("models/generator_probability_chain.pkl",
                 'wb'))


if __name__ == "__main__":
    main()

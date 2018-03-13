from keras.preprocessing.text import Tokenizer
import numpy as np


class MathSymbolTokenizer:
    def __init__(self, chars):
        self.tokenizer = Tokenizer(char_level=True)
        self.tokenizer.fit_on_texts(chars)
        self.word2token = self.tokenizer.word_index
        self.token2word = {token: word for word, token in self.word2token.items()}

    def text2seq(self, text):
        return self.tokenizer.texts_to_sequences(text)

    def seq2text(self, seq):
        text = ""
        token_list = np.array(seq).reshape(-1,)
        for token in token_list:
            text += self.token2word[token]
        return text

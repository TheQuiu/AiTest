import os
import pickle

from loguru import logger
from tensorflow import keras


class AiTokenizer:
    def __init__(self, max_words=1000, max_len=1000):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = None
        logger.info("Tokenizer initialized")

        self.tokenizer_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'dataset', 'tokenizer.pickle'))

    def tokenize(self, data):
        tokenizer = keras.preprocessing.text.Tokenizer(num_words=1000)
        tokenizer.fit_on_texts(data)
        with open(self.tokenizer_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.tokenizer = tokenizer
        return self.tokenizer

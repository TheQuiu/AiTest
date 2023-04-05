import io
import os
import matplotlib.pyplot as plt

# Обучаем модель и сохраняем результаты обучения

from loguru import logger
from tensorflow import keras

from network.tokenizer.tokenizer import AiTokenizer


class Trainer:
    def __init__(self, ai_tokenizer: AiTokenizer):
        logger.info("Trainer initialized")
        self.dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'data.txt')
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'model.h5')
        self.dataset = []
        self.labels = []
        self.load_dataset()
        self.history = None
        self.ai_tokenizer = ai_tokenizer
        self.tokenizer = self.ai_tokenizer.tokenize(self.dataset)
        self.tokenizer_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'dataset', 'tokenizer.pickle'))
        self.graphs_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'dataset', 'graphs'))
    def load_dataset(self):
        with open(self.dataset_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('%')
                if len(parts) != 2:
                    continue
                sentence, label = parts[0].strip(), int(parts[1].strip())
                sentence = sentence.replace('ё', 'е')\
                    .replace("'", '')\
                    .replace('"', '')\
                    .replace('.', '')\
                    .replace(',', '')\
                    .replace('!', '')\
                    .replace('?', '')
                self.dataset.append(sentence.lower())
                self.labels.append(label)
            logger.info("Dataset loaded:\n Dataset: {}\n Labels: {}".format(self.dataset, self.labels))

    async def train(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        if os.path.exists(self.tokenizer_path):
            os.remove(self.tokenizer_path)
        self.tokenizer = self.ai_tokenizer.tokenize(self.dataset)
        x_train = self.tokenizer.texts_to_sequences(self.dataset)
        x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=1000)

        y_train = keras.utils.to_categorical(self.labels)
        model = keras.Sequential([
            keras.layers.Embedding(1000, 16, input_length=1000),
            keras.layers.Flatten(),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(2, activation='sigmoid')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.history = model.fit(x_train, y_train, epochs=13, batch_size=1)
        model.save(self.model_path)
        return "Model trained"

    def get_history(self):
        if self.history is None:
            return "Model not trained"
        else:
            return self.history.history

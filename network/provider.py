import os
import pickle

import numpy as np
from loguru import logger
from tensorflow import keras

from network.tokenizer.tokenizer import AiTokenizer


class Provider:
    def __init__(self, ai_tokenizer: AiTokenizer):
        self.model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'model.h5'))
        self.ai_tokenizer = ai_tokenizer
        if os.path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)
        else:
            self.model = None
            logger.info("Model not found, Please train model")
        self.threshold = 0.7  # пороговое значение для определения ответа
        self.tokenizer = None
        logger.info("Provider initialized")

    def make_predicts(self, data) -> np.ndarray:
        if self.model is None:
            return None
        else:
            self.tokenizer = self.ai_tokenizer.tokenize(data)
            input_data = self.tokenizer.texts_to_sequences(data)
            input_data = keras.preprocessing.sequence.pad_sequences(input_data, maxlen=1000)
            preds = self.model.predict(input_data)
            logger.debug(preds)
            return preds

    def predict_sentiment(self, sentence) -> str:
        data = [sentence]
        preds = self.make_predicts(data)
        if preds is None:
            return "Модель не обучена. Пожалуйста обучите модель."
        if np.max(preds) < self.threshold:
            return f"Не удалось определить ответ. Возможно я не знаю такое предложение. :/ \nPreds: {preds}"
        else:
            class_idx = np.argmax(preds)
            if class_idx == 0:
                sentiment = 'негативное'
            else:
                sentiment = 'позитивное'
            # вывод вероятности и предсказанного класса
            return f'Предложение "{sentence}" - {sentiment} ({preds[0][class_idx]:.2f})\nPreds: {preds}'

    def reload_model(self):
        self.model = None
        self.model = keras.models.load_model(self.model_path)
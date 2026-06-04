import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import pickle
import os

class SymbolRecognizer:
    def __init__(self, model_path=None, classes_path=None):
        """Загрузка модели и классов"""
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'model', 'best_model.h5')
        if classes_path is None:
            classes_path = os.path.join(os.path.dirname(__file__), 'model', 'class_names.pkl')
        
        self.model = load_model(model_path)
        with open(classes_path, 'rb') as f:
            self.class_names = pickle.load(f)
    
    def predict(self, image, return_confidence=True):
        """
        Распознает символ на изображении
        
        Args:
            image: PIL Image или путь к файлу
            return_confidence: возвращать ли уверенность
            
        Returns:
            символ (str), уверенность (float) - если return_confidence=True
            иначе только символ
        """
        if isinstance(image, str):
            image = Image.open(image).convert('L')
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('L')
        
        # Предобработка
        img = image.resize((64, 64))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 64, 64, 1)
        
        # Предсказание
        pred = self.model.predict(img_array, verbose=0)
        idx = np.argmax(pred[0])
        confidence = pred[0][idx]
        
        if return_confidence:
            return self.class_names[idx], confidence
        return self.class_names[idx]
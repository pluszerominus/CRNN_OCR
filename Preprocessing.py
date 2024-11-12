import math

import pandas as pd
import numpy as np
import cv2


# Класс для предобработки данных
class Data_Preprocessing():
    # Изменение размеров изображения
    def resize_img(self, img, shape_to=(128, 512)):
        # Вычисление минимального коэффициента масштабирования
        shrink_multiplayer = min(math.floor(shape_to[0] / img.shape[0] * 100) / 100,
                                      math.floor(shape_to[1] / img.shape[1] * 100) / 100)

        # Масштабирование изображения
        img = cv2.resize(img,
                         None,
                         fx=shrink_multiplayer,
                         fy=shrink_multiplayer,
                         interpolation=cv2.INTER_AREA)

        # Создание рамок на изображении
        img = cv2.copyMakeBorder(img, math.ceil(shape_to[0]/2) - math.ceil(img.shape[0]/2),
                                math.floor(shape_to[0]/2) - math.floor(img.shape[0]/2),
                                math.ceil(shape_to[1]/2) - math.ceil(img.shape[1]/2),
                                math.floor(shape_to[1]/2) - math.floor(img.shape[1]/2),
                                cv2.BORDER_CONSTANT, value=255)

        return img

    # Поворот изображения
    def rotate_img(self, img):
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # Установка адаптивное порогового значения
    def add_adaptiveThreshold(self, img, classification=False):
        new_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        if classification:
          return new_img
        else:
          return new_img.astype("bool")

    # Функция для объединения всех функций обработки изображений
    def all_img_preprocess(self, img):
        for func in [self.resize_img, self.rotate_img, self.add_adaptiveThreshold]:
          img = func(img)
        return img

    # Функция для кодирования определённого текста
    def label_to_num(self, label, alphabet):
        label_num = []
        # Обход всех символов текста
        for ch in label:
          # Формирование списка с индексами символов
          label_num.append(alphabet.find(ch))
        return np.array(label_num)

    # Кодирование всех данных
    def encode_text(self, texts):
        # Формирование алфавита символов из всех данных
        alphabet = ''.join(sorted(pd.Series(texts).apply(list).apply(pd.Series).stack().unique()))

        # Формирование матрицы для закодированных данных
        nums = np.ones([len(texts), max([len(text) for text in texts])], dtype='int64') * len(alphabet)
        # Заполнение матрицы закодированными данными
        for i, text in enumerate(texts):
          nums[i][:len(text)] = self.label_to_num(text, alphabet)

        return nums, alphabet
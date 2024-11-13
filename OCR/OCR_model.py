from keras.models import Sequential
from keras.layers import *


def ocr_model():
    model = Sequential()
    # Первый блок слоев модели включает в себя 7 сверточных слоев
    # Cо слоями пулинга для понижения размерности
    model.add(Conv2D(64, (5, 5), padding='same', activation=LeakyReLU(alpha=0.01), input_shape=(512, 128, 1)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (5, 5), padding='same', activation=LeakyReLU(alpha=0.01)))
    model.add(MaxPooling2D((1, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01)))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01)))

    model.add(Conv2D(256, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01)))
    model.add(MaxPooling2D((1, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(512, (3, 3), padding='same', activation=LeakyReLU(alpha=0.01)))
    model.add(MaxPooling2D((1, 2)))

    # Слой для формирование матрицы на основе изображения
    model.add(Reshape((128, 512)))

    # Два слоя LSTM для получения списка символов на основе матрицы
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))

    # Слой для вывода списка символов считанных с изображения в виде вектора
    model.add(Dense(len(alphabet) + 1, activation='softmax')) # +1 для ctc

    return model

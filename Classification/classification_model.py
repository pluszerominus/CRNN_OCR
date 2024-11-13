from keras.layers import *
from keras.models import Model

# Функция для создания модели
def class_model():
      # Входящий слой
      inputs = Input(shape=(128, 512, 1))

      # Два упорядоченных сверточных слоя
      conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
      conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)

      # Слой получающий средний сверточный слой
      pool1 = GlobalAveragePooling2D()(conv2)

      # Плотный слой для классификации
      dense1 = Dense(128, activation="relu")(pool1)
      # Dropout для исключения части нейронов при обучении
      drop1 = Dropout(0.5)(dense1)
      # Выходной слой с значением от 0 до 1, указывающий на пренадлежность к классу
      output = Dense(1, activation="sigmoid")(drop1)

      model = Model(inputs=inputs, outputs=output)

      # Компиляция модели
      model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

      return model
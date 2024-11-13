from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils import plot_model

import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import zipfile

from Preprocessing import Data_Preprocessing
from classification_model import class_model

# Чтение архива с данными
zf = zipfile.ZipFile('/content/drive/MyDrive/Task1/train.zip')
# Чтение архива с изображениями
zf_imgs = zipfile.ZipFile('/content/drive/MyDrive/Task1/imgs.zip')

# Чтение csv файлов
df_train = pd.read_csv(zf.open('train.csv'))

# Количество используемых для обучения данных
N_TRAIN = 20000

# Инициализация модуля предобработки данных
preprocess = Data_Preprocessing()

# Цикл формирования данных
x_data, y_data = [], []
for n in tqdm(df_train["name"].values[:N_TRAIN]):
    data = zf_imgs.read("imgs/"+n)
    # Преобразование изображений в числовой формат
    img = cv2.imdecode(np.frombuffer(data, np.uint8), 0)
    # Предобработка изображений
    img = preprocess.resize_img(img)
    img = preprocess.add_adaptiveThreshold(img)

    x_data.append(img)
    y_data.append(df_train[df_train['name'] == n]['label'].tolist()[0])

x_data = np.array(x_data)
y_data = np.array(y_data)

# Разделение данных на тренировочную и тестовую
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Настройка параметров сохранения модели
model_save_path = "drive/MyDrive/Class1.keras"
callback = ModelCheckpoint(model_save_path, monitor="loss", save_best_only=True, verbose=1)
earlystop = EarlyStopping(monitor="accuracy", patience=5)

# Инициализация модели
model = class_model()

# Вывод параметров модели
print(model.summary())
plot_model(model, 'my_first_model_with_shape_info.png', show_shapes=True)

# Обучение модели
history = model.fit(x_train, y_train,
                    batch_size=32, epochs = 10,
                    validation_data=(x_test, y_test),
                    callbacks=[callback, earlystop])

# Отрисовка графиков обучения модели
fig, ax = plt.subplots(2,1)


def plot_charts(ax_id,metrics):
      ax[ax_id].plot(history.history[f"{metrics}"][1:], label=f"train {metrics}")
      ax[ax_id].plot(history.history[f"val_{metrics}"][1:], label=f"test {metrics}")
      ax[ax_id].set_xlabel("Epoch")
      ax[ax_id].set_ylabel(f"{metrics}")
      ax[ax_id].set_title(f"{metrics} chart")


plot_charts(0, "loss")
plot_charts(1, "accuracy")
fig.tight_layout()
plt.show()
import zipfile
from tqdm import tqdm

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.losses import ctc
from keras.callbacks import EarlyStopping, ReduceLROnPlateau,ModelCheckpoint
from keras.optimizers import Nadam

from Preprocessing import Data_Preprocessing
from OCR_model import ocr_model
from cer_metric import CERMetric

zf = zipfile.ZipFile('/content/drive/MyDrive/Task1/train.zip')
df_train = pd.read_csv(zf.open('train.csv'))

df_train = df_train
df_train = df_train.sort_values("label", ascending=False)
df_train.head()

preprocess = Data_Preprocessing()

# Количество данных для обучения
N_TRAIN = 35_000
x_train, y_train = [], []
zf_imgs = zipfile.ZipFile('/content/drive/MyDrive/Task1/imgs.zip')


# Функция для разбиения
def create_data(x_train, y_train, class_type):
    for n in tqdm(df_train[df_train["label"] == class_type]["name"].values[:N_TRAIN]):
        data = zf_imgs.read("imgs/" + n)
        # Преобразование данных в числовой тип
        img = cv2.imdecode(np.frombuffer(data, np.uint8), 0)
        # Предобработка изображений
        img = preprocess.all_img_preprocess(img)
        x_train.append(img)
        y_train.append(df_train[df_train['name'] == n]['text'].tolist()[0])

    return x_train, y_train


# Формирование датасета
x_train, y_train = create_data(x_train, y_train, 0)
x_train, y_train = create_data(x_train, y_train, 1)
# Кодирование текста
y_train, alphabet = preprocess.encode_text(y_train)

# Разделение на тестовую и обучающую выборку
x_train, test_x, y_train, test_y = train_test_split(x_train,
                                                    y_train,
                                                    test_size=0.2,
                                                    random_state=42)

# Инициализация модели
model = ocr_model()

# Компиляция модели
model.compile(optimizer=Nadam(learning_rate=0.001, clipnorm=1.0), loss=ctc, metrics=[CERMetric()])
model.summary()

# Инициализация функций callback
model_save_path = "drive/MyDrive/Model_L0.keras"
callback = ModelCheckpoint(model_save_path, monitor="val_loss", save_best_only=True, verbose=1)
early_stop = EarlyStopping(patience=10, restore_best_weights=True, monitor='loss')
reduce_lr = ReduceLROnPlateau(factor=0.5, min_lr=1e-5, patience=4, monitor='loss')

# Обучение модели
history = model.fit(x_train, y_train,
                    epochs=40, batch_size=32,
                    callbacks=[early_stop, callback, reduce_lr],
                    verbose=1)

# Отрисовка графиков обучения модели
fig, ax = plt.subplots(2, 1)


def plot_charts(ax_id,metrics):
      ax[ax_id].plot(history.history[f"{metrics}"][1:], label=f"train {metrics}")
      ax[ax_id].plot(history.history[f"val_{metrics}"][1:], label=f"test {metrics}")
      ax[ax_id].set_xlabel("Epoch")
      ax[ax_id].set_ylabel(f"{metrics}")
      ax[ax_id].set_title(f"{metrics} chart")


plot_charts(0, "loss")
plot_charts(1, "CER_metric")
fig.tight_layout()
plt.show()

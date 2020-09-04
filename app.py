import os
import cv2
from model_loading import PneumoniaModel
import numpy as np
import matplotlib.pyplot as plt

model = PneumoniaModel()

dirpath = input('Введите путь к папке с файлами для анализа: ')


def load_data(filepath):
    images = []
    for file in os.listdir(filepath):
        img = cv2.imread(filepath + '\\' + file, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (150, 150), cv2.INTER_AREA)
        img = img.astype('float32') / 255

        images.append(img)

    images = np.array(images)
    return images


def make_prediction(data):
    predictions = model.stacked_prediction(data)
    for i in range(len(predictions)):
        if predictions[i] > 0.95:
            predictions[i] = 1
        else:
            predictions[i] = 0
    return predictions


def plot_graph(predicts, data):
    plt.figure(figsize=(7, 7))
    for i in range(len(predicts)):
        if predicts[i] == 1:
            plt.subplot(3, 3, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(data[i])
            plt.xlabel('Пневмония')
        else:
            plt.subplot(3, 3, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(data[i])
            plt.xlabel('Здоров')
    plt.show()


images = load_data(dirpath)
predictions = make_prediction(images)
plot_graph(predictions, images)

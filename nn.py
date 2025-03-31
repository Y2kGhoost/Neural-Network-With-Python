import numpy as np
from data import get_mnist
import matplotlib.pyplot as plt

images, labels = get_mnist()
w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
b_i_h = np.zeros((20, 1))
b_h_o = np.zeros((10, 1))

nn_Correct = 0
train_count = 5
rate = 0.01

def Train(images, labels, w_i_h, w_h_o, b_i_h, b_h_o, nn_Correct, train_count, rate):
    for train in range(train_count):
        for img, label in zip(images, labels):
            img.shape += (1,)
            label.shape += (1,)
            h_pre = w_i_h @ img + b_i_h
            h = 1 / (1 + np.exp(-h_pre))
            
            o_pre = w_h_o @ h + b_h_o
            o = 1 / (1 + np.exp(-o_pre))

            err = 1 / len(o) * np.sum((o - label), axis=0)
            nn_Correct += int(np.argmax(label) == np.argmax(o))

            delta_o = o - label
            w_h_o += -rate * delta_o @ np.transpose(h)
            b_h_o += -rate * delta_o

            delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
            w_i_h += -rate * delta_h @ np.transpose(img)
            b_i_h += -rate * delta_h

        print(f"Acc: {round((nn_Correct / images.shape[0]) * 100, 2)}%")
        nr_correct = 0

def Show():
    while True:
        index = int(input("Enter a number (0 - 59999): "))
        img = images[index]
        plt.imshow(img.reshape(28, 28), cmap="Grays")

        img.shape += (1,)
        h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
        h = 1 / (1 + np.exp(-h_pre))

        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))

        plt.title(f"It's {o.argmax()} :3")
        plt.show()

Train(images, labels, w_i_h, w_h_o, b_i_h, b_h_o, nn_Correct, train_count, rate)
Show()
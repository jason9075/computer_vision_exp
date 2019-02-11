import cv2
import matplotlib.pyplot as plt
import face_recognition as fr
import numpy as np


def display_image(img, cmap=None):
    plt.figure()
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_image, cmap=cmap)


def crop_image(img, center, padding):
    c_x = center[0]
    c_y = center[1]
    return img[c_y - padding:c_y + padding, c_x - padding:c_x + padding, :]


def main():
    bad = cv2.imread('dataset/public_face/bad/0_0a8256df-205d-4792-b342-2f4babda3859.jpg')
    good = cv2.imread('dataset/public_face/good/1_0c011c05-5ce8-4495-829b-40f354a768c5.jpg')

    h, w, _ = good.shape
    landmarks = fr.api.face_landmarks(good, [(0, w, h, 0)])[0]

    p37 = landmarks['left_eye'][0]
    p46 = landmarks['right_eye'][3]
    croped = crop_image(good, p37, 30)

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    display_image(croped)

    plt.show()


if __name__ == '__main__':
    main()

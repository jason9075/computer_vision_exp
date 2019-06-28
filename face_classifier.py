# paper https://arxiv.org/ftp/arxiv/papers/1708/1708.02694.pdf
import glob
import os
import shutil
import numpy as np
import cv2

BLUR_THR = 100
COVERAGE_THR = 0.5


def variance_of_laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()


def main():
    shutil.rmtree('good/', ignore_errors=True)
    shutil.rmtree('bad/', ignore_errors=True)
    os.makedirs('good/')
    os.makedirs('bad/')

    for imageFile in glob.glob("../collect_face/*.jpg"):
        print(imageFile)
        # image = cv2.imread("../collect_face/5a879aa0.jpg")
        image = cv2.imread(imageFile)
        image = cv2.resize(image, (100, 100))

        output_folder = "good/" if is_good_face(image) else "bad/"
        shutil.copyfile(imageFile, output_folder + os.path.basename(imageFile))

        # break


def is_good_face(img):
    if skin_coverage(img) < COVERAGE_THR:
        return False
    if variance_of_laplacian(img) < BLUR_THR:
        return False

    return True


def skin_coverage(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = np.dstack((img, img_hsv[:, :, :2]))  # channel: BGRHS

    # 20 < B and 40 < 40 and 95 < R and  and 0.23 and 160 < H < 180 and 40 < S < 174
    mask1 = cv2.inRange(img, (20, 40, 95, 0, 40), (255, 255, 255, 50, 174))
    mask2 = cv2.inRange(img, (20, 40, 95, 160, 40), (255, 255, 255, 180, 174))
    mask = (mask1 + mask2).astype(bool)
    # B < R and G < R and 15 < |R - G|
    mask3 = (img[:, :, 0] < img[:, :, 2]) * (15 < img[:, :, 2] - img[:, :, 1])
    mask = mask * mask3

    # cv2.imwrite('skin.jpg', mask*255)

    return mask.sum() / (mask.shape[0] * mask.shape[1])


if __name__ == '__main__':
    main()

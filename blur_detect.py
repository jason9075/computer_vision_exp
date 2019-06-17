import glob
import os
import shutil

import cv2

BLUR_THR = 100


def variance_of_laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()


def main():
    shutil.rmtree('not_blur/', ignore_errors=True)
    shutil.rmtree('blur/', ignore_errors=True)
    os.makedirs('not_blur/')
    os.makedirs('blur/')

    for imageFile in glob.glob("../collect_face/*.jpg"):
        image = cv2.imread(imageFile)
        image = cv2.resize(image, (100, 100))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        print(fm)

        output_folder = "not_blur/" if BLUR_THR < fm else "blur/"

        shutil.copyfile(imageFile, output_folder + os.path.basename(imageFile))


if __name__ == '__main__':
    main()

import cv2
import matplotlib.pyplot as plt
import face_recognition as fr
from imutils.face_utils.facealigner import FaceAligner
import dlib


def display_image(img, cmap=None):
    plt.figure()
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_image, cmap=cmap)


def crop_image(img, center, size):
    width, height = int(size[0] / 2), int(size[1] / 2)
    c_x = center[0]
    c_y = center[1]
    return img[c_y - height:c_y + height - 1, c_x - width:c_x + width - 1, :]


def calc_hog(aligner, detector, hog, image, crop_size, is_show=False):
    h, w, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 2)
    image = aligner.align(image, gray, rects[0])
    landmarks = fr.api.face_landmarks(image, [(0, w, h, 0)])[0]
    p37 = landmarks['left_eye'][0]
    p46 = landmarks['right_eye'][3]

    croped_37 = crop_image(image, p37, crop_size)
    croped_46 = cv2.flip(crop_image(image, p46, crop_size), 1)  # flip horizontal

    hist_left_eyes = hog.compute(croped_37)
    hist_right_eyes = hog.compute(croped_46)
    eye_score = cv2.compareHist(hist_left_eyes, hist_right_eyes, cv2.HISTCMP_CORREL)

    if is_show:
        display_image(croped_37)
        display_image(croped_46)
        plt.show()

    return eye_score


def main():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
    aligner = FaceAligner(predictor)

    good = cv2.imread('dataset/public_face/good/1_545ea37c-e55a-4628-9e0b-769aa7bb6750.jpg')
    bad = cv2.imread('dataset/public_face/bad/0_2f1fdaed-d3be-41e5-992e-70de6d0e366c.jpg')

    win_size = (64, 64)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, cell_size, cell_size, cell_size, nbins)

    good_score = calc_hog(aligner, detector, hog, good, win_size, is_show=True)
    bad_socre = calc_hog(aligner, detector, hog, bad, win_size)

    print("Test Result: G: {:.2f}, B: {:.2f}".format(good_score, bad_socre))


if __name__ == '__main__':
    main()

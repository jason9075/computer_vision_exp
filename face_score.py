import glob

import cv2
import face_recognition as fr
import matplotlib.pyplot as plt
import numpy as np
import copy


def display_image(img, cmap=None):
    plt.figure()
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_image, cmap=cmap)


def crop_image(img, center, size):
    width, height = int(size[0] / 2), int(size[1] / 2)
    c_x = center[0]
    c_y = center[1]
    return img[c_y - height:c_y + height - 1, c_x - width:c_x + width - 1, :]


def calc_hog(hog, image, crop_size, is_show=False):
    h, w, _ = image.shape

    landmarks = fr.api.face_landmarks(image, [(0, w, h, 0)])[0]
    image, matrix = align(image, landmarks)

    p37 = np.dot(matrix, landmarks['left_eye'][0] + (1,)).astype(int)
    p46 = np.dot(matrix, landmarks['right_eye'][3] + (1,)).astype(int)

    croped_37 = crop_image(image, p37, crop_size)
    croped_46 = cv2.flip(crop_image(image, p46, crop_size), 1)  # flip horizontal
    hist_left_eyes = hog.compute(copy.deepcopy(croped_37))
    hist_right_eyes = hog.compute(copy.deepcopy(croped_46))

    eye_score = cv2.compareHist(hist_left_eyes, hist_right_eyes, cv2.HISTCMP_CORREL)

    if is_show:
        display_image(croped_37)
        display_image(croped_46)
        plt.show()

    return eye_score


def align(image, landmarks):
    desired_left_eye = (0.35, 0.35)
    desired_face_width = 256
    desired_face_height = 256

    right_eye_pts = np.asarray(landmarks['right_eye'])
    left_eye_pts = np.asarray(landmarks['left_eye'])

    # compute the center of mass for each eye
    left_eye_center = right_eye_pts.mean(axis=0).astype("int")
    right_eye_center = left_eye_pts.mean(axis=0).astype("int")

    # compute the angle between the eye centroids
    d_y = right_eye_center[1] - left_eye_center[1]
    d_x = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(d_y, d_x)) - 180

    # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    desired_right_eye_x = 1.0 - desired_left_eye[0]

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((d_x ** 2) + (d_y ** 2))
    desired_dist = (desired_right_eye_x - desired_left_eye[0])
    desired_dist *= desired_face_width
    scale = desired_dist / dist

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                   (left_eye_center[1] + right_eye_center[1]) // 2)

    # grab the rotation matrix for rotating and scaling the face
    transform_m = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    # update the translation component of the matrix
    t_x = desired_face_width * 0.5
    t_y = desired_face_height * desired_left_eye[1]
    transform_m[0, 2] += (t_x - eyes_center[0])
    transform_m[1, 2] += (t_y - eyes_center[1])

    # apply the affine transformation
    (w, h) = (desired_face_width, desired_face_height)
    output = cv2.warpAffine(image, transform_m, (w, h),
                            flags=cv2.INTER_CUBIC)

    # return the aligned face
    return output, transform_m


def main():
    win_size = (64, 64)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, cell_size, cell_size, cell_size, nbins)

    good = cv2.imread('dataset/public_face/good/1_545ea37c-e55a-4628-9e0b-769aa7bb6750.jpg')
    bad = cv2.imread('dataset/public_face/bad/0_2f1fdaed-d3be-41e5-992e-70de6d0e366c.jpg')

    good_score = calc_hog(hog, good, win_size, is_show=False)
    bad_socre = calc_hog(hog, bad, win_size, is_show=False)

    print("Test Result: G: {:.2f}, B: {:.2f}".format(good_score, bad_socre))

    thr = 0.5
    image_names = glob.glob("dataset/public_face/good/*.jpg")

    correct = 0
    for image_name in image_names:
        print("reading: {}".format(image_name))
        image = cv2.imread(image_name)
        score = calc_hog(hog, image, win_size)
        if thr < score:
            correct += 1

    print("Good face: {}/{} rate:{:.2f}".format(correct, len(image_names), correct / len(image_names)))

    image_names = glob.glob("dataset/public_face/bad/*.jpg")

    correct = 0
    for image_name in image_names:
        image = cv2.imread(image_name)
        score = calc_hog(hog, image, win_size)
        if score < thr:
            correct += 1

    print("bad face: {}/{} rate:{:.2f}".format(correct, len(image_names), correct / len(image_names)))


if __name__ == '__main__':
    main()

import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_image(img, cmap=None):
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_image, cmap=cmap)


def main():
    image_file = 'no_person.jpg'
    desk_poly = [(0, 480), (850, 180), (1110, 210), (350, 800), (0, 626)]
    desk_color = (255, 0, 255, 255)
    dilated_color = (0, 255, 0, 255)
    dilated_size = (300, 300)
    dilated_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dilated_size)
    person_location = [(220, 80), (390, 360)]
    person_color = (255, 0, 0, 255)

    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

    desk_mask = np.zeros(image.shape, dtype=np.uint8)
    desk_range = np.array([desk_poly], dtype=np.int32)
    cv2.fillPoly(desk_mask, desk_range, desk_color)

    dilated_mask = cv2.dilate(desk_mask, dilated_kernel)
    detect_mask = cv2.bitwise_xor(desk_mask, dilated_mask)
    mask = cv2.inRange(detect_mask, desk_color, desk_color)
    detect_mask[0 < mask] = dilated_color

    merge_image = cv2.addWeighted(image, 1, desk_mask, 0.5, 0)
    merge_image = cv2.addWeighted(merge_image, 1, detect_mask, 0.5, 0)

    person_mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.rectangle(person_mask, person_location[0], person_location[1], dilated_color, -1)
    person_area = cv2.bitwise_and(detect_mask, person_mask)
    crop_area = person_area[person_location[0][1]:person_location[1][1], person_location[0][0]:person_location[1][0], ]
    percent = crop_area.any(axis=-1).sum() / (crop_area.shape[0] * crop_area.shape[1]) *100
    print("overlay: {:.1f}%".format(percent))

    cv2.rectangle(merge_image, person_location[0], person_location[1], person_color, 4)

    display_image(merge_image)
    plt.show()


if __name__ == '__main__':
    main()

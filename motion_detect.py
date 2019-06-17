import cv2


def main():
    resource = cv2.VideoCapture(0)
    last_frame = None

    while True:
        ret, frame = resource.read()

        if not ret:
            break

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (21, 21), 0)

        if last_frame is None:
            last_frame = frame
            continue

        frame_delta = cv2.absdiff(last_frame, frame)

        print(f'Diff Percentage:{frame_delta.sum() / (frame_delta.size * 255) * 100:.1f}%')
        cv2.imshow('frame', frame_delta)
        cv2.waitKey(1)

        last_frame = frame

    cv2.destroyAllWindows()
    cv2.waitKey(4)


if __name__ == '__main__':
    main()

import gestureTracking as gesture
import cv2 as cv
import time


def main():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    p_time = 0

    print("Opening Window")

    while True:
        success, image = cap.read()
        if not success:
            break

        key = cv.waitKey(10)
        if key == 27:
            break

        image = cv.flip(image, 1)  # mirror the display

        results = gesture.process_image(image)
        gesture.draw_hands(image, results, key)

        # calculate the time between frames
        c_time = time.time()
        fps = 1/(c_time - p_time)
        p_time = c_time
        # print the fps
        cv.putText(image, f'FPS: {int(fps)}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv.LINE_AA)

        cv.imshow("Output", image)
        if cv.getWindowProperty("Output", cv.WND_PROP_VISIBLE) < 1:
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

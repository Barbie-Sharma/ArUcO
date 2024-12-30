import cv2 as cv
import os

def captureImagesFromCamera():
    root = os.getcwd()
    saveDir = os.path.join(root, 'my_chessboards')
    os.makedirs(saveDir, exist_ok=True)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera")
        return

    print("Press 'c' to capture an image and 'q' to quit.")
    imgCount = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame")
            break

        cv.imshow("Camera", frame)
        key = cv.waitKey(1) & 0xFF

        if key == ord('c'):
            imgPath = os.path.join(saveDir, f'chessboard_{imgCount}.jpg')
            cv.imwrite(imgPath, frame)
            imgCount += 1
            print(f"Image {imgCount} saved to {imgPath}")
        
        elif key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    print(f"All images saved in {saveDir}")

if __name__ == '__main__':
    captureImagesFromCamera()

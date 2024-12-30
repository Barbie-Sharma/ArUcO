import cv2
import numpy as np

calibration_data = np.load('calibration.npz')

camera_matrix = calibration_data['camMatrix']
dist_coeffs = calibration_data['distCoeff']

print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

def pose_estimation(frame): 
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None:
        for i in range(0, len(ids)):
            #rotation vector, translation vector
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, camera_matrix, dist_coeffs)
            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.01)
    return frame


capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    isTrue, frame = capture.read()
    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    output = pose_estimation(gray)

    cv2.imshow('Estimated Pose', output)

    cv2.waitKey(1)

capture.release()
cv2.destroyAllWindows()

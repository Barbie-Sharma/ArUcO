import cv2
import numpy as np

#open camera 
capture = cv2.VideoCapture(0)

while True:
    isTrue, frame = capture.read()

    #press 'd' to delete video screen
    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

    #convert frame to grayscale 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #generate ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()

    #create ArUco detector 
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    #to detect id
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        for i in range(len(ids)):
            corner = corners[i][0]

            top_left = (int(corner[0][0]) - 4, int(corner[0][1]) - 4) 
            bottom_right = (int(corner[2][0]) + 4, int(corner[2][1]) + 4) 

            cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), thickness=4)
    
            print(f"Detected marker ID: {ids}")

    cv2.imshow('Detected Markers', frame)
    cv2.waitKey(1)

capture.release()
cv2.destroyAllWindows()


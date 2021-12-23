git import cv2
import numpy as np
import matplotlib.pylab as plt

def nothing(x):
    pass

cap = cv2.VideoCapture('Bonus_Test.mp4')


#cv2.namedWindow("Tracking")
#cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
#cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
#cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
#cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
#cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
#cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)
while cap.isOpened():
    ret, frame = cap.read()
    #framen = cv2.resize(frame, (500, 300))
    #cv2.imshow("framen", frame)
    height = frame.shape[0]
    width = frame.shape[1]
    #cropped_image = frame[0:height, width:height]
    #cv2.imshow("crop", cropped_image)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # l_h = cv2.getTrackbarPos("LH", "Tracking")
    # l_s = cv2.getTrackbarPos("LS", "Tracking")
    # l_v = cv2.getTrackbarPos("LV", "Tracking")

    # u_h = cv2.getTrackbarPos("UH", "Tracking")
    # u_s = cv2.getTrackbarPos("US", "Tracking")
    # u_v = cv2.getTrackbarPos("UV", "Tracking")

    #l_b = np.array([l_h, l_s, l_v])
    #u_b = np.array([u_h, u_s, u_v])

    l_b = np.array([0, 0, 110])
    u_b = np.array([190, 60, 190])

    mask = cv2.inRange(hsv, l_b, u_b)
    maskn = cv2.resize(mask, (500, 300))

    res = cv2.bitwise_and(frame, frame, mask=mask)
    resn = cv2.resize(res, (500, 300))
    framen = cv2.resize(frame, (500, 300))

    cv2.imshow("frame", framen)

    #cv2.imshow("mask", maskn)
    cv2.imshow("res", resn)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2 as cv
import mediapipe as mp

mp_hands = mp.solutions.hands #following and identify hands
mp_draw = mp.solutions.drawing_utils #drawing landmark and link node 

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame")
        break
    rgbFrame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = hands.process(rgbFrame) #read frames to process
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            ) 
            point = hand_landmarks.landmark[0]
            print(f"{point.x} : {point.y}")

    cv.imshow("frame", cv.flip(frame, 1))
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


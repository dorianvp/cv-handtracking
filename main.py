import numpy as np
import cv2
from collections import deque
import mediapipe as mp
from utils.utils_v2 import get_idx_to_coordinates, rescale_frame

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

import easyocr

reader = easyocr.Reader(
    ["en"]
)

def main():
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    hand_landmark_drawing_spec = mp_drawing.DrawingSpec(
        thickness=1, circle_radius=5, color=(200, 0, 0)
    )
    hand_connection_drawing_spec = mp_drawing.DrawingSpec(
        thickness=3, circle_radius=10, color=(100, 255, 100)
    )
    cap = cv2.VideoCapture(0)
    pts = deque(maxlen=16)
    boxes = []
    word_pos = [
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
    ]
    count = 0
    while cap.isOpened():
        count += 1
        idx_to_coordinates = {}
        ret, image = cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_hand = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results_hand.multi_hand_landmarks:
            for hand_landmarks in results_hand.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    # landmark_drawing_spec=hand_landmark_drawing_spec,
                    connection_drawing_spec=hand_connection_drawing_spec,
                )
                idx_to_coordinates = get_idx_to_coordinates(image, results_hand)
        if 8 in idx_to_coordinates:
            pts.appendleft(idx_to_coordinates[7])  # Index Finger
            new_point = (
                int(
                    idx_to_coordinates[8][0]
                    + (idx_to_coordinates[8][0] - idx_to_coordinates[7][0]) / 2
                ),
                int(
                    idx_to_coordinates[8][1]
                    + (idx_to_coordinates[8][1] - idx_to_coordinates[7][1]) / 2
                ),
            )
            image = cv2.circle(
                image, new_point, radius=5, color=(0, 0, 255), thickness=1
            )
            if count == 15:
                count = 0
                boxes.clear()
                result = reader.readtext(image)

                if len(result) > 0:
                    for i in range(0, len(result)):
                        boxes.append(
                            [
                                (
                                    result[i][0][0][0],
                                    result[i][0][0][1],
                                ),
                                (
                                    result[i][0][1][0],
                                    result[i][0][1][1]
                                ),
                                (
                                    result[i][0][2][0],
                                    result[i][0][2][1]                                    
                                ),
                                (
                                    result[i][0][3][0],
                                    result[i][0][3][1]                                   
                                ),
                                result[i][1]
                            ]
                        )
            
            if len(boxes) > 0:
                for i in range(0, len(boxes)):
                    image = cv2.line(
                        img=image,
                        pt1=[int(boxes[i][0][0]), int(boxes[i][0][1])],
                        pt2=[int(boxes[i][1][0]), int(boxes[i][1][1])],
                        thickness=1,
                        color=(0, 255, 0),
                    )
                    image = cv2.line(
                        img=image,
                        pt1=[int(boxes[i][1][0]), int(boxes[i][1][1])],
                        pt2=[int(boxes[i][2][0]), int(boxes[i][2][1])],
                        thickness=1,
                        color=(0, 255, 0),
                    )
                    image = cv2.line(
                        img=image,
                        pt1=[int(boxes[i][2][0]), int(boxes[i][2][1])],
                        pt2=[int(boxes[i][3][0]), int(boxes[i][3][1])],
                        thickness=1,
                        color=(0, 255, 0),
                    )
                    image = cv2.line(
                        img=image,
                        pt1=[int(boxes[i][3][0]), int(boxes[i][3][1])],
                        pt2=[int(boxes[i][0][0]), int(boxes[i][0][1])],
                        thickness=1,
                        color=(0, 255, 0),
                    )

                    # Check if point inside box
                    # Calculate rectangle area and check if triangles add up to the same amount
                    area = int((boxes[i][1][0] - boxes[i][0][0]) * (boxes[i][2][1] - boxes[i][1][1]))
                    t1 = abs(
                        (boxes[i][1][0] * boxes[i][0][1] - boxes[i][0][0] * boxes[i][0][1]) + (new_point[0] * boxes[i][1][1] - boxes[i][1][0] * new_point[1]) + (boxes[i][0][0] * new_point[1] - new_point[0] * boxes[i][0][1])
                    ) / 2
                    t2 = abs(
                        (boxes[i][2][0] * boxes[i][1][1] - boxes[i][1][0] * boxes[i][1][1]) + (new_point[0] * boxes[i][2][1] - boxes[i][2][0] * new_point[1]) + (boxes[i][1][0] * new_point[1] - new_point[0] * boxes[i][1][1])
                    ) / 2
                    t3 = abs(
                        (boxes[i][3][0] * boxes[i][2][1] - boxes[i][2][0] * boxes[i][2][1]) + (new_point[0] * boxes[i][3][1] - boxes[i][3][0] * new_point[1]) + (boxes[i][2][0] * new_point[1] - new_point[0] * boxes[i][2][1])
                    ) / 2
                    t4 = abs(
                        (boxes[i][0][0] * boxes[i][3][1] - boxes[i][3][0] * boxes[i][3][1]) + (new_point[0] * boxes[i][0][1] - boxes[i][0][0] * new_point[1]) + (boxes[i][3][0] * new_point[1] - new_point[0] * boxes[i][3][1])
                    ) / 2

                    if area > int(t1 + t2 + t3 + t4) / 2:
                        print(boxes[i][4])
                        # print('Area', area)
                        # print('Triangles', t1 + t2 + t3 + t4)
                        print()

        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            thick = int(np.sqrt(len(pts) / float(i + 1)) * 4.5)
            # cv2.line(image, pts[i - 1], pts[i], (0, 0, 200), thick)

        cv2.imshow("Res", rescale_frame(image, percent=100))
        if cv2.waitKey(5) & 0xFF == 27:
            break
        if count == 15:
                count = 0
    hands.close()
    cap.release()


if __name__ == "__main__":
    main()

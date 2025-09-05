import sys
import cv2
import mediapipe as mp

# mediapipe의 Hand Landmark 를 추출을 위한 옵션
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode = False, #고정이미지 아님
    max_num_hands = 3,
    min_detection_confidence = 0.5, #감지 확률 0.5 이상만
    min_tracking_confidence = 0.5 # 트래킹 확률 0.5이상만
)

vcap = cv2.VideoCapture(0)

while True:
    ret, frame = vcap.read()
    if not ret:
        print("웹캠이 작동하지 않습니다.")
        sys.exit()

    ###### Hands Landmark 설정하기 ########
    # 손 감지하기
    results = hands.process(frame)

    # 그리기
    if results.multi_hand_landmarks:
        print(len(results.multi_hand_landmarks)) # 탐지된 손의 갯수
        for hand_landmarks in results.multi_hand_landmarks:
            height, width, _ = frame.shape

            points = hand_landmarks.landmark # 21 개의 손 좌표
            show_points = [0, 5, 6, 7, 8, 9, 10, 11, 12]
            for index, landmark in enumerate(points):
                if index not in show_points:
                    continue
                print(landmark.x, landmark.y)
            
                point_x = int(landmark.x * width)
                point_y = int(landmark.y * height)

                cv2.circle(frame, (point_x, point_y), 5, (0,0,255), 2)
            
            lines = [
                    # (0,1), (1,2), (2, 3), (3, 4), 
                     (0, 5), (5, 6), (6, 7), (7, 8), 
                     (5, 9), 
                     (9, 10), (10, 11), (11, 12), 
                    #  (9, 13), 
                    #  (13, 14), (14, 15), (15, 16), 
                    #  (13, 17), 
                    #  (0, 17), (17, 18), (18, 19), (19, 20),
                     ]
            
            for start, end in lines:
                point_1 = hand_landmarks.landmark[start]
                point_x1 = int(point_1.x * width)
                point_y1 = int(point_1.y * height)

                point_2 = hand_landmarks.landmark[end]
                point_x2 = int(point_2.x * width)
                point_y2 = int(point_2.y * height)

                cv2.line(frame, (point_x1, point_y1), (point_x2, point_y2), (255,0,0), 1, lineType=cv2.LINE_8)

            # 자동 그리기
            # mp_drawing.draw_landmarks(
            #     frame,
            #     hand_landmarks,
            #     mp_hands.HAND_CONNECTIONS,
            #     mp_drawing_styles.get_default_hand_landmarks_style(),
            #     mp_drawing_styles.get_default_hand_connections_style()
            # )
    # 좌우 반전
    flipped_frame = cv2.flip(frame, 1)
    #contrast_frame = 255 - flipped_frame

    rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)

    # 손 그리기 설정
    frame.flags.writeable = False
    
    # 화면 띄우기
    cv2.imshow("webcam", flipped_frame)

    # 꺼지는 조건
    key = cv2.waitKey(1)
    if key == 27:
        break

vcap.release()
cv2.destroyAllWindows()
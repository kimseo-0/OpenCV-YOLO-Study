import sys 
import cv2 
import os 
import csv 
import mediapipe as mp 

# mediapipe hand landmark 옵션
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 저장할 데이터 설정 
file_path = './data/hand_data.csv'
image_path = './data/images/'

if not os.path.exists(file_path):
    with open(file_path, "w") as file:
        writer = csv.writer(file)

# 카메라 설정 
vcap = cv2.VideoCapture(0)

while True:
    ret, frame = vcap.read()
    if not ret:
        print("카메라가 작동하지 않습니다.")
        sys.exit()
    
    # 좌우반전
    frame = cv2.flip(frame, 1)

    ################################
    ###### Hands Landmark 추출 ######
    ################################
    # 손 그리기 설정
    frame.flags.writeable = True 

    # 손 감지
    results = hands.process(frame)

    # 추출 및 그리기 
    if results.multi_hand_landmarks:
        one_hand = results.multi_hand_landmarks[0]

        # 좌표 모으기 
        height, width, _ = frame.shape
        landmarks = []
        for landmark in one_hand.landmark:
            ## 좌표 모으기
            landmarks.extend([landmark.x, landmark.y, landmark.z])

            ## 그리기
            point_x = int(landmark.x * width)
            point_y = int(landmark.y * height)
            cv2.circle(frame, (point_x, point_y), 5, (0,0,255), 2)
        
        key = cv2.waitKey(1) # ASCII 코드
        if key == ord("1"):
            # 정답 라벨 추가
            landmarks.append("rock")
            # 데이터 추가
            with open(file_path, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(landmarks)
                # cv2.imwrite(image_path + '/rock/1.jpg', frame)
                cv2.putText(frame, "Save Rock Data!", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
        elif key == ord("2"):
            # 정답 라벨 추가
            landmarks.append("Sissors")
            # 데이터 추가
            with open(file_path, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(landmarks)
                cv2.putText(frame, "Save Sissors Data!", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
        elif key == ord("3"):
            # 정답 라벨 추가
            landmarks.append("Paper")
            # 데이터 추가
            with open(file_path, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(landmarks)
                cv2.putText(frame, "Save Paper Data!", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
    ################################

    # 화면 띄우기
    cv2.imshow("webcam", frame)

    # 꺼지는 조건
    key = cv2.waitKey(1) # ASCII 코드
    if key == 27: # ESC 
        break

vcap.release()
cv2.destroyAllWindows()
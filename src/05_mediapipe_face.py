# 05_mediapipe_face.py
import sys 
import cv2 
import mediapipe as mp 

# mediapipe의 Face Landmark를 추출을 위한 옵션
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh 

face_detection = mp_face_detection.FaceDetection(
    model_selection = 1,  # 0: 근거리, 1: 원거리
    min_detection_confidence=0.5
)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    min_detection_confidence=0.5,
    refine_landmarks=True
)


vcap = cv2.VideoCapture(0)

while True:
    ret, frame = vcap.read()
    if not ret:
        print("카메라가 작동하지 않습니다.")
        sys.exit()
    
    # 좌우 반전
    frame = cv2.flip(frame, 1)

    #################################
    
    ########### Face 찾기 ###########
    # # 얼굴 그리기 설정
    # frame.flags.writeable = True

    # # 얼굴 감지
    # detection_results = face_detection.process(frame)

    # # 얼굴 그리기 
    # if detection_results.detections:
    #     for detection in detection_results.detections:
    #         mp_drawing.draw_detection(frame, detection)
    #################################

    ###### Face Landmark 그리기 ######
    # 얼굴 그리기 설정
    frame.flags.writeable = True

    # 얼굴 Mesh 감지 
    mesh_results = face_mesh.process(frame)

    # 눈
    left_eye = [
            133,
            173,
            157,
            158,
            159,
            160,
            161,
            246,
            33,
            130,
            7,
            163,
            144,
            145,
            153,
            154,
            155,
        ]
    right_eye = [
        362,
        382,
        381,
        380,
        374,
        373,
        390,
        249,
        359,
        263,
        466,
        388,
        387,
        386,
        385,
        384,
        398,
    ]

    # 코
    nose_tip = [168, 245, 128, 114, 217, 198, 209, 49, 64, 98, 97, 2, 326, 327, 278, 360, 420, 399, 351]

    # 입
    top_lip = [61, 185, 40, 39, 37, 0, 267, 270]
    bottom_lip = [146, 91, 181, 84, 17, 314, 405, 321]

    # 눈썹
    left_eyebrow = [336, 296, 334, 293, 276, 283, 282, 295, 285]
    right_eyebrow = [107, 66, 105, 63, 70, 53, 52, 65, 66]

    # 얼굴 윤곽
    face_contour = [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
        10,
    ]

    if mesh_results.multi_face_landmarks:
        for index, face_landmarks in enumerate(mesh_results.multi_face_landmarks):
            #################################
            ########### 직접 그리기 ###########
            # height, width, _ = frame.shape 

            # landmarks = face_landmarks.landmark
            # for idx in right_eye:
            #     x = landmarks[idx].x
            #     y = landmarks[idx].y

            #     point_x = int(x * width)
            #     point_y = int(y * height)

            #     cv2.circle(frame, (point_x, point_y), 2, (255,0,0), 2)

            ########### 자동 그리기 ###########
            # mp_drawing.draw_landmarks(
            #     frame,
            #     face_landmarks,
            #     mp_face_mesh.FACEMESH_TESSELATION,
            #     mp_drawing.DrawingSpec(
            #         color=(0,255,0),
            #         thickness=1,
            #         circle_radius=1
            #     )
            # )

            ######## 입 사이 거리 구하기 ########
            height, width, _ = frame.shape 

            landmarks = face_landmarks.landmark
            landmark1 = landmarks[13]
            landmark2 = landmarks[14]

            point_x1 = int(landmark1.x * width)
            point_x2 = int(landmark2.x * width)
            point_y1 = int(landmark1.y * height)
            point_y2 = int(landmark2.y * height)

            cv2.circle(frame, (point_x1, point_y1), 3, (0,0,255), 2)
            cv2.circle(frame, (point_x2, point_y2), 3, (0,0,255), 2)

            cv2.line(frame, (point_x1, point_y1), (point_x2, point_y2), (0,0,255), 3)
            
            ########### 거리 계산하기 ###########
            distance = ((point_x1 - point_x2)**2 + (point_y1 - point_y2)**2) ** 0.5
            print(f"거리: {distance}")

            if distance < 50:
                print("입 더 벌리세요!!!")
    #################################
    #################################

    # 화면 띄우기
    cv2.imshow("webcam", frame)

    # 꺼지는 조건
    key = cv2.waitKey(1)
    if key == 27:  # ESC 버튼을 누르면 종료
        break 

vcap.release()
cv2.destroyAllWindows()
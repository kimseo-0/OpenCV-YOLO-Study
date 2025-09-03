# uv add yt_dlp
from ultralytics import YOLO
import yt_dlp
import cv2

# -------------------
# 모델 불러오기
# -------------------
model = YOLO("yolo11n.pt")


# -------------------
# 비디오 불러오기
# -------------------
video_url = "https://youtu.be/S5nsDT5oU90"

# yt_dlp 옵션 설정
ydl_opts = {
    "format": "best[ext=mp4][protocol=https]/best",
    'quite': True,
    'no_warnings': True
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(video_url, download=False)
    stream_url = info_dict['url']

vcap = cv2.VideoCapture(stream_url)

while True:
    if not vcap.isOpened():
        print("비디오를 열 수 없습니다.")
        break
    
    ret, frame = vcap.read()
    if not ret:
        print("비디오 프레임을 읽어올 수 없습니다.")
        break

    # 여기에 YOLO 예측을 넣을 수 있을 듯
    results = model(frame)
    result = results[0]
    boxes = result.boxes
    print(boxes.cls)

    cv2.imshow("Youtube Video", frame)

    # 꺼지는 조건
    key = cv2.waitKey(1)
    if key == 27:  # ESC 버튼을 누르면 종료
        break 

vcap.release()
cv2.destroyAllWindows()
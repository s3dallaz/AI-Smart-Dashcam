from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent

video_path = BASE_DIR / "videos" / "test_video.mp4"

model = YOLO("yolov8n.pt")

results = model(str(video_path), show=True , save=True)

print("Detection finished!")
import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
import torchvision
from torchvision import transforms

# Cấu hình
video_path = "./input_data/istockphoto-1311408017-640_adpp_is.mp4"
output_path = "./output/output_video.mp4"
conf_threshold = 0.5
tracking_class = 0  # None: track all
model_path = "pickpocket_model_9955phantram.pth"

# Khởi tạo DeepSort
tracker = DeepSort(max_age=30)

# Khởi tạo YOLOv9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectMultiBackend(weights="yolov9-c.pt", device=device, fuse=True)
model = AutoShape(model)

# Load class names
with open("./input_data/classes.names") as f:
    class_names = f.read().strip().split('\n')

colors = np.random.randint(0, 255, size=(len(class_names), 3))

# Khởi tạo model phát hiện móc túi
def load_pickpocket_model(model_path):
    model = torchvision.models.video.r3d_18(pretrained=False)
    num_classes = 2
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)

pickpocket_model = load_pickpocket_model(model_path)

# Hàm chuẩn bị video cho model phát hiện móc túi
def prepare_video(frames):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
    ])
    video_tensor = torch.stack([transform(frame) for frame in frames])
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    return video_tensor.unsqueeze(0).to(device)

# Hàm dự đoán hành vi móc túi
def predict_pickpocket(model, video_tensor):
    with torch.no_grad():
        output = model(video_tensor)
        probabilities = torch.softmax(output, dim=1)
        pickpocket_probability = probabilities[0, 1].item()
        if pickpocket_probability < 0.80:
            prediction = 0
        else:
            prediction = torch.argmax(probabilities, dim=1).item()
    return prediction, pickpocket_probability

# Khởi tạo VideoCapture
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Khởi tạo VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_buffer = {}
pickpocket_status = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)

    detect = []
    for detect_object in results.pred[0]:
        label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)

        if tracking_class is None:
            if confidence < conf_threshold:
                continue
        else:
            if class_id != tracking_class or confidence < conf_threshold:
                continue

        detect.append([[x1, y1, x2-x1, y2-y1], confidence, class_id])

    tracks = tracker.update_tracks(detect, frame=frame)

    # Thay đổi cách định nghĩa màu sắc
    colors = np.random.randint(0, 255, size=(len(class_names), 3)).tolist()

    # Trong vòng lặp chính
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            color = colors[class_id]

            # Lưu frame cho mỗi track
            if track_id not in frame_buffer:
                frame_buffer[track_id] = []
            frame_buffer[track_id].append(frame[y1:y2, x1:x2])

            # Nếu đủ 16 frame, phân tích hành vi móc túi
            if len(frame_buffer[track_id]) == 16:
                video_tensor = prepare_video(frame_buffer[track_id])
                prediction, probability = predict_pickpocket(pickpocket_model, video_tensor)
                pickpocket_status[track_id] = (prediction, probability)
                frame_buffer[track_id] = []  # Reset buffer

            # Vẽ bounding box và label
            label = "{}-{}".format(class_names[class_id], track_id)
            if track_id in pickpocket_status:
                prediction, probability = pickpocket_status[track_id]
                if prediction == 1:
                    label += f" PICKPOCKET ({probability:.2f})"
                    color = (0, 0, 255)  # Đỏ cho móc túi
                else:
                    color = (0, 255, 0)  # Xanh lá cho bình thường

            # Đảm bảo color là tuple của 3 số nguyên
            color = tuple(map(int, color))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video đã được lưu tại: {output_path}")
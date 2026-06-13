import cv2
import numpy as np
import onnxruntime as ort
import pygame
import time
import os
from collections import defaultdict
from picamera2 import Picamera2

SWAP_RB_PREVIEW = True       # 预览时交换红蓝通道
MODEL_NEEDS_BGR = True       # 模型需要BGR输入

CLASS_NAMES = [
    'pothole', 'obstacle', 'barricade', 'person', 'vehicle', 'crosswalk',
    'tree', 'pole', 'stairs', 'red', 'green'
]

ALWAYS_ALERT_CLASSES = {'pothole', 'red', 'green', 'barricade', 'obstacle'}  # 始终播报的类别

INPUT_WIDTH = 416
INPUT_HEIGHT = 416
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
CLOSE_AREA_RATIO = 0.05                                      # 近距离判定面积比
LEFT_RATIO = 1 / 3.0
RIGHT_RATIO = 2 / 3.0
VOICE_DIR = "voice_prompts"
PERSON_COUNT_THRESHOLD = 5                                   # 行人过多时不播报单人
ALERT_COOLDOWN = 4.0                                         # 同类告警冷却时间(秒)

last_alert_time = defaultdict(float)

CLASS_CN_MAP = {
    'pothole': '坑洞', 'obstacle': '障碍物', 'barricade': '路障',
    'person': '行人', 'vehicle': '车辆', 'crosswalk': '斑马线',
    'tree': '树木', 'pole': '杆子', 'stairs': '台阶',
    'red': '红灯', 'green': '绿灯'
}

POS_CN_MAP = {
    'left': '左前方', 'center': '前方', 'right': '右前方'
}

def init_onnx_model(onnx_path):
    return ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])  # 加载ONNX模型

def init_pygame_mixer():  # 初始化音频播放器
    try:
        pygame.mixer.pre_init(48000, -16, 2, 2048)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(8)
        print("音频初始化成功")
    except Exception as e:
        print(f" 音频初始化失败: {e}")
        raise

def xywh2xyxy(x):  # 中心宽高转左上右下
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def nms(boxes, scores, conf_thres, iou_thres):  # 非极大值抑制，返回保留的索引
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres)
    return indices.flatten() if len(indices) > 0 else []

def postprocess(output, img_shape, conf_thres=0.5, iou_thres=0.4):  # 后处理：解码、缩放、NMS
    predictions = output[0]
    boxes_xywh = predictions[:, :4]
    obj_conf = predictions[:, 4:5]
    class_probs = predictions[:, 5:]
    class_scores = obj_conf * class_probs
    class_ids = np.argmax(class_scores, axis=1)
    scores = np.max(class_scores, axis=1)

    mask = scores > conf_thres
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes_xywh) == 0:
        return []

    boxes_xyxy = xywh2xyxy(boxes_xywh)
    scale_x = img_shape[1] / INPUT_WIDTH
    scale_y = img_shape[0] / INPUT_HEIGHT
    boxes_xyxy[:, [0, 2]] *= scale_x
    boxes_xyxy[:, [1, 3]] *= scale_y

    keep = nms(boxes_xyxy, scores, conf_thres, iou_thres)

    detections = []
    for idx in keep:
        detections.append([
            int(boxes_xyxy[idx, 0]), int(boxes_xyxy[idx, 1]),
            int(boxes_xyxy[idx, 2]), int(boxes_xyxy[idx, 3]),
            float(scores[idx]), int(class_ids[idx])
        ])
    return detections

def get_position(x_center, img_width):  # 根据水平中心位置返回方位
    ratio = x_center / img_width
    if ratio < LEFT_RATIO:
        return "left"
    elif ratio < RIGHT_RATIO:
        return "center"
    else:
        return "right"

def is_close(box, img_area):  # 判断目标是否近距离
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    return (box_area / img_area) > CLOSE_AREA_RATIO

def play_alert(class_name, position):  # 播放对应的语音文件
    cn_class = CLASS_CN_MAP.get(class_name, class_name)
    cn_pos = POS_CN_MAP.get(position, position)
    for name in [f"{cn_class}_{cn_pos}", f"{cn_class}{cn_pos}"]:
        filepath = os.path.join(VOICE_DIR, f"{name}.ogg")
        if os.path.exists(filepath):
            try:
                sound = pygame.mixer.Sound(filepath)
                sound.play()
                print(f"播放语音：{name}.ogg")
                return True
            except Exception as e:
                print(f"播放失败：{filepath}，错误：{e}")
                return False
    print("语音文件不存在")
    return False

def should_alert(class_name, position, box, img_area, person_count=0):  # 告警规则判断
    now = time.time()
    key = class_name
    if now - last_alert_time[key] < ALERT_COOLDOWN:
        print(f"冷却中，跳过播报：{class_name} ({ALERT_COOLDOWN - (now - last_alert_time[key]):.1f}s)")
        return False
    if class_name in ALWAYS_ALERT_CLASSES:
        return True
    if class_name == 'person' and person_count <= PERSON_COUNT_THRESHOLD:
        return False
    return is_close(box, img_area)

def main():
    onnx_path = "ONNX/best2.onnx"
    if not os.path.exists(onnx_path):
        print(f"模型文件不存在：{onnx_path}")
        return
    if not os.path.exists(VOICE_DIR):
        print(f"语音目录不存在：{VOICE_DIR}")
        return

    session = init_onnx_model(onnx_path)  # 加载ONNX模型
    init_pygame_mixer()

    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "BGR888"},
            controls={"AwbEnable": True, "AwbMode": 0}       # 自动白平衡
        )
        picam2.configure(config)
        picam2.start()
        print("摄像头已启动 (BGR888 + 自动白平衡)")
    except Exception as e:
        print(f"摄像头启动失败：{e}")
        return

    prev_time = time.time()
    print("语音导盲系统启动，按 'q' 退出...")
    print(f"预览红蓝交换：{SWAP_RB_PREVIEW}，模型BGR输入：{MODEL_NEEDS_BGR}")

    try:
        while True:
            raw = picam2.capture_array()  # 原始图像 (480,640,3) BGR

            if SWAP_RB_PREVIEW:
                frame = raw[:, :, [2, 1, 0]]   # 预览通道交换 (BGR->RGB)
            else:
                frame = raw

            img_h, img_w = frame.shape[:2]
            img_area = img_w * img_h

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-10)
            prev_time = curr_time

            # 构造模型输入，确保通道顺序与MODEL_NEEDS_BGR设置一致
            if MODEL_NEEDS_BGR:
                model_input = raw if SWAP_RB_PREVIEW else raw
            else:
                model_input = frame if SWAP_RB_PREVIEW else raw[:, :, [2, 1, 0]]

            blob = cv2.dnn.blobFromImage(model_input, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT),
                                         swapRB=False, crop=False)
            inputs = {session.get_inputs()[0].name: blob}
            outputs = session.run(None, inputs)

            detections = postprocess(outputs[0], (img_h, img_w), CONF_THRESHOLD, NMS_THRESHOLD)
            person_count = sum(1 for det in detections if CLASS_NAMES[det[5]] == 'person')

            for det in detections:
                x1, y1, x2, y2, conf, cls_id = det
                class_name = CLASS_NAMES[cls_id]
                x_center = (x1 + x2) // 2
                position = get_position(x_center, img_w)
                if should_alert(class_name, position, (x1, y1, x2, y2), img_area, person_count):
                    last_alert_time[class_name] = time.time()
                    play_alert(class_name, position)
                    print(f"播报触发：{class_name} {position}")

            display = frame.copy()
            for det in detections:
                x1, y1, x2, y2, conf, cls_id = det
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, f"{CLASS_NAMES[cls_id]} {conf:.2f}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.putText(display, f"FPS: {fps:.1f}", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Preview", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print("程序退出")

if __name__ == "__main__":
    main()

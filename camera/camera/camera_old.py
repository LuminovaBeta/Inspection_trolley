import sys
import os
sys.path.append('/mnt/disk/elf/camera_ws/src/camera/camera')
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import cv2
from queue import Queue
import threading
import time
import numpy as np
from control import pid_control, control
from rknnlite.api import RKNNLite

from control import rotate
from control import servo_control

# 队列
info = Queue(maxsize=1)

# 图像尺寸(剪切)
W_img = 640
H_img = 640
image_area = W_img * H_img
center_point = (W_img / 2, H_img / 2)

# 拍照阈值
Catch_max = 0.80
Catch_min = 0.50
clock = 1

# PID 参数
Servo_Kp = 0.00005              #0.00005
Servo_Ki = 0
Servo_Kd = 0.000002              #0.000002
dead_zone = 0.004           #0.004

# 舵机和升降机
limit_servo = (0.025, 0.125)
limit_lift = (-1600, 1600)
servo_angle = 0.08
lift_angle = 0

rknn_model = None


class ImagePublisher(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.image_pub = self.create_publisher(
            Image,
            '/camera/detected_image',
            10
        )

  
def load_rknn_model():
    global rknn_model
    rknn_model = RKNNLite()
    if rknn_model.load_rknn('/mnt/disk/elf/camera_ws/src/best.rknn') != 0:
        exit(-1)
    if rknn_model.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
        exit(-1)


def nms_boxes(boxes, scores, iou_threshold=0.4):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def decode_output(outputs, conf_thres=0.25):
    pred = outputs[0]
    pred = pred[0].transpose(1, 0)
    scores = pred[:, 4]
    mask = scores > conf_thres
    boxes = pred[mask][:, :4]
    scores = scores[mask]
    if len(boxes) == 0:
        return np.array([]), np.array([])
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return boxes_xyxy, scores

def main_job():
    rclpy.init()
    node = ImagePublisher()
    load_rknn_model()
    bridge = CvBridge()
    cap = cv2.VideoCapture('/dev/video21')

    Last_error_S = 0
    Integral_S = 0
    Last_error_L = 0
    Integral_L = 0
    servo_angle = 0.055
    lift_angle = 0

    if not cap.isOpened():
        node.get_logger().error("无法打开摄像头")
        exit(1)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0)
            
            ret, img = cap.read()
            if not ret:
                node.get_logger().warning("未读取到摄像头帧")
                continue

            frame = cv2.resize(img, (640, 640))
            input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_img = np.expand_dims(input_img, axis=0)
            input_img = input_img.astype(np.uint8)
            outputs = rknn_model.inference(inputs=[input_img])
            if outputs is None:
                print("⚠️ RKNN 推理失败，跳过该帧")
                continue

            boxes, scores = decode_output(outputs, conf_thres=0.25)
            if boxes.shape[0] > 0:
                keep = nms_boxes(boxes, scores, 0.4)
                for i in keep:
                    if scores[i] < 0.4:
                        continue
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img.shape[1], x2)
                    y2 = min(img.shape[0], y2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{scores[i]:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.imwrite('/mnt/disk/elf/camera_ws/src/camera/camera/output.jpg', frame)

                    target_point = ((x1 + x2) / 2, (y1 + y2) / 2)

                    error = (target_point[0] - center_point[0], target_point[1] - center_point[1])
                    try:
                        info.put_nowait(error)
                    except:
                        try:
                            info.get_nowait()
                        except:
                            pass
                        info.put_nowait(error)


            
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        node.destroy_node()
        rclpy.shutdown()


# 0.025, 0.130
def control_job():
    global servo_angle
    Last_error_S = 0
    Integral_S = 0
    global dead_zone

    servo_control(servo_angle)
    t0 = 0
    while True:
        if not info.empty():
            error = info.get()
            t1 = time.time()
            if t1 - t0 >= 0.3:
                Out_S, Last_error_S, Integral_S = pid_control(
                    0, error[0], Servo_Kp, Servo_Ki, Servo_Kd,
                    Last_error_S, Integral_S, 0.3, 0.01)
                #print('1:',Out_S)

                if abs(Out_S) > dead_zone:
                    servo_angle += Out_S
                    servo_angle = max(min(servo_angle, limit_servo[1]), limit_servo[0])
                    servo_control(servo_angle)
                    print(f"{Out_S:.8f}") 
                t0 = t1

            if error[1] > 50:
                rotate(steps=200, freq_hz=16000, direction=1)
                time.sleep(0.05)
                print('down')
            elif error[1] <= -50:
                rotate(steps=200, freq_hz=16000, direction=0)
                time.sleep(0.05)
                print('up')
            

        else:
            time.sleep(0.01)




def main():


    t1 = threading.Thread(target=main_job, daemon=True)
    t2 = threading.Thread(target=control_job, daemon=True)

    t1.start()
    t2.start()

    try:
        t1.join()
        t2.join()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
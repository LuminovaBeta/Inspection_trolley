import sys
import os
sys.path.append('/mnt/disk/elf/camera_ws/src/camera/camera')
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
import cv2
from queue import Queue
import threading
import time
import numpy as np
from control import pid_control, control
from rknnlite.api import RKNNLite
from cv_bridge import CvBridge

from control import rotate
from control import servo_control
import select

node = None

# 队列
info = Queue(maxsize=1)

img = None
p0 = None


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
# yolo_success = False
prev_gray = None
clock = 1  # 拍照锁 默认上锁
num = 0  # 用于计数拍照次数

GPIO_PIN = 123
GPIO_PATH = f"/sys/class/gpio/gpio{GPIO_PIN}"
flag_in = False

t0 = 0

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.br = CvBridge()
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.yolo_success = False
        self.publisher_2 = self.create_publisher(Image, '/camera/image_processed', 10)
        self.img_pub = self.create_publisher(
            Int32,
            '/camera/pub',
            10
        )

        self.subscription = self.create_subscription(
            Int32,                                      
            '/camera/img_r', 
            self.callback,
            10
        )
    
    def send(self, data: int):
        msg = Int32()
        msg.data = data
        self.img_pub.publish(msg)

    def send_img(self, frame):
        img_msg = self.br.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher_.publish(img_msg)

    def send_img_p(self, frame):
        img_msg = self.br.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher_2.publish(img_msg)




    def callback(self, msg):        #此时解除锁，相机可以拍摄
        try:
            global clock
            global t0
            self.get_logger().info("接收成功")
            global p0
            clock = 0
            t0 = 0
            p0 = None

        except Exception as e:
            self.get_logger().error(f"Error in callback: {e}")


def init_gpio_input(pin):
    gpio_path = f"/sys/class/gpio/gpio{pin}"
    if not os.path.exists(gpio_path):
        with open("/sys/class/gpio/export", "w") as f:
            f.write(str(pin))
        time.sleep(0.1)  # 等 sysfs 文件创建

    with open(f"{gpio_path}/direction", "w") as f:
        f.write("in")

    with open(f"{gpio_path}/edge", "w") as f:
        f.write("rising")

def load_rknn_model():
    global rknn_model
    rknn_model = RKNNLite()
    if rknn_model.load_rknn('/mnt/disk/elf/camera_ws/src/best.rknn') != 0:
        exit(-1)
    if rknn_model.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
        exit(-1)


#NMS处理
def nms_boxes(boxes, scores, iou_threshold=0.4):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]      #置信度由高到低排列
    keep = []
    while order.size > 0:
        i = order[0]            #当前最大分数的索引
        keep.append(i)          #保留该框

        #计算与剩下框的交集
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        #IOU计算
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]

         # 找出 IOU 小于阈值的框，保留下次处理
        order = order[inds + 1]
    return keep


def decode_output(outputs, conf_thres=0.25):
    pred = outputs[0]
    pred = pred[0].transpose(1, 0)              #[x, y, w, h, score, class1, class2, ...]
    scores = pred[:, 4]
    mask = scores > conf_thres                  #筛除低置信框
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

def crop_center_1200x1200(image):
    h, w = image.shape[:2]
    crop_w, crop_h = 1200, 1200
    start_x = (w - crop_w) // 2  # 计算水平方向起点
    start_y = (h - crop_h) // 2  # 计算垂直方向起点
    return image[start_y:start_y + crop_h, start_x:start_x + crop_w]


def main_job():
    load_rknn_model()
    #设置摄像头
    os.system("v4l2-ctl -d /dev/video21 --set-fmt-video=width=1920,height=1200,pixelformat=MJPG --set-parm=90")
    cap = cv2.VideoCapture('/dev/video21')
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    cap.set(cv2.CAP_PROP_FPS, 90)
    #参数
    global Flag
    global img
    global Img
    # global yolo_success
    global p0
    global prev_gray

    Last_error_S = 0
    Integral_S = 0
    Last_error_L = 0
    Integral_L = 0
    servo_angle = 0.055
    lift_angle = 0
    last_detect_time = time.time()

    if not cap.isOpened():
        node.get_logger().error("无法打开摄像头")
        exit(1)

    # 初始化光流跟踪状态
    prev_gray = None
    p0 = None
    has_detection = False

    try:
        while rclpy.ok():

            rclpy.spin_once(node, timeout_sec=0)
            
            
            if clock == 1:      #如果主机不发信号就不执行
                continue

            yolo_success = False

            ret, img = cap.read()
            if not ret:
                node.get_logger().warning("未读取到摄像头帧")
                continue
            cv2.imwrite('/mnt/disk/elf/camera_ws/src/camera/camera/output_2.jpg', img)
            #图片处理
            frame = crop_center_1200x1200(img)
            frame = cv2.resize(img, (640, 640))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_img = np.expand_dims(input_img, axis=0).astype(np.uint8)
            outputs = rknn_model.inference(inputs=[input_img])
            

            if outputs is not None:
                boxes, scores = decode_output(outputs, conf_thres=0.25)
                if boxes.shape[0] > 0:
                    keep = nms_boxes(boxes, scores, 0.4)
                    for i in keep:
                        if scores[i] < 0.65:                 #低于0.45的舍弃
                            yolo_success = False
                            continue
                        x1, y1, x2, y2 = boxes[i].astype(int)
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)


                        # draw
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{scores[i]:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        cv2.imwrite('/mnt/disk/elf/camera_ws/src/camera/camera/output.jpg', frame)
                        node.send_img_p(frame)
                        


                        # 计算
                        target_point = ((x1 + x2) / 2, (y1 + y2) / 2)
                        error = (target_point[0] - center_point[0], target_point[1] - center_point[1])
                        try:
                            info.put_nowait(error)
                        except:
                            try: info.get_nowait()
                            except: pass
                            info.put_nowait(error)

                        # 初始化光流角点
                        mask = np.zeros_like(gray)
                        mask[y1:y2, x1:x2] = 255
                        p0 = cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=50,
                                                     qualityLevel=0.3, minDistance=7)
                        prev_gray = gray.copy()
                        has_detection = True
                        yolo_success = True
                        node.yolo_success = True
                        last_detect_time = time.time()
                        break 
                else:
                    yolo_success = False


            if not yolo_success and has_detection and p0 is not None and prev_gray is not None and time.time() - last_detect_time <= 2:
                # 使用光流追踪
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None,
                                                     winSize=(15, 15), maxLevel=2)
                good_new = p1[st == 1]
                if len(good_new) >= 5:
                    x, y, w, h = cv2.boundingRect(good_new.reshape(-1, 1, 2))

                    # draw
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(frame, 'OpticalFlow', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    target_point = (x + w / 2, y + h / 2)
                    error = (target_point[0] - center_point[0], target_point[1] - center_point[1])
                    cv2.imwrite('/mnt/disk/elf/camera_ws/src/camera/camera/output.jpg', frame)
                    

                    try:
                        info.put_nowait(error)
                    except:
                        try: info.get_nowait()
                        except: 
                            pass
                        info.put_nowait(error)
                    
                    prev_gray = gray.copy()
                    p0 = good_new.reshape(-1, 1, 2)
                else:
                    p0 = None

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        node.destroy_node()
        rclpy.shutdown()



# 0.025, 0.130
def control_job():
    global servo_angle
    global dead_zone
    global clock
    global img
    global num
    global flag_in
    global t0
    global yolo_success

    Last_error_S = 0
    Integral_S = 0
    #t0 = time.time()
    flag = 0
    n = 5
    dir = -1
    start = False
    servo_control(0.075)
    rotate(steps=2000, freq_hz=8000, direction=0)
    node.get_logger().info("升降台正在初始化")
    time.sleep(0.5)
    while flag_in is False:             #下降复位
                        rotate(steps=100, freq_hz=16000, direction=1)
                        time.sleep(0.02)
                        print('V')
    node.get_logger().info("初始化成功")

    while True:
        if clock == 1:      # 如果主机不发话题就不执行
            continue

        if not info.empty():
            start = True
            error = info.get()
            t1 = time.time()
            t = time.time()
            if t1 - t0 >= 0.4:
                Out_S, Last_error_S, Integral_S = pid_control(
                    0, error[0], Servo_Kp, Servo_Ki, Servo_Kd,
                    Last_error_S, Integral_S, 0.3, 0.01)

                if abs(Out_S) > dead_zone:
                    servo_angle += Out_S
                    servo_angle = max(min(servo_angle, limit_servo[1]), limit_servo[0])
                    servo_control(servo_angle)
                    flag = 1
                    #time.sleep(0.01)
                    print(f"舵机转动到{servo_angle:.8f}")
                    print(f"Error{error[0]:.8f}")

                t0 = t1

            if error[1] > 85 and flag_in is False and clock == 0:
                rotate(steps=200, freq_hz=16000, direction=1)
                time.sleep(0.05)
                flag = 1
                print("down")
            elif error[1] <= -85 and clock == 0:
                rotate(steps=200, freq_hz=16000, direction=0)
                time.sleep(0.05)
                flag = 1
                print("Up")
            

            print(f"er:{error[1]}")
            print(f"yolo_success:{yolo_success}")
            if clock == 0 and flag == 0:
                time.sleep(2)
                if img is not None:
                    timestamp = time.strftime("%Y%m%d%H%M%S")
                    filename = f'/mnt/disk/elf/Pictures/Img{num}_{timestamp}.jpg'
                    cv2.imwrite(filename, img)
                    node.get_logger().info(f"拍照成功")
                    time.sleep(1)
                    servo_control(0.075)
                    num += 1
                    start = False
                    node.get_logger().info(f"升降台开始回收")
                    while flag_in is False:
                        rotate(steps=200, freq_hz=16000, direction=1)
                    node.get_logger().info(f"升降台回收成功")
                    node.send(1)
                    time.sleep(4)
                    
                clock = 1
            elif error[1] < 85 and error[1] >= -85:
                flag = 0

        else:
            if start is True:
                time.sleep(0.01)
                t1 = time.time()
                if t1 - t0 >= 2:
                    start = False
            else:
                t1 = time.time()
                if t1 - t0 >= 2:
                    n += dir
                    if n >= 10:
                        dir = -1 
                    elif n <= 1:
                        dir = 1 
                    angle = 0.0250 + (n - 1) * 0.0100
                    servo_control(angle)
                    t0 = t1

                
def gpio_listener():
    fd = os.open(f"{GPIO_PATH}/value", os.O_RDONLY)
    os.read(fd, 1)
    poller = select.poll()
    poller.register(fd, select.POLLPRI)
    last_trigger = 0
    debounce_ms = 50
    global flag_in

    while not os.path.exists(f"{GPIO_PATH}/value"):
        time.sleep(0.1)
    while True:
        poller.poll()
        now = time.time() * 1000
        if now - last_trigger < debounce_ms:
            continue
        os.lseek(fd, 0, os.SEEK_SET)
        val = os.read(fd, 1).decode().strip()
        if val == '0':
            flag_in = True
            print("True")
        else:
            flag_in = False
            print("False")
        last_trigger = now


def main():
    global node
    rclpy.init()
    node = ImagePublisher()
    rclpy.spin_once(node, timeout_sec=0)
    t1 = threading.Thread(target=main_job, daemon=True)
    t2 = threading.Thread(target=control_job, daemon=True)
    t3 = threading.Thread(target=gpio_listener, daemon=True)

    t1.start()
    t2.start()
    t3.start()

    try:
        t1.join()
        t2.join()
        t3.join()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
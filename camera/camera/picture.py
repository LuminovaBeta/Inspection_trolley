# image_publisher.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_node')
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.timer = self.create_timer(1.0 / 120, self.timer_callback)  # 120 FPS
        self.br = CvBridge()
        self.cap = cv2.VideoCapture('/dev/video21')

        if not self.cap.isOpened():
            self.get_logger().error("无法打开摄像头")
            exit(1)
        else:
            self.get_logger().info("摄像头图像发布节点启动")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("未读取到摄像头帧")
            return

        img_msg = self.br.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher_.publish(img_msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

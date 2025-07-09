import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

class Bridge(Node):
    def __init__(self):
        super().__init__('bridge_node')

        self.subscription = self.create_subscription(
            Int32,                                      #待修改
            '/camera/start_capture', 
            self.callback,
            10
        )

        self.publisher_ = self.create_publisher(
            Int32,
             '/camera/switch_clock',
              10
        )
        self.get_logger().info("桥接节点启动成功")
        self.clock = 1             # 拍照锁 默认上锁

    def callback(self, msg):        #此时解除锁，相机可以拍摄
        try:
            self.get_logger().info("接收成功")
            self.clock = 0  # 解锁
            clock_msg = Int32()
            clock_msg.data = self.clock
            self.publisher_.publish(clock_msg)
            self.clock = 1  # 重置锁状态
        except Exception as e:
            self.get_logger().error(f"Error in callback: {e}")

def main(args=None):
    rclpy.init(args=args)
    bridge_node = Bridge()
    rclpy.spin(bridge_node)
    bridge_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
import os
import select

GPIO_PIN = 123
GPIO_PATH = f"/sys/class/gpio/gpio{GPIO_PIN}"

def wait_for_interrupt():
    fd = os.open(f"{GPIO_PATH}/value", os.O_RDONLY)
    os.read(fd, 1)  # 清掉初始电平

    poller = select.poll()
    poller.register(fd, select.POLLPRI)

    print("等待 GPIO123 中断触发...")

    while True:
        events = poller.poll()
        if events:
            os.lseek(fd, 0, os.SEEK_SET)
            val = os.read(fd, 1).decode().strip()
            print(f"中断触发，当前电平: {val}")

    os.close(fd)

if __name__ == "__main__":
    wait_for_interrupt()

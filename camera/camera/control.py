import time
import gpiod


class PersistentGPIO:
    def __init__(self, chip_path: str, pin: int, name="pctrl"):
        self.chip = gpiod.Chip(chip_path)
        self.pin = pin
        self.line = self.chip.get_line(pin)
        self.line.request(consumer=name, type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])

    def set(self, value: int):
        self.line.set_value(value)
#        print(f"[GPIO] 设置 GPIO{self.pin} = {value}")

    def close(self):
        self.line.set_value(0)
        self.line.release()
#        print(f"[GPIO] GPIO{self.pin} 已释放")

gpio_pul = PersistentGPIO("/dev/gpiochip3", 13)
gpio_dir = PersistentGPIO("/dev/gpiochip3", 8)
gpio_servo = PersistentGPIO("/dev/gpiochip3", 12)


# ====== 软件 PWM 控制（舵机）======
def software_pwm(gpio_obj, duty_cycle, duration=0.5, frequency=50):
    # 限制安全占空比范围（舵机推荐 2.5% ~ 12.5%）
    duty_cycle = max(min(duty_cycle, 0.125), 0.025)

    period = 1.0 / frequency
    high_time = period * duty_cycle
    low_time = period - high_time

    end_time = time.perf_counter() + duration
    while time.perf_counter() < end_time:
        gpio_obj.set(1)
        time.sleep(high_time)
        gpio_obj.set(0)
        time.sleep(low_time)


# ====== PID 控制器 ======
def pid_control(target, current, kp, ki, kd, last_error, integral, dt, limit):
    error = target - current
    integral += error * dt
    derivative = (error - last_error) / dt
    output = kp * error + ki * integral + kd * derivative
    integral = max(min(integral, limit), -limit)  # 限幅积分
    return output, error, integral

# ====== 步进电机控制 ======
def rotate(steps, freq_hz, direction):
    delay = 1.0 / (freq_hz * 2)

    gpio_dir.set(direction)
    for _ in range(steps):
        gpio_pul.set(1)
        time.sleep(delay)
        gpio_pul.set(0)
        time.sleep(delay)

# ====== 舵机角度控制 ======
def servo_control(duty):
    #占空比 2.5% ~ 12.5%
   # duty = 0.025 + (angle / 180.0) * 0.1
    software_pwm(gpio_servo, duty_cycle=duty, duration=0.5, frequency=75)

# ====== 综合控制接口 ======
def control(servo_angle, lift_steps):
    servo_control(servo_angle)

    if lift_steps > 0:
        rotate(steps=int(lift_steps), freq_hz=5000, direction=0)
        time.sleep(0.5)
    elif lift_steps < 0:
        rotate(steps=int(abs(lift_steps)), freq_hz=5000, direction=1)
        time.sleep(0.5)

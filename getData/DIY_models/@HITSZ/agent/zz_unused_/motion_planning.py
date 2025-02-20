from pynput.keyboard import Controller, KeyCode, Key
from zz_unused_.threading_utils import call_slow_function
from dataclasses import dataclass
import pygetwindow as gw
import threading
import keyboard
import time
import zmq


class PID:
    def __init__(
        self, Kp, Ki, Kd, output_limits=(None, None), integral_limits=(None, None)
    ):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.previous_error = 0
        self.integral = 0
        self.output_limits = output_limits
        self.integral_limits = integral_limits

    def compute(self, setpoint, measured_value, dt):
        error = setpoint - measured_value
        self.integral += error * dt

        if self.integral_limits[0] is not None:
            self.integral = max(self.integral_limits[0], self.integral)
        if self.integral_limits[1] is not None:
            self.integral = min(self.integral_limits[1], self.integral)

        derivative = (error - self.previous_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        if self.output_limits[0] is not None:
            output = max(self.output_limits[0], output)
        if self.output_limits[1] is not None:
            output = min(self.output_limits[1], output)

        self.previous_error = error
        return output


class NonLinearMotionPlanner:
    def __init__(self):
        context = zmq.Context()
        self.running = True
        self.key_freelook = KeyCode.from_vk(0x6A)
        self.keyboard = Controller()
        self.arma3_window = gw.getWindowsWithTitle('Arma 3 "')[0]
        self.pose_sub_socket = context.socket(zmq.SUB)
        self.pose_sub_socket.connect("tcp://localhost:515")
        self.pose_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.pose = []
        self.start()

    def start(self):
        self.multithreading(self.monitor_esc)
        self.fetch_pose()
        # self.multithreading(self.fetch_pose)
        # self.arma3_window.activate()
        # time.sleep(0.5)

        # while self.running:
        #     time.sleep(0.5)

        # position_pid = PID(
        #     Kp=1.0,
        #     Ki=0.1,
        #     Kd=0.05,
        #     output_limits=(-0.02, 0.02),
        #     integral_limits=(-5, 5),
        # )
        # velocity_pid = PID(
        #     Kp=1.0,
        #     Ki=0.1,
        #     Kd=0.05,
        #     output_limits=(-0.02, 0.02),
        #     integral_limits=(-5, 5),
        # )
        # target_height = 50.0
        # current_height = self.pose[2]
        # current_velocity = self.pose[8]

        # previous_time = time.time()
        # while True:
        #     current_time = time.time()
        #     dt = current_time - previous_time
        #     previous_time = current_time
        #     target_velocity = position_pid.compute(
        #         target_height, current_height, self.dt
        #     )
        #     control_signal = velocity_pid.compute(
        #         target_velocity, current_velocity, self.dt
        #     )
        #     # self.control_output(0, 0, control_signal)
        #     time.sleep(self.dt)

    def fetch_pose(self):
        while self.running:
            message = self.pose_sub_socket.recv_string()
            split_result = message.split(":")
            print(split_result)
            # split_message = message.split(",")
            # self.pose = [float(item) for item in split_message]

    def press_key(self, key, duration):
        def release_key():
            self.keyboard.release(key)

        self.keyboard.press(key)
        timer = threading.Timer(duration, release_key)
        timer.start()

    def multithreading(self, func):
        thread = threading.Thread(target=func, daemon=True)
        thread.start()

    def monitor_esc(self):
        while self.running:
            if keyboard.is_pressed("esc"):
                self.running = False


def main():
    NonLinearMotionPlanner()


if __name__ == "__main__":
    main()

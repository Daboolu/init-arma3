import subprocess
import time
import zmq

context = zmq.Context()
pub_socket = context.socket(zmq.PUB)
pub_socket.bind("tcp://*:114")

sub_socket = context.socket(zmq.SUB)
sub_socket.connect("tcp://localhost:514")
sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")


def send_message(message):
    """
    ["agent.send_message", ["message"]] call py3_fnc_callExtension;
    """
    pub_socket.send_string(message)


def read_message():
    """
    ["agent.read_message", []] call py3_fnc_callExtension;
    """
    try:
        message = sub_socket.recv_string(zmq.NOBLOCK)
        return str(message)
    except zmq.Again as e:
        return "NONE"


def disparity_window():
    """
    ["agent.disparity_window", []] call py3_fnc_callExtension;
    """
    command = [
        "D:/anaconda3/envs/arma3/python.exe",
        "d:/steam/steamapps/common/Arma 3/DIY_models/@HITSZ/agent/disparity_window.py",
    ]
    subprocess.Popen(command, shell=True)


def points_window():
    """
    ["agent.points_window", []] call py3_fnc_callExtension;
    """
    command = [
        "D:/anaconda3/envs/arma3/python.exe",
        "d:/steam/steamapps/common/Arma 3/DIY_models/@HITSZ/agent/points_window.py",
    ]
    subprocess.Popen(command, shell=True)

import time
from PIL import ImageGrab
import os

# 定义截图区域的坐标 (左, 上, 右, 下)
regions = [
    (0, 255, 455, 511),  # 区域1
    (455, 255, 911, 511), # 区域2
    (911, 255, 1365, 511), # 区域3
    (0, 511, 455, 767), # 区域4
    (455, 511, 911, 767),# 区域5
    (911, 511, 1365, 767) # 区域6
]

# 创建保存截图的文件夹
output_folder = 'screenshots'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def take_screenshots():
    for i, region in enumerate(regions):
        screenshot = ImageGrab.grab(bbox=region)
        screenshot.save(os.path.join(output_folder, f'{filename} screenshot_{i+1} .png'))

# 定时截图间隔时间（秒）
interval = 2

try:
    time.sleep(2)
    while True:
        filename=time.strftime("%Y%m%d-%H%M%S")
        take_screenshots()
        print(f"Screenshots taken at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(interval)
except KeyboardInterrupt:
    print("Screenshot process stopped.")
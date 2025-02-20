import open3d as o3d
import numpy as np

# 加载保存的点云数据
point_cloud = o3d.io.read_point_cloud(
    r"D:\steam\steamapps\common\Arma 3\DIY_models\@HITSZ\agent\output.ply"
)

# 可视化点云数据
o3d.visualization.draw_geometries(
    [point_cloud],
    point_show_normal=False,
)

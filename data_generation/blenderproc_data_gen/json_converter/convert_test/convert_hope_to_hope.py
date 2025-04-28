import json
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from validate_data_HOPE import OBJECT_SIZES_M, get_3d_bbox, project_points

def convert_pose_to_quaternion(pose):
    rotation = R.from_matrix(pose[:3, :3])
    quat = rotation.as_quat()  # xyzw
    return quat.tolist()

def convert_hope_json(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    K = np.array(data["camera"]["intrinsics"])
    output = {
    "camera_data": data.get("camera", {}),
    "objects": []
    }


    for obj in data["objects"]:
        obj_class = obj["class"]
        pose = np.array(obj["pose"])
        pose[:3, 3] *= 0.01  # cm → m

        # 1. 物体位移
        location = pose[:3, 3].tolist()

        # 2. 四元数
        quat = convert_pose_to_quaternion(pose)

        # 3. 3D bounding box in camera frame
        bbox_3d_obj = get_3d_bbox(obj_class)
        bbox_3d_obj_hom = np.hstack([bbox_3d_obj, np.ones((8, 1))])
        bbox_3d_cam = (pose @ bbox_3d_obj_hom.T).T[:, :3]

        # 4. 投影到图像平面
        projected_2d, valid = project_points(K, bbox_3d_cam)

        # 5. 添加中心点（0, 0, 0）
        center_obj = np.array([[0, 0, 0, 1]]).T
        center_cam = pose @ center_obj
        if center_cam[2, 0] > 0:
            center_2d = (K @ center_cam[:3]).flatten()
            center_2d /= center_2d[2]
            projected_2d = np.vstack([projected_2d, center_2d[:2]])
        else:
            continue  # skip object if center is behind camera

        # 6. 记录结果
        output["objects"].append({
            "class": obj_class.lower(),
            "visibility": 1.0,
            "location": location,
            "quaternion_xyzw": quat,
            "projected_cuboid": projected_2d.tolist()
        })

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✅ Saved converted file to {output_path}")
if __name__ == "__main__":
    # 举例：将 0000.json 转换为 000000.json
    convert_hope_json("0001.json", "000000.json")
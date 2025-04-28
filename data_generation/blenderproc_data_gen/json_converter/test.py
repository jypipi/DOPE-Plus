import json
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
# from validate_data_HOPE import OBJECT_SIZES_M, get_3d_bbox, project_points

# Map from object class to dimensions in cm (from HOPE dataset), converted to meters
OBJECT_SIZES_M = {
    "AlphabetSoup": [0.0836, 0.0711, 0.0661],
    "Butter": [0.0528, 0.0239, 0.1033],
    "Ketchup": [0.1486, 0.0434, 0.0645],
    "Pineapple": [0.0576, 0.0696, 0.0657],
    "BBQSauce": [0.1483, 0.0435, 0.0646],
    "MacaroniAndCheese": [0.1663, 0.0402, 0.1235],
    "Popcorn": [0.0850, 0.0383, 0.1265],
    "Mayo": [0.1479, 0.0410, 0.0645],
    "Raisins": [0.1232, 0.0398, 0.0859],
    "Cherries": [0.0580, 0.0709, 0.0661],
    "Milk": [0.1904, 0.0733, 0.0722],
    "SaladDressing": [0.1474, 0.0437, 0.0640],
    "ChocolatePudding": [0.0495, 0.0299, 0.0835],
    "Mushrooms": [0.0333, 0.0708, 0.0659],
    "Spaghetti": [0.0498, 0.0285, 0.2499],
    "Cookies": [0.1672, 0.0402, 0.1227],
    "Mustard": [0.1600, 0.0486, 0.0651],
    "TomatoSauce": [0.0828, 0.0702, 0.0665],
    "Corn": [0.0580, 0.0709, 0.0661],
    "OrangeJuice": [0.1925, 0.0661, 0.0711],
    "Yogurt": [0.0537, 0.0680, 0.0679],
    "GranolaBars": [0.1240, 0.0387, 0.1653],
    "Peaches": [0.0578, 0.0710, 0.0659],
    "GreenBeans": [0.0576, 0.0706, 0.0657],
    "PeasAndCarrots": [0.0585, 0.0706, 0.0659]
}

def project_points(K, points_3d):
    """ Project 3D points into 2D using pinhole camera intrinsics """
    points_2d = K @ points_3d.T
    points_2d = (points_2d[:2, :] / points_3d[:, 2]).T
    valid = points_3d[:, 2] > 0  # z > 0 is in front of camera
    return points_2d, valid


def get_3d_bbox(object_class):
    size = OBJECT_SIZES_M.get(object_class, (0.06, 0.06, 0.10))
    w, h, l = size
    x, y, z = l / 2, w / 2, h / 2
    corners = np.array([
        [-x, -y, -z],
        [ x, -y, -z],
        [ x,  y, -z],
        [-x,  y, -z],
        [-x, -y,  z],
        [ x, -y,  z],
        [ x,  y,  z],
        [-x,  y,  z]
    ])
    return corners


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
            "class": obj_class,
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
    convert_hope_json("0001.json", "0001_test.json")
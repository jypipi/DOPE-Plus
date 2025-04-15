#!/usr/bin/env python3

import json
import numpy as np
import os
from PIL import Image, ImageDraw
import sys


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


def main(json_files):
    for json_fn in json_files:
        base, _ = os.path.splitext(json_fn)
        img_fn = base + '_rgb.jpg'
        if not os.path.isfile(img_fn):
            print(f"Could not locate '{img_fn}'. Skipping..")
            continue

        with open(json_fn, 'r') as f:
            data = json.load(f)

        intrinsics = np.array(data['camera']['intrinsics'])
        K = intrinsics

        img = Image.open(img_fn)
        draw = ImageDraw.Draw(img)

        for obj in data['objects']:
            pose = np.array(obj['pose'])
            pose[:3, 3] *= 0.01  # cm → meters

            bbox_h = np.hstack([get_3d_bbox(obj['class']), np.ones((8, 1))])
            bbox_camera = (pose @ bbox_h.T).T[:, :3]

            bbox_2d, valid_mask = project_points(K, bbox_camera)

            if np.sum(valid_mask) < 4:
                print(f"Skipping {obj['class']} (not enough valid points in front of camera)")
                continue

            # Draw points
            for pt in bbox_2d:
                draw.ellipse((pt[0] - 3, pt[1] - 3, pt[0] + 3, pt[1] + 3), fill='cyan')

            # Draw edges
            edges = [[0, 1], [1, 2], [2, 3], [3, 0],
                     [4, 5], [5, 6], [6, 7], [7, 4],
                     [0, 4], [1, 5], [2, 6], [3, 7]]
            for i, j in edges:
                if valid_mask[i] and valid_mask[j]:
                    draw.line([tuple(bbox_2d[i]), tuple(bbox_2d[j])], fill='cyan', width=1)

            # Draw centroid
            centroid_obj = np.array([[0, 0, 0, 1]]).T
            centroid_camera = pose @ centroid_obj
            if centroid_camera[2, 0] > 0:
                centroid_2d = (K @ centroid_camera[:3]).flatten()
                centroid_2d /= centroid_2d[2]
                draw.ellipse((centroid_2d[0] - 4, centroid_2d[1] - 4,
                              centroid_2d[0] + 4, centroid_2d[1] + 4), fill='red')

            # Draw coordinate axes
            axis_length = 0.05
            origin = np.array([[0, 0, 0, 1]])
            x_axis = np.array([[axis_length, 0, 0, 1]])
            y_axis = np.array([[0, axis_length, 0, 1]])
            z_axis = np.array([[0, 0, axis_length, 1]])

            axis_pts_obj = np.vstack([origin, x_axis, y_axis, z_axis])
            axis_pts_cam = (pose @ axis_pts_obj.T).T[:, :3]
            axis_2d, valid = project_points(K, axis_pts_cam)

            if np.all(valid):
                origin_2d, x_2d, y_2d, z_2d = axis_2d
                draw.line([tuple(origin_2d), tuple(x_2d)], fill='red', width=2)
                draw.line([tuple(origin_2d), tuple(y_2d)], fill='green', width=2)
                draw.line([tuple(origin_2d), tuple(z_2d)], fill='blue', width=2)
            else:
                print(f"Skipping axes for {obj['class']} (invalid projection)")

        img.save(base + '-validate.png')
        print(f"Saved visualization to {base}-validate.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} _JSON FILES_")
        exit(0)
    s = sys.argv[1].lstrip('-')
    if s == "h" or s == "help":
        print(f"Usage: {sys.argv[0]} _JSON FILES_")
        exit(0)
    main(sys.argv[1:])

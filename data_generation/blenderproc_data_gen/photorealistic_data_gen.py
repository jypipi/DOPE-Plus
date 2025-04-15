import blenderproc as bp
import os
import argparse
import numpy as np
from pyquaternion import Quaternion
import random
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_model', type=str, required=True, help='Path to the textured OBJ model')
    parser.add_argument('--hdri_folder', type=str, required=True, help='Folder with HDRI .hdr files')
    parser.add_argument('--output_dir', type=str, default='photoreal_output', help='Output directory')
    parser.add_argument('--num_frames', type=int, default=500)
    parser.add_argument('--image_width', type=int, default=640)
    parser.add_argument('--image_height', type=int, default=480)
    parser.add_argument('--scale', type=float, default=1.0, help='Scale of the object model')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Photorealism: High-quality HDRI lighting --- #
    hdri_paths = [os.path.join(args.hdri_folder, f) for f in os.listdir(args.hdri_folder)
                  if f.lower().endswith('.hdr')]

    if not hdri_paths:
        raise FileNotFoundError("No HDRI files found in provided folder")

    bp.init()

    # Camera setup
    bp.camera.set_resolution(args.image_width, args.image_height)
    bp.camera.set_intrinsics_from_blender_params(lens=0.785398, lens_unit='FOV')

    # Load object
    obj = bp.loader.load_obj(args.object_model)[0]
    obj.set_scale([args.scale] * 3)

    # Photorealism: Enable Cycles and realistic material behavior
    bp.renderer.set_renderer_type("CYCLES")  # Physically-based renderer ★
    bp.renderer.set_output_format("PNG")

    for frame in range(args.num_frames):
        # Random HDRI environment for realism ★
        hdri = random.choice(hdri_paths)
        bp.world.set_world_background(hdri_path=hdri, strength=random.uniform(1.0, 3.0))

        # Camera pose
        cam_pose = bp.math.build_transformation_mat(
            location=[random.uniform(-1, 1), -random.uniform(1.5, 2.5), random.uniform(0.5, 1.5)],
            rotation=[np.radians(90), 0, random.uniform(-0.5, 0.5)]
        )
        bp.camera.add_camera_pose(cam_pose)

        # Object pose
        obj_pose = bp.math.build_transformation_mat(
            location=[0, 0, 0],
            rotation=bp.math.rotation_matrix_from_euler([0, 0, random.uniform(0, 2 * np.pi)])
        )
        obj.set_local2world_mat(obj_pose)

        # Render
        data = bp.renderer.render()
        image_path = os.path.join(args.output_dir, f"{frame:06d}.png")
        bp.writer.write_image(image_path, data["colors"][0])

        # Extract 6D pose label
        cam_T_obj = np.linalg.inv(cam_pose) @ obj_pose
        translation = cam_T_obj[:3, 3].tolist()
        rotation = Quaternion(matrix=cam_T_obj[:3, :3]).elements
        quaternion = [rotation[1], rotation[2], rotation[3], rotation[0]]  # x, y, z, w

        label = {
            "camera_pose": cam_pose.tolist(),
            "obj_pose_cam_frame": {
                "position": translation,
                "quaternion_xyzw": quaternion
            }
        }
        json_path = os.path.join(args.output_dir, f"{frame:06d}.json")
        with open(json_path, 'w') as f:
            json.dump(label, f, indent=4)

if __name__ == '__main__':
    main()

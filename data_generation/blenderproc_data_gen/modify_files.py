import os
import shutil

def copy_textures(base_dir):
    """
    Copies 'texture.png' from each subfolder's 'materials/textures' directory
    to the corresponding 'meshes' directory.
    """
    count_success = 0
    count_fail = 0
    for subfolder in os.listdir(base_dir):
        subfolder_path = os.path.join(base_dir, subfolder)
        
        if os.path.isdir(subfolder_path):
            src_texture_path = os.path.join(subfolder_path, 'materials', 'textures', 'texture.png')
            dst_texture_path = os.path.join(subfolder_path, 'meshes', 'texture.png')
            
            if os.path.exists(src_texture_path):
                shutil.copy2(src_texture_path, dst_texture_path)
                # print(f"Copied: {src_texture_path} -> {dst_texture_path}")
                count_success += 1
            else:
                count_fail += 1
                print(f"Texture not found: {src_texture_path}")

        print("Success: ", count_success, ", Fail: ", count_fail)

if __name__ == "__main__":
    base_directory = os.path.expanduser("~/ROB590_WS/Deep_Object_Pose/data_generation/blenderproc_data_gen/google_scanned_models")
    copy_textures(base_directory)

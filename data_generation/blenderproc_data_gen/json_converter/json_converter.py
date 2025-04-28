import os
import glob
from test import convert_hope_json  # Your conversion function

def convert_all_json_recursive(input_dir, suffix="_convert"):
    pattern = os.path.join(input_dir, "**", "*.json").replace("\\", "/")
    print(f"🔍 Searching with pattern: {pattern}")

    json_files = glob.glob(pattern, recursive=True)
    print(f"🔍 Found {len(json_files)} JSON files under {input_dir}")

    count = 0
    for fpath in sorted(json_files):
        try:
            if fpath.endswith(f"{suffix}.json"):
                continue

            dir_path = os.path.dirname(fpath)
            base_name = os.path.splitext(os.path.basename(fpath))[0]
            output_path = os.path.join(dir_path, base_name + suffix + ".json")

            convert_hope_json(fpath, output_path)
            count += 1
        except Exception as e:
            print(f"⚠️ Failed to convert {fpath}: {e}")

    print(f"✅ Converted {count} files.")



if __name__ == "__main__":
    input_root = "hope-dataset"
    convert_all_json_recursive(input_root)

    # files = glob.glob("hope-dataset/hope_video/scene_0000/*.json", recursive=True)
    # files = glob.glob("\\wsl.localhost\\Ubuntu-20.04\\home\\joeluo\\Deep_Object_Pose\\hope-dataset\\hope_video\\scene_0000\\*.json", recursive=True)
    
    # print(files)

    # path = "hope-dataset"
    # print("Exists:", os.path.exists(path))
    # print("Files inside:", os.listdir(path))
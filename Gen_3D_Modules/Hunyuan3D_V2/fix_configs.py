import os

TARGET_DIR = "Checkpoints/Diffusers/tencent/Hunyuan3D-2"
OLD_IMPORT = "hy3dgen"
NEW_IMPORT = "Hunyuan3D_V2.hy3dgen"

def patch_yaml_imports(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".yaml") or file.endswith(".yml"):
                file_path = os.path.join(subdir, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                if OLD_IMPORT in content:
                    new_content = content.replace(f"{OLD_IMPORT}.", f"{NEW_IMPORT}.")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    print(f"✔ Updated: {file_path}")
                else:
                    print(f"– Skipped (no match): {file_path}")

if __name__ == "__main__":
    patch_yaml_imports(TARGET_DIR)

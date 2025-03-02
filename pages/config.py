import sys
import os
import shutil


class GlobalData:
    """Global Excel data"""

    df = None
    filtered_df = None


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def clean_img_folder():
    img_dir = resource_path("img")

    if os.path.exists(img_dir):
        for filename in os.listdir(img_dir):
            file_path = os.path.join(img_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"删除 {file_path} 时出错: {e}")

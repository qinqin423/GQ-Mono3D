"""Generate depth images from the ARKitScene dataset."""

import os
import json
import cv2
import pandas as pd
from tqdm import tqdm
import zipfile  # 添加zipfile模块
import shutil  # 添加shutil模块，用于清理操作

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from download_data import download_data


def rotate_image(img, direction):
    if direction == "Up":
        pass
    elif direction == "Left":
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif direction == "Right":
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif direction == "Down":
        img = cv2.rotate(img, cv2.ROTATE_180)
    else:
        raise Exception(f"No such direction (={direction}) rotation")
    return img


def extract_zip_with_zipfile(zip_path, extract_dir):
    """
    使用Python内置的zipfile模块解压ZIP文件
    这是跨平台的解决方案
    """
    print(f"Extracting zip file {zip_path} with zipfile module...")
    try:
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 获取压缩包中的所有文件列表
            zip_files = zip_ref.namelist()
            # 使用tqdm显示解压进度
            for file in tqdm(zip_files, desc="Extracting", leave=False):
                zip_ref.extract(file, extract_dir)
        print(f"Successfully extracted to {extract_dir}")
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False


def safe_download_and_extract(
    dataset_type,
    video_ids,
    splits,
    data_dir,
    keep_zip=False,
    raw_dataset_assets=None,
    should_download_laser_scanner_point_cloud=None
):
    """
    安全下载并解压数据的函数
    这是对原download_data函数的封装，添加了zipfile解压
    """
    # 首先尝试调用原下载函数
    try:
        download_data(
            dataset_type,
            video_ids,
            splits,
            data_dir,
            keep_zip=True,  # 确保保留ZIP文件
            raw_dataset_assets=raw_dataset_assets,
            should_download_laser_scanner_point_cloud=should_download_laser_scanner_point_cloud
        )

        # 假设download_data会将ZIP文件下载到 data_dir/3dod/split/video_id.zip
        for video_id, split in zip(video_ids, splits):
            zip_path = os.path.join(data_dir, "3dod", split, f"{video_id}.zip")

            if os.path.exists(zip_path):
                # 计算解压目录
                extract_dir = os.path.join(data_dir, "3dod", split)

                # 使用zipfile解压
                if extract_zip_with_zipfile(zip_path, extract_dir):
                    # 如果需要清理ZIP文件
                    if not keep_zip:
                        os.remove(zip_path)
                        print(f"Removed zip file: {zip_path}")
                else:
                    print(f"Failed to extract {zip_path}")
            else:
                print(f"ZIP file not found at expected location: {zip_path}")
                # 检查是否已解压
                video_dir = os.path.join(extract_dir, video_id)
                if os.path.exists(video_dir):
                    print(f"Video directory already exists: {video_dir}")
                else:
                    print(f"Warning: Neither ZIP file nor extracted directory found for {video_id}")

    except Exception as e:
        print(f"Error in download_data: {e}")
        # 回退方案：尝试直接使用ZIP文件
        for video_id, split in zip(video_ids, splits):
            zip_path = os.path.join(data_dir, "3dod", split, f"{video_id}.zip")
            if os.path.exists(zip_path):
                print(f"Trying direct extraction for {video_id}")
                extract_dir = os.path.join(data_dir, "3dod", split)
                extract_zip_with_zipfile(zip_path, extract_dir)


def generate_depth(
    omni3d_data_root: str = "data/omni3d",
    data_dir: str = "data/ARKitScenes",
    target_data_dir: str = "data/ARKitScenes_depth",
    depth_scale: float = 1000.0,
) -> None:
    """Generate depth for ARKitScenes dataset."""
    os.makedirs(target_data_dir, exist_ok=True)

    meta_data = None
    for dataset in [
        "ARKitScenes_train",
        "ARKitScenes_val",
        "ARKitScenes_test",
    ]:
        print(f"Parsing {dataset}...")

        video_ids = []
        lowres_samples = {}
        not_found = 0
        not_found_depths = []

        annotation = os.path.join(
            omni3d_data_root, "annotations", f"{dataset}.json"
        )

        with open(annotation, "r") as file:
            samples = json.load(file)

        for img in tqdm(samples["images"]):
            _, split, video_id, img_name = img["file_path"].split("/")

            if not video_id in video_ids:
                # 使用修改后的下载和解压函数
                safe_download_and_extract(
                    "3dod",
                    [video_id],
                    [split],
                    data_dir,
                    keep_zip=False,
                    raw_dataset_assets=None,
                    should_download_laser_scanner_point_cloud=None
                )
                video_ids.append(video_id)

            if meta_data is None:
                meta_data = pd.read_csv(
                    os.path.join(data_dir, "3dod", "metadata.csv")
                )

                sky_directions = {}
                for vid, sky_direction in zip(
                    meta_data["video_id"], meta_data["sky_direction"]
                ):
                    sky_directions[vid] = sky_direction

            # 构建深度目录路径
            depth_dir = os.path.join(
                data_dir,
                "3dod",
                split,
                video_id,
                f"{video_id}_frames",
                "lowres_depth",
            )

            # 检查目录是否存在，如果不存在尝试手动解压
            if not os.path.exists(depth_dir):
                print(f"Depth directory not found: {depth_dir}")

                # 检查ZIP文件是否存在
                zip_path = os.path.join(data_dir, "3dod", split, f"{video_id}.zip")
                if os.path.exists(zip_path):
                    print(f"Found ZIP file, extracting: {zip_path}")
                    extract_zip_with_zipfile(zip_path, os.path.join(data_dir, "3dod", split))
                else:
                    print(f"ZIP file not found: {zip_path}")
                    # 跳过这个视频
                    not_found += 1
                    not_found_depths.append(img["file_path"])
                    continue

            if not video_id in lowres_samples:
                # 再次检查目录是否存在
                if os.path.exists(depth_dir):
                    try:
                        lowres_samples[video_id] = [
                            f"{float(f.split('_')[1].replace('.png', '')):.3f}"
                            for f in os.listdir(depth_dir)
                            if f.endswith(".png")
                        ]
                    except Exception as e:
                        print(f"Error reading files in {depth_dir}: {e}")
                        lowres_samples[video_id] = []
                else:
                    print(f"Directory still not accessible: {depth_dir}")
                    lowres_samples[video_id] = []
                    not_found += 1
                    not_found_depths.append(img["file_path"])
                    continue

            sample_time = img_name.split("_")[0]

            if not sample_time in lowres_samples[video_id]:
                if (
                    f"{float(sample_time) - 0.001:.3f}"
                    in lowres_samples[video_id]
                ):
                    sample_time = f"{float(sample_time) - 0.001:.3f}"
                elif (
                    f"{float(sample_time) + 0.001:.3f}"
                    in lowres_samples[video_id]
                ):
                    sample_time = f"{float(sample_time) + 0.001:.3f}"
                else:
                    not_found += 1
                    not_found_depths.append(img["file_path"])

            depth_image_path = os.path.join(depth_dir, f"{video_id}_{sample_time}.png")

            if not os.path.exists(depth_image_path):
                print(f"Depth image not found: {depth_image_path}")
                not_found += 1
                not_found_depths.append(img["file_path"])
                continue

            depth_image = cv2.imread(
                depth_image_path,
                cv2.IMREAD_UNCHANGED,
            )

            if depth_image is None:
                print(f"Failed to load depth image: {depth_image_path}")
                not_found += 1
                not_found_depths.append(img["file_path"])
                continue

            sky_direction = sky_directions[int(video_id)]

            depth_image = rotate_image(depth_image, sky_direction)

            depth = depth_image.astype(np.float32) / depth_scale

            depth = F.interpolate(
                torch.from_numpy(depth)[None, None],
                (img["height"], img["width"]),
                mode="nearest",
                align_corners=None,
                antialias=False,
            ).numpy()[0, 0]

            if depth.max() > 10.0:
                print(f"Depth max: {depth.max()}")

            numpy_image = np.clip(
                np.clip(depth, a_min=0.0, a_max=10.0) * depth_scale,
                a_min=0,
                a_max=2**16 - 1,
            ).astype(np.uint16)

            depth_folder = os.path.join(target_data_dir, split, video_id)
            os.makedirs(depth_folder, exist_ok=True)

            depth_file_path = os.path.join(
                depth_folder, img_name.replace("jpg", "png")
            )

            Image.fromarray(numpy_image).save(depth_file_path)

        print(f"Samples not found: {not_found}")
        if not_found_depths:
            print("Not found depth files:")
            for f in not_found_depths:
                print(f"  {f}")


if __name__ == "__main__":  # pragma: no cover
    generate_depth()
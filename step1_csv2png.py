import os
import pandas as pd
import numpy as np
import cv2
import imageio.v3 as iio

# === 参数配置 ===
csv_folder = "./csv"
output_folder = "./TemperatureData_TIFF"
target_width, target_height = 4096, 3000

# === 确保输出目录存在 ===
os.makedirs(output_folder, exist_ok=True)

# === 批量处理所有 CSV 文件 ===
for filename in os.listdir(csv_folder):
    if filename.lower().endswith('.csv'):
        csv_path = os.path.join(csv_folder, filename)
        output_name = os.path.splitext(filename)[0] + ".tiff"
        output_path = os.path.join(output_folder, output_name)

        try:
            # === 读取温度数据 ===
            df = pd.read_csv(csv_path, skiprows=7).dropna(axis=1, how='all')
            data = df.astype(np.float32).values  # 原始温度数据

            # === 上采样到目标尺寸，保持温度值 ===
            resized_data = cv2.resize(data, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

            # === 保存为浮点型 TIFF（每像素即为温度值） ===
            iio.imwrite(output_path, resized_data, plugin="tifffile")
            print(f"✔ 处理成功: {filename} → {output_name}")
        except Exception as e:
            print(f"✘ 处理失败: {filename}, 错误信息: {e}")

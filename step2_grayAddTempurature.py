import os
import numpy as np
import cv2
import imageio.v3 as iio

# === 路径配置 ===
gray_folder = "./bmp"
temperature_folder = "./TemperatureData_TIFF"
lut_path = "gray_to_ir_lut.npz"
output_folder = "./result_ROI"
os.makedirs(output_folder, exist_ok=True)

# === ROI 配置 ===
margin = 30
gx1_init, gy1_init = 1890, 1320
gx2_init, gy2_init = 2230, 1666
gx1_raw, gy1_raw = gx1_init + margin, gy1_init + margin
gx2_raw, gy2_raw = gx2_init - margin, gy2_init - margin
offset_x = 0
offset_y = 0
gx1_shifted = gx1_raw + offset_x
gx2_shifted = gx2_raw + offset_x
gy1_shifted = gy1_raw + offset_y
gy2_shifted = gy2_raw + offset_y

# === 加载 LUT 映射 ===
lut = np.load(lut_path)
map_x = lut["map_x"]
map_y = lut["map_y"]

# === 遍历所有 bmp 文件 ===
for fname in os.listdir(gray_folder):
    if fname.lower().endswith('.bmp'):
        name_base = os.path.splitext(fname)[0]
        gray_path = os.path.join(gray_folder, fname)
        temperature_path = os.path.join(temperature_folder, name_base + ".tiff")

        # 检查温度图是否存在
        if not os.path.exists(temperature_path):
            print(f"⚠ 缺失对应温度图: {temperature_path}，跳过该样本")
            continue

        try:
            # === 加载图像数据 ===
            gray_img = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
            temperature_img = iio.imread(temperature_path).astype(np.float32)

            # === 重映射温度图 ===
            temperature_mapped = cv2.remap(
                temperature_img, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=np.nan
            )

            # === ROI 区域掩膜提取（使用偏移后的 ROI） ===
            roi_mask = np.zeros_like(gray_img, dtype=np.uint8)
            roi_mask[gy1_shifted:gy2_shifted, gx1_shifted:gx2_shifted] = 1
            temperature_roi = np.where(roi_mask > 0, temperature_mapped, np.nan)

            # === 可视化伪彩色图 ===
            vis = cv2.normalize(np.nan_to_num(temperature_roi), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)

            # === 保存路径 ===
            output_temp_png = os.path.join(output_folder, f"{name_base}gray_temp_map_vis.png")
            # output_temp_npy = os.path.join(output_folder, f"{name_base}gray_roi_temperature.npy")
            output_temp_tiff = os.path.join(output_folder, f"{name_base}gray_roi_temperature.tiff")

            # === 保存文件 ===
            cv2.imwrite(output_temp_png, vis_color)
            # np.save(output_temp_npy, temperature_roi)
            iio.imwrite(output_temp_tiff, temperature_roi, plugin="tifffile")

            print(f"✅ 完成处理: {name_base}")
        except Exception as e:
            print(f"❌ 处理失败: {name_base}, 错误: {e}")

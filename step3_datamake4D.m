%% === 1. 读取数据 ===
pc = pcread('./pointcloud/3DPC1.pcd');           
xyz = double(pc.Location);  % 原始点云

uv = load('./uvCoords/uv_coords.txt');          
u = uv(:,1) + 1;  % MATLAB 索引从1开始
v = uv(:,2) + 1;

temp_img = double(imread('./result_ROI/cube100gray_roi_temperature.tiff'));
[H, W] = size(temp_img);

% 有效像素筛选
valid = (u >= 1 & u <= W) & (v >= 1 & v <= H);
u = u(valid); v = v(valid); xyz = xyz(valid,:);
T = temp_img(sub2ind([H, W], v, u));

%% === 2. SOR 滤波（Statistical Outlier Removal）===
k = 30;             % 邻域点数
threshold = 1.0;    % 判定阈值（默认均值 + 1σ）

[idx, dists] = knnsearch(xyz, xyz, 'K', k+1);  % 包含自身
mean_dists = mean(dists(:,2:end), 2);          % 排除自身后求平均

global_mean = mean(mean_dists);
global_std = std(mean_dists);
valid_sor = mean_dists < global_mean + threshold * global_std;

xyz = xyz(valid_sor, :);
T = T(valid_sor);

%% === 3. 体素下采样 ===
grid_step = 0.05;  % 单位：mm
xyz_ds = round(xyz / grid_step) * grid_step;
[xyz_unique, idx_unique] = unique(xyz_ds, 'rows', 'stable');
xyz = xyz(idx_unique, :);
T = T(idx_unique);

%% === 4. SVD 主轴对齐 + Z方向反转 ===
mu = mean(xyz, 1);
xyz_centered = xyz - mu;
[~, ~, V] = svd(xyz_centered, 'econ');  % SVD 分解

% 使 Z 主轴对齐世界坐标正向 [0 0 1]
if dot(V(:,3), [0; 0; 1]) < 0
    V(:,3) = -V(:,3);
    V(:,1:2) = -V(:,1:2);  % 保持右手系
end

xyz_pca = xyz_centered * V;  % 主轴变换
x = xyz_pca(:,1);
y = xyz_pca(:,2);
z = -xyz_pca(:,3);  % ✅ 翻转 Z 方向
T = double(T);


%% === 5. 网格构建 ===
grid_size = 200;
x_range = linspace(min(x), max(x), grid_size);
y_range = linspace(min(y), max(y), grid_size);
[xq, yq] = meshgrid(x_range, y_range);

%% === 6. 使用 gridfit 拟合表面形貌 ===
zq = gridfit(x, y, z, xq(1,:), yq(:,1), 'smooth', 1);
zq_um = zq * 1000;  % μm单位

%% === 7. 温度插值 ===
F_temp = scatteredInterpolant(x, y, T, 'linear', 'none');
Tq = F_temp(xq, yq);

% 归一化温度颜色映射
Tq_min = nanmin(T(:));
Tq_max = nanmax(T(:));
Tq_norm = (Tq - Tq_min) / (Tq_max - Tq_min);
Tq_norm(isnan(Tq_norm)) = 0;

cmap = jet(256);
T_color = ind2rgb(floor(Tq_norm * 255) + 1, cmap);

%% === 8. 可视化结果 ===
figure;
surf(xq, yq, zq_um, Tq, 'EdgeColor', 'none');
shading interp;
colormap(jet);
caxis([Tq_min, Tq_max]);
cb = colorbar;
cb.Label.String = 'Temperature (°C)';
cb.Label.FontSize = 12;

xlabel('X (mm)');
ylabel('Y (mm)');
zlabel('Z (μm)');
title('4D (x, y, z, T) data fusion visualization');
axis equal;
view(45, 30);
set(gca, 'Color', 'w');
set(gcf, 'Color', 'w');

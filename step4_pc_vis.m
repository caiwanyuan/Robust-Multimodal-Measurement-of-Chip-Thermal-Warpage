%% === 1. 读取点云 ===
pc = pcread('./pointcloud/3DPC1.pcd');           
xyz = double(pc.Location);  % 点云坐标 (N x 3)

%% === 2. 点云体素下采样（去冗余）===
grid_step = 0.1;  % 下采样步长（单位：mm）
xyz_ds = round(xyz / grid_step) * grid_step;
[~, idx_unique] = unique(xyz_ds, 'rows', 'stable');
xyz = xyz(idx_unique, :);  % 仅保留唯一体素中心点

%% === 3. SOR 统计滤波（Statistical Outlier Removal）===
k = 30;             % 每个点的近邻数量
% threshold = 1.0;    % 离群阈值（均值 + threshold * std）

[idx, dists] = knnsearch(xyz, xyz, 'K', k+1);  % +1 是包括自己
mean_dists = mean(dists(:,2:end), 2);          % 去掉自己点后求均值距离

% 计算全体均值和标准差
global_mean = mean(mean_dists);
global_std  = std(mean_dists);

% 保留非离群点
valid = mean_dists < global_mean + threshold * global_std;
xyz = xyz(valid, :);  % 过滤掉离群点

%% === 4. 主方向对齐（SVD + Z轴统一）===
mu = mean(xyz, 1);
xyz_centered = xyz - mu;
[~, ~, V] = svd(xyz_centered, 'econ');

% Z轴与 [0 0 1] 对齐，保持右手系
if dot(V(:,3), [0; 0; 1]) < 0
    V(:,3) = -V(:,3);
    V(:,1:2) = -V(:,1:2);
end

xyz_pca = xyz_centered * V;
x = xyz_pca(:,1);
y = xyz_pca(:,2);
z = xyz_pca(:,3);  % 保留原始 z

%% === 5. 构建规则网格 ===
grid_size = 200;
x_range = linspace(min(x), max(x), grid_size);
y_range = linspace(min(y), max(y), grid_size);
[xq, yq] = meshgrid(x_range, y_range);

%% === 6. 拟合表面形貌（gridfit）===
zq = gridfit(x, y, z, xq(1,:), yq(:,1), 'smooth', 1);

% ✅ 翻转 Z 方向
zq = -zq;

% 单位转换为微米
zq_um = zq * 1000;

%% === 7. 可视化 ===
figure;
surf(xq, yq, zq_um, 'EdgeColor', 'none');
shading interp;

colormap parula;
cb = colorbar;
cb.Label.String = 'Z (μm)';
cb.Label.FontSize = 12;
cb.Label.FontWeight = 'bold';

xlabel('X (mm)');
ylabel('Y (mm)');
zlabel('Z (μm)');
title('xyz data visualization');
axis equal;
view(45, 30);
set(gca, 'Color', 'w');
set(gcf, 'Color', 'w');

% 可选叠加点云
% hold on;
% scatter3(x, y, -z*1000, 5, 'k', 'filled');  % 注意同步翻转

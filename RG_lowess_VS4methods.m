%==============================
% 点云表面拟合方法对比（论文完整出图版）
% Raw Point Cloud + Polynomial + RBF + LOWESS + Robust Gaussian Lowess
%
% 功能包括：
% 1) 原始点云读取
% 2) 参考平面统一调平
% 3) 四种曲面拟合
% 4) 原始点云 RMS + 各方法残差 RMS + Time 统计
% 5) 原始点云单独绘制（带 RMS 标注）
% 6) 四种拟合方法 2x2 对比图（每张子图带 RMS 标注）
% 7) 横截面对比图
% 8) RMS 与 Time 柱状图
%==============================

clear; clc; close all;

%% ==============================
% 1. 输入设置
% ===============================
pcd_path  = 'D:/code/gigaforce_matlab/cube_rms_pointcloud.pcd';

grid_res  = 80;
knn       = 12;
visualize = true;
draw_profile = true;
draw_stats = true;

% Robust Gaussian Lowess 参数
rg_span = 0.3;
rg_iter = 100;

% 横截面参数
profile_mode = 'middle';   % 'middle' or 'manual'
y0_manual = 0;
band_scale = 1.5;

% -------- 论文绘图统一参数 --------
font_name   = 'Times New Roman';
font_size   = 11;
title_size  = 14;
label_size  = 11;
line_width  = 1.5;
curve_width = 2.2;
scatter_size_3d = 8;
scatter_size_2d = 18;

% 导出图像分辨率
export_dpi = 600;

%% ==============================
% 2. 读取点云
% ===============================
ptCloud = pcread(pcd_path);
x = ptCloud.Location(:,1);
y = ptCloud.Location(:,2);
z = ptCloud.Location(:,3);

%% ==============================
% 3. 参考平面拟合（统一调平）
%    拟合平面 z = ax + by + c
% ===============================
P = [x, y, ones(size(x))];
coef = P \ z;
a = coef(1);
b = coef(2);
c = coef(3);

% 原始点云调平
z_plane_raw = a * x + b * y + c;
z_level = z - z_plane_raw;

% 原始点云相对于参考平面的 RMS
rms_raw = rms(z_level);

%% ==============================
% 4. 网格采样
% ===============================
x_lin = linspace(min(x), max(x), grid_res);
y_lin = linspace(min(y), max(y), grid_res);
[Xq, Yq] = meshgrid(x_lin, y_lin);

% 参考平面在网格上的值
Z_plane = a * Xq + b * Yq + c;

%% ==============================
% 5. 四种方法拟合
% ===============================

% ---------- 方法1：Robust Gaussian Lowess ----------
tic;
Z_rg_lowess = fit_g3d_lowess(x, y, z, Xq, Yq, rg_span, rg_iter);
Z_rg_lowess_level = Z_rg_lowess - Z_plane;
z_fit_rg_lowess = interp2(Xq, Yq, Z_rg_lowess_level, x, y, 'linear');
time_rg_lowess = toc;

valid_idx = ~isnan(z_fit_rg_lowess);
rms_rg_lowess = rms(z_level(valid_idx) - z_fit_rg_lowess(valid_idx));

% ---------- 方法2：LOWESS ----------
tic;
Z_lowess = fit_lowess_surface(x, y, z, Xq, Yq, knn);
Z_lowess_level = Z_lowess - Z_plane;
z_fit_lowess = interp2(Xq, Yq, Z_lowess_level, x, y, 'linear');
time_lowess = toc;

valid_idx = ~isnan(z_fit_lowess);
rms_lowess = rms(z_level(valid_idx) - z_fit_lowess(valid_idx));

% ---------- 方法3：Polynomial ----------
tic;
Z_poly = fit_poly_surface(x, y, z, Xq, Yq, 2);   % 2阶多项式
Z_poly_level = Z_poly - Z_plane;
z_fit_poly = interp2(Xq, Yq, Z_poly_level, x, y, 'linear');
time_poly = toc;

valid_idx = ~isnan(z_fit_poly);
rms_poly = rms(z_level(valid_idx) - z_fit_poly(valid_idx));

% ---------- 方法4：RBF ----------
tic;
Z_rbf = fit_rbf_surface(x, y, z, Xq, Yq);
Z_rbf_level = Z_rbf - Z_plane;
z_fit_rbf = interp2(Xq, Yq, Z_rbf_level, x, y, 'linear');
time_rbf = toc;

valid_idx = ~isnan(z_fit_rbf);
rms_rbf = rms(z_level(valid_idx) - z_fit_rbf(valid_idx));

%% ==============================
% 6. 汇总表格
% ===============================
methods = {'Raw Point Cloud'; 'Polynomial'; 'RBF'; 'LOWESS'; 'Robust Gaussian Lowess'};
rms_list = [rms_raw; rms_poly; rms_rbf; rms_lowess; rms_rg_lowess];
time_list = [0; time_poly; time_rbf; time_lowess; time_rg_lowess];

% 图2专用 RMS 顺序
rms_surface_list = [rms_poly, rms_rbf, rms_lowess, rms_rg_lowess];

T = table(methods, rms_list, time_list, ...
    'VariableNames', {'Method', 'RMS_Error_mm', 'Time_sec'});

disp('================== 拟合结果汇总 ==================');
disp(T);

%% ==============================
% 7. 颜色范围统一
% ===============================
all_z = [z_level; Z_poly_level(:); Z_rbf_level(:); Z_lowess_level(:); Z_rg_lowess_level(:)];
zmin = prctile(all_z, 2);
zmax = prctile(all_z, 98);

if abs(zmax - zmin) < 1e-12
    zmin = min(all_z);
    zmax = max(all_z);
end

%% ==============================
% 8. 图1：原始点云单独绘制
% ===============================
if visualize
    fig1 = figure('Color', 'w', 'Position', [100,100,720,560]);

    scatter3(x, y, z_level, scatter_size_3d, z_level, 'filled');
    view(45, 30);
    axis tight;
    grid on;
    box on;
    colormap(parula);
    caxis([zmin zmax]);

    cb = colorbar;
    cb.Label.String = 'Height (mm)';
    cb.FontName = font_name;
    cb.FontSize = font_size;

    xlabel('X (mm)', 'FontName', font_name, 'FontSize', label_size);
    ylabel('Y (mm)', 'FontName', font_name, 'FontSize', label_size);
    zlabel('Z (mm)', 'FontName', font_name, 'FontSize', label_size);
    title('Raw Point Cloud', 'FontName', font_name, 'FontSize', title_size);

    set(gca, 'FontName', font_name, 'FontSize', font_size, 'LineWidth', line_width);

    % RMS 标注
    text(0.03, 0.95, sprintf('Plane RMS = %.4f mm', rms_raw), ...
        'Units', 'normalized', ...
        'FontName', font_name, ...
        'FontSize', 11, ...
        'FontWeight', 'bold', ...
        'BackgroundColor', 'w', ...
        'EdgeColor', [0.3 0.3 0.3], ...
        'Margin', 5, ...
        'VerticalAlignment', 'top');

    exportgraphics(fig1, 'Fig1_RawPointCloud.png', 'Resolution', export_dpi);
end

%% ==============================
% 9. 图2：四种拟合方法 2x2 对比
% ===============================
if visualize
    fig2 = figure('Color', 'w', 'Position', [120,120,1120,860]);

    Z_all = {Z_poly_level, Z_rbf_level, Z_lowess_level, Z_rg_lowess_level};
    titles = {'Polynomial', 'RBF', 'Lowess', 'Robust Gaussian Lowess'};

    for i = 1:4
        subplot(2,2,i);
        surf(Xq, Yq, Z_all{i}, 'EdgeColor', 'none', 'FaceColor', 'interp');
        view(45, 30);
        axis tight;
        grid on;
        box on;
        colormap(parula);
        caxis([zmin zmax]);

        xlabel('X (mm)', 'FontName', font_name, 'FontSize', label_size);
        ylabel('Y (mm)', 'FontName', font_name, 'FontSize', label_size);
        zlabel('Z (mm)', 'FontName', font_name, 'FontSize', label_size);
        title(titles{i}, 'FontName', font_name, 'FontSize', title_size);

        set(gca, 'FontName', font_name, 'FontSize', font_size, 'LineWidth', line_width);

        % RMS 标注
        text(0.03, 0.95, sprintf('Plane RMS = %.5f mm', rms_surface_list(i)), ...
            'Units', 'normalized', ...
            'FontName', font_name, ...
            'FontSize', 11, ...
            'FontWeight', 'bold', ...
            'BackgroundColor', 'w', ...
            'EdgeColor', [0.3 0.3 0.3], ...
            'Margin', 5, ...
            'VerticalAlignment', 'top');
    end

    % 共用 colorbar
    h = colorbar('Position', [0.92 0.11 0.02 0.8]);
    h.Label.String = 'Height (mm)';
    h.FontName = font_name;
    h.FontSize = font_size;

    exportgraphics(fig2, 'Fig2_SurfaceComparison_2x2.png', 'Resolution', export_dpi);
end

%% ==============================
% 10. 横截面准备
% ===============================
if draw_profile
    switch lower(profile_mode)
        case 'middle'
            y0 = mean(y);
        case 'manual'
            y0 = y0_manual;
        otherwise
            error('profile_mode 只能是 ''middle'' 或 ''manual'' ');
    end

    % 原始点云截面带宽
    band_width = (max(y) - min(y)) / grid_res * band_scale;

    % 原始点云横截面
    raw_idx = abs(y - y0) < band_width;
    x_raw_sec = x(raw_idx);
    z_raw_sec = z_level(raw_idx);

    % 如果截到点太少，自动放宽
    if numel(x_raw_sec) < 10
        band_width = band_width * 2;
        raw_idx = abs(y - y0) < band_width;
        x_raw_sec = x(raw_idx);
        z_raw_sec = z_level(raw_idx);
    end

    [x_raw_sec, sort_idx] = sort(x_raw_sec);
    z_raw_sec = z_raw_sec(sort_idx);

    % 各拟合曲面横截面
    [~, row_idx] = min(abs(y_lin - y0));
    x_sec = Xq(row_idx, :);
    z_poly_sec      = Z_poly_level(row_idx, :);
    z_rbf_sec       = Z_rbf_level(row_idx, :);
    z_lowess_sec    = Z_lowess_level(row_idx, :);
    z_rg_lowess_sec = Z_rg_lowess_level(row_idx, :);
end

%% ==============================
% 11. 图3：横截面对比图
% ===============================
if draw_profile
    fig3 = figure('Color', 'w', 'Position', [150,150,930,540]);
    hold on;

    % 原始点云截面
    scatter(x_raw_sec, z_raw_sec, scatter_size_2d, ...
        'o', ...
        'MarkerEdgeColor', [0.6 0.6 0.6], ...
        'MarkerFaceColor', [0.78 0.78 0.78], ...
        'DisplayName', 'Raw Point Cloud');

    % 四种拟合曲线
    plot(x_sec, z_poly_sec, '-', 'LineWidth', curve_width, 'DisplayName', 'Polynomial');
    plot(x_sec, z_rbf_sec, '-', 'LineWidth', curve_width, 'DisplayName', 'RBF');
    plot(x_sec, z_lowess_sec, '-', 'LineWidth', curve_width, 'DisplayName', 'LOWESS');
    plot(x_sec, z_rg_lowess_sec, '-', 'LineWidth', 2.8, 'DisplayName', 'Robust Gaussian Lowess');

    xlabel('X (mm)', 'FontName', font_name, 'FontSize', label_size);
    ylabel('Z (mm)', 'FontName', font_name, 'FontSize', label_size);
    title(sprintf('Cross-sectional Profile at y = %.4f mm', y0), ...
        'FontName', font_name, 'FontSize', title_size);

    legend('Location', 'best', 'FontName', font_name, 'FontSize', 10);
    grid on;
    box on;
    set(gca, 'FontName', font_name, 'FontSize', font_size, 'LineWidth', line_width);
    hold off;

    exportgraphics(fig3, 'Fig3_ProfileComparison.png', 'Resolution', export_dpi);
end

%% ==============================
% 12. 图4：RMS 与 Time 统计图
% ===============================
if draw_stats
    fig4 = figure('Color', 'w', 'Position', [180,180,1040,440]);

    % ---------- RMS ----------
    subplot(1,2,1);
    bar(rms_list);
    xticks(1:numel(methods));
    xticklabels({'Raw', 'Poly', 'RBF', 'LOWESS', 'RGL'});
    ylabel('RMS Error (mm)', 'FontName', font_name, 'FontSize', label_size);
    title('RMS Error Comparison', 'FontName', font_name, 'FontSize', title_size);
    grid on;
    box on;
    set(gca, 'FontName', font_name, 'FontSize', font_size, 'LineWidth', line_width);

    for i = 1:numel(rms_list)
        text(i, rms_list(i), sprintf('%.4f', rms_list(i)), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontName', font_name, 'FontSize', 10);
    end

    % ---------- Time ----------
    subplot(1,2,2);
    bar(time_list);
    xticks(1:numel(methods));
    xticklabels({'Raw', 'Poly', 'RBF', 'LOWESS', 'RGL'});
    ylabel('Time (s)', 'FontName', font_name, 'FontSize', label_size);
    title('Computation Time Comparison', 'FontName', font_name, 'FontSize', title_size);
    grid on;
    box on;
    set(gca, 'FontName', font_name, 'FontSize', font_size, 'LineWidth', line_width);

    for i = 1:numel(time_list)
        text(i, time_list(i), sprintf('%.4f', time_list(i)), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontName', font_name, 'FontSize', 10);
    end

    exportgraphics(fig4, 'Fig4_RMS_Time_Comparison.png', 'Resolution', export_dpi);
end

%% ==============================
% 13. 控制台输出横截面信息
% ===============================
if draw_profile
    fprintf('\n================== 横截面信息 ==================\n');
    fprintf('Section mode     : %s\n', profile_mode);
    fprintf('Section y0        : %.6f mm\n', y0);
    fprintf('Band width        : %.6f mm\n', band_width);
    fprintf('Raw section points: %d\n', numel(x_raw_sec));
end
% === 配置路径 ===
pcd_folder = './pcd/';
uv_folder = './txt/';
temp_folder = './result_ROI/';

pcd_files = dir(fullfile(pcd_folder, '*.pcd'));

% === 参数配置 ===
k = 30;
threshold = 1.0;
grid_step = 0.4;       % mm
grid_size = 600;
lowess_span = 0.4;

% === 基准平面的旋转基 V_ref 和中心 mu_ref ===
V_ref = [];
mu_ref = [];

for i = 1:length(pcd_files)
    [~, name, ~] = fileparts(pcd_files(i).name);
    fprintf('\n--- 正在处理: %s ---\n', name);

    % === 文件路径 ===
    pcd_path = fullfile(pcd_folder, [name '.pcd']);
    uv_path = fullfile(uv_folder, [name '.txt']);
    temp_path = fullfile(temp_folder, [name 'gray_roi_temperature.tiff']);

    if ~isfile(uv_path) || ~isfile(temp_path)
        fprintf('⚠ 缺少对应的UV或温度图文件，跳过该样本\n');
        continue;
    end

    % === 1. 读取数据 ===
    pc = pcread(pcd_path);
    xyz = double(pc.Location);

    uv = load(uv_path);
    u = uv(:,1) + 1;
    v = uv(:,2) + 1;

    temp_img = double(imread(temp_path));
    [H, W] = size(temp_img);

    valid = (u >= 1 & u <= W) & (v >= 1 & v <= H);
    u = u(valid); v = v(valid); xyz = xyz(valid,:);
    T = temp_img(sub2ind([H, W], v, u));

    % === 2. SOR滤波 + 下采样 ===
    [~, dists] = knnsearch(xyz, xyz, 'K', k+1);
    mean_dists = mean(dists(:,2:end), 2);
    valid_sor = mean_dists < mean(mean_dists) + threshold * std(mean_dists);
    xyz = xyz(valid_sor, :);
    T = T(valid_sor);

    xyz_ds = round(xyz / grid_step) * grid_step;
    [xyz_unique, idx_unique] = unique(xyz_ds, 'rows', 'stable');
    xyz = xyz(idx_unique, :);
    T = T(idx_unique);

    % === 3. 坐标中心化 ===
    mu = mean(xyz, 1);
    xyz_centered = xyz - mu;

    % === 4. 计算全局主轴方向（只用第一个）===
    if i == 1
        [~, ~, V_ref] = svd(xyz_centered, 'econ');
        if dot(V_ref(:,3), [0; 0; 1]) < 0
            V_ref(:,3) = -V_ref(:,3);
            V_ref(:,1:2) = -V_ref(:,1:2);
        end
        mu_ref = mu;
        fprintf('✅ 使用 %s 构建全局主方向坐标系\n', name);
    end

    % === 5. 应用全局主方向对齐 ===
    xyz_aligned = xyz_centered * V_ref;
    x = xyz_aligned(:,1);
    y = xyz_aligned(:,2);
    z = xyz_aligned(:,3);

    % === 6. 网格构建（中心居中）===
    x_center = (min(x) + max(x)) / 2;
    y_center = (min(y) + max(y)) / 2;
    x_shifted = x - x_center;
    y_shifted = y - y_center;

    x_range = linspace(min(x_shifted), max(x_shifted), grid_size);
    y_range = linspace(min(y_shifted), max(y_shifted), grid_size);
    [xq, yq] = meshgrid(x_range, y_range);

    % === 7. LOWESS 表面拟合 ===
    fprintf('  -> LOWESS曲面拟合中...\n');
    ft = fit([x_shifted, y_shifted], z, 'lowess', 'Span', lowess_span);
    zq = feval(ft, xq, yq);
    zq_um = min(max(zq * 1000, -10), 50);  % μm单位范围裁剪

    % === 8. 温度插值映射 ===
    F_temp = scatteredInterpolant(x_shifted, y_shifted, T, 'linear', 'none');
    Tq = F_temp(xq, yq);
    Tq_min = nanmin(T(:));
    Tq_max = nanmax(T(:));
    Tq_norm = (Tq - Tq_min) / (Tq_max - Tq_min);
    Tq_norm(isnan(Tq_norm)) = 0;
    cmap = jet(256);
    T_color = ind2rgb(floor(Tq_norm * 255) + 1, cmap);

    % === 9. 三维可视化 ===
    figure('Name', name);
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
    title(['4D Global Aligned Visualization - ', name]);
    axis equal;
    view(45, 30);
    set(gca, 'Color', 'w');
    set(gcf, 'Color', 'w');

    fprintf("  ✅ %s 拟合后 Z 范围：%.2f ~ %.2f μm\n", name, min(zq_um(:)), max(zq_um(:)));
end

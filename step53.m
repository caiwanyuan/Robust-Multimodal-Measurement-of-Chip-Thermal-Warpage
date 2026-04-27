%% === 配置路径 ===
pcd_folder = './pcd/';
uv_folder = './txt/';
temp_folder = './result_ROI/';
vis_save_folder = './vis_result/';
if ~exist(vis_save_folder, 'dir'); mkdir(vis_save_folder); end
pcd_files = dir(fullfile(pcd_folder, '*.pcd'));

%% === 参数配置 ===
k = 30;
threshold = 0.2;
grid_step = 0.2;       % mm
grid_size = 300;
lowess_span = 0.4;

%% === 初始化记录 ===
V_ref = [];
mu_ref = [];
warpage_record = {};
sampling_stats = {};  % {filename, raw, SOR后, 下采样后}

%% === 主循环：处理每一个样本 ===
for i = 1:length(pcd_files)
    [~, name, ~] = fileparts(pcd_files(i).name);
    fprintf('\n--- 正在处理: %s ---\n', name);

    % === 构建路径 ===
    pcd_path = fullfile(pcd_folder, [name '.pcd']);
    uv_path = fullfile(uv_folder, [name '.txt']);
    temp_path = fullfile(temp_folder, [name 'gray_roi_temperature.tiff']);

    if ~isfile(uv_path) || ~isfile(temp_path)
        fprintf('⚠ 缺失 UV 或温度图像，跳过\n');
        continue;
    end

    % === 1. 读取数据 ===
    pc = pcread(pcd_path); xyz = double(pc.Location);
    uv = load(uv_path); u = uv(:,1) + 1; v = uv(:,2) + 1;
    temp_img = double(imread(temp_path));
    [H, W] = size(temp_img);
    valid = (u >= 1 & u <= W) & (v >= 1 & v <= H);
    u = u(valid); v = v(valid); xyz = xyz(valid,:);
    T = temp_img(sub2ind([H, W], v, u));

    raw_count = size(xyz,1);
    fprintf('  - 原始点数: %d\n', raw_count);

%      % === 1.1 Z方向直通滤波 ===
%     z_mm = xyz(:,3);  % 提取Z坐标（单位mm）
%     z_min = -7.0;    % -30 μm
%     z_max =  6.5;    % +30 μm
%     z_valid = (z_mm >= z_min) & (z_mm <= z_max);
%     xyz = xyz(z_valid, :); u = u(z_valid); v = v(z_valid); T = T(z_valid);
% 
%     fprintf('  - Z方向直通滤波后点数: %d\n', size(xyz,1));

    % === 2. SOR滤波 + 下采样 ===
    [~, dists] = knnsearch(xyz, xyz, 'K', k+1);
    mean_dists = mean(dists(:,2:end), 2);
    valid_sor = mean_dists < mean(mean_dists) + threshold * std(mean_dists);
    xyz = xyz(valid_sor,:); T = T(valid_sor);

    sor_count = size(xyz,1);
    fprintf('  - SOR滤波后点数: %d\n', sor_count);

    xyz_ds = round(xyz / grid_step) * grid_step;
    [xyz_unique, idx_unique] = unique(xyz_ds, 'rows', 'stable');
    xyz = xyz(idx_unique,:); T = T(idx_unique);

    ds_count = size(xyz,1);
    fprintf('  - 下采样后点数: %d\n', ds_count);

    % === 点数记录 ===
    sampling_stats{end+1,1} = name;
    sampling_stats{end,2} = raw_count;
    sampling_stats{end,3} = sor_count;
    sampling_stats{end,4} = ds_count;

    % === 3. PCA对齐 ===
    mu = mean(xyz,1); xyz_centered = xyz - mu;
    if i == 1
        [~, ~, V_ref] = svd(xyz_centered, 'econ');
        if dot(V_ref(:,3), [0;0;1]) < 0
            V_ref(:,3) = -V_ref(:,3); V_ref(:,1:2) = -V_ref(:,1:2);
        end
        mu_ref = mu;
    end
    xyz_aligned = xyz_centered * V_ref;
    x = xyz_aligned(:,1); y = xyz_aligned(:,2); z = xyz_aligned(:,3);

    % === ✅ 保存三维 & 四维数据 ===
    xyz_out = [x, y, z];  % 单位：mm
    xyz_filename = fullfile(vis_save_folder, [name '_3Dpoints.txt']);
    writematrix(xyz_out, xyz_filename, 'Delimiter', 'tab');
    fprintf('  - 三维点云已保存: %s\n', xyz_filename);
    
    if length(T) == size(xyz_aligned, 1)
        xyzt_out = [x, y, z, T];
    else
        min_len = min(length(T), size(xyz_aligned, 1));
        xyzt_out = [x(1:min_len), y(1:min_len), z(1:min_len), T(1:min_len)];
    end
    xyzt_filename = fullfile(vis_save_folder, [name '_4Dpoints.txt']);
    writematrix(xyzt_out, xyzt_filename, 'Delimiter', 'tab');
    fprintf('  - 四维点云已保存: %s\n', xyzt_filename);

    % === 4. LOWESS曲面拟合 ===
    x_center = (min(x) + max(x))/2; y_center = (min(y) + max(y))/2;
    x_shifted = x - x_center; y_shifted = y - y_center;
    x_range = linspace(min(x_shifted), max(x_shifted), grid_size);
    y_range = linspace(min(y_shifted), max(y_shifted), grid_size);
    [xq, yq] = meshgrid(x_range, y_range);
    ft = fit([x_shifted, y_shifted], z, 'lowess', 'Span', lowess_span);
    zq = feval(ft, xq, yq);
    zq_um = min(max(zq * 1000, -10), 50);  % μm 裁剪
    warpage_um = max(zq_um(:)) - min(zq_um(:));

    % === 5. 温度插值 ===
    F_temp = scatteredInterpolant(x_shifted, y_shifted, T, 'linear', 'none');
    Tq = F_temp(xq, yq);
    Tq_min = nanmin(T(:)); Tq_max = nanmax(T(:));
    Tq_norm = (Tq - Tq_min) / (Tq_max - Tq_min);
    Tq_norm(isnan(Tq_norm)) = 0;

    % === 6. 四维图像保存 ===
    fig = figure('Visible', 'off');
    surf(xq*0.25, yq*0.25, zq_um, Tq, 'EdgeColor', 'none');
    shading interp; colormap(jet); caxis([Tq_min, Tq_max]);
    cb = colorbar;
    cb.Label.String = 'Temperature (°C)';
    cb.Label.FontSize = 12;
    xlabel('X (mm)', 'Interpreter','none');
    ylabel('Y (mm)', 'Interpreter','none');
    zlabel('Z (μm)', 'Interpreter','none');
    tokens = regexp(name, '_', 'split');
    if ~isempty(tokens) && length(tokens) >= 2
        temp_label = tokens{2};
        temp_value = str2double(temp_label);
        index_value = str2double(tokens{1});
        if index_value < 5
            temp_title = sprintf('Warm up %s°C', temp_label);
        else
            temp_title = sprintf('Cooling %s°C', temp_label);
        end
    else
        temp_title = name;
    end
    title(['4D Visualization - ', temp_title], 'Interpreter','none');
%     title(['4D Visualization - ', name], 'Interpreter','none');
    axis equal; view(45,30);
    set(gca,'Color','w'); set(gcf,'Color','w');
    saveas(fig, fullfile(vis_save_folder, [name '_4Dvis.png']));
    close(fig);

    % === 7. 对角线剖面图绘制（无差值线） ===
    diag1_z = diag(zq_um); diag1_x = diag(xq); diag1_y = diag(yq);
    diag1_d = sqrt((diag1_x - diag1_x(1)).^2 + (diag1_y - diag1_y(1)).^2);
    diag1_warpage = max(diag1_z) - min(diag1_z);

    diag2_z = diag(flipud(zq_um)); 
    diag2_x = diag(fliplr(xq)); diag2_y = diag(fliplr(yq));
    diag2_d = sqrt((diag2_x - diag2_x(1)).^2 + (diag2_y - diag2_y(1)).^2);
    diag2_warpage = max(diag2_z) - min(diag2_z);

    % 保存图像
    fig_diag = figure('Visible', 'off');
    plot(diag1_d*0.25, diag1_z, '-r', 'LineWidth', 1.5); hold on;
    plot(diag2_d*0.25, diag2_z, '-b', 'LineWidth', 1.5);
    legend('Main Diagonal', 'Secondary Diagonal', ...
        'Location', 'best', 'Interpreter','none');
    xlabel('Distance along diagonal (mm)', 'Interpreter','none');
    ylabel('Warpage(μm)', 'Interpreter','none');
    title(['Diagonal Profile - ', temp_title], 'Interpreter','none');

    subtitle(sprintf('Surface: %.2f μm | Diag1: %.2f μm | Diag2: %.2f μm', ...
        warpage_um, diag1_warpage, diag2_warpage), 'Interpreter','none');
    grid on;
    saveas(fig_diag, fullfile(vis_save_folder, [name '_diagonal_diff.png']));
    close(fig_diag);

    % === 记录统计数据 ===
    warpage_record{end+1,1} = name;
    warpage_record{end,2} = warpage_um;
    warpage_record{end,3} = diag1_warpage;
    warpage_record{end,4} = diag2_warpage;
end
%% === 保存翘曲度结果 ===
T_result = cell2table(warpage_record, ...
    'VariableNames', {'FileName', 'SurfaceWarpage_um', 'Diag1Warpage_um', 'Diag2Warpage_um'});
writetable(T_result, 'warpage_result.csv');
fprintf('\n📄 翘曲度结果已保存至 warpage_result.csv\n');

%% === 保存点数统计结果 ===
sampling_table = cell2table(sampling_stats, ...
    'VariableNames', {'FileName', 'RawPointCount', 'SORFilteredCount', 'DownsampledCount'});
writetable(sampling_table, 'pointcloud_stats.csv');
fprintf('📊 点云采样统计结果已保存至 pointcloud_stats.csv\n');

%% === 合成四维GIF动画 ===
fprintf('\n🎬 正在生成4D动画...\n');
gif_path = fullfile(vis_save_folder, 'AllSamples_4D.gif');
frame_list = dir(fullfile(vis_save_folder, '*_4Dvis.png'));

if length(frame_list) < 2
    fprintf('⚠ 样本不足，无法生成动画\n');
else
    fig_gif = figure('Visible', 'off');
    for k = 1:length(frame_list)
        img = imread(fullfile(vis_save_folder, frame_list(k).name));
        imshow(img);
        % === 提取温度标签用于帧标题 ===
frame_name = frame_list(k).name;
tokens = regexp(frame_name, '_', 'split');
if ~isempty(tokens) && length(tokens) >= 2
    temp_label = tokens{2};
    temp_value = str2double(temp_label);
    index_value = str2double(tokens{1});
    if index_value < 5
        temp_title = sprintf('Warm up %s°C', temp_label);
    else
        temp_title = sprintf('Cooling %s°C', temp_label);
    end
else
    temp_title = frame_name;
end

title(sprintf('Frame %d / %d: %s', k, length(frame_list), temp_title), 'Interpreter','none');

%         title(sprintf('Frame %d / %d: %s', k, length(frame_list), frame_list(k).name), 'Interpreter','none');
        drawnow;
        frame = getframe(fig_gif);
        im = frame2im(frame);
        [imind, cm] = rgb2ind(im, 256);
        if k == 1
            imwrite(imind, cm, gif_path, 'gif', 'Loopcount', inf, 'DelayTime', 0.6);
        else
            imwrite(imind, cm, gif_path, 'gif', 'WriteMode', 'append', 'DelayTime', 0.6);
        end
    end
    close(fig_gif);
    fprintf('✅ 动画已生成: %s\n', gif_path);
end

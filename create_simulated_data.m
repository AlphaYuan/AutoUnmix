

function [ideal_image, mixed_image, ideal_spectra] = create_simulated_data(num_fluorophores, snr, cluster_size_ratio, plots, image, img_name, image_size, autofluorescence_strength, af_fraction, fluorophore_peaks)
    global autofluorescence_distribution autofluorescence_peak

    save_path = "./SimulatedData/simulated_mixed_image";
    sparse_image_save_path = "./SimulatedData/simulated_unmixed_image_ideal";
    save_type = "single_tiff";
    %microscope setup details
    spectral_peak_stdev = 10; 
    laser_wavelengths = [750, 970];
    %filter_cubes = {0:485; 485:570; 570:650; 650:690;}; %DM485 + DM650 + DM570 
    filter_cubes = {420:460; 495:540; 575:630; 645:685;}; %B/G + R/fR + DM570
    cube_colors = {[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1]};
    num_channels = 4;

    %create distribution
    if ~exist('fluorophore_peaks','var')
        fluorophore_peaks = floor(linspace(420, 685, num_fluorophores));
    end
    
    if ~exist('autofluorescence_strength','var')
        autofluorescence_strength = -1;
    end
    
    if ~exist('af_fraction','var')
        af_fraction = 0;
    end
    
    %autofluorescence distribution
    pd = makedist('Weibull','a',250,'b',2);
    af_distribution_peak = 178;
    x_min = 0;
    x_max = 600;
    x = x_min:1:x_max;
    autofluorescence_distribution = pdf(pd, x)/max(pdf(pd, x))*autofluorescence_strength;
    autofluorescence_peak = 485;
    
    %fluorophore distribution
    pd = makedist('Weibull','a',100,'b',1.7);
    distribution_peak = 59;
    x_min = 0;
    x_max = 300;
    x = x_min:1:x_max;
    fluorophore_distribution = pdf(pd, x)/max(pdf(pd, x));
    
    
    %plot distributions
    
    
    if plots
        figure
        for i=1:num_fluorophores
            peak_rgb{i} = spectrumRGB(fluorophore_peaks(i));
            key(i) = "Fluorophore " + int2str(i);
            x_range{i} = x + fluorophore_peaks(i) - distribution_peak;
            plot(x_range{i}, fluorophore_distribution, 'Color', peak_rgb{i});
            hold on

            shade = false;
            if shade
                x_left = x_range{i}(1:60);
                dist_left = fluorophore_distribution(1:60);
                x_right = x_range{i}(60:end);
                dist_right = fluorophore_distribution(60:end);

                lower_left_x = x_left - spectral_peak_stdev;
                upper_left_x = x_left + spectral_peak_stdev;
                lower_right_x = x_right - spectral_peak_stdev;
                upper_right_x = x_right + spectral_peak_stdev;

                x2_left = [lower_left_x, fliplr(upper_left_x)];
                inBetween_left = [dist_left, fliplr(dist_left)];
                x2_right = [lower_right_x, fliplr(upper_right_x)];
                inBetween_right = [dist_right, fliplr(dist_right)];

                fill(x2_left, inBetween_left, 'r');
                fill(x2_right, inBetween_right, 'r');
                
                plot(x_range{i}, fluorophore_distribution, 'k', 'LineWidth', 2);
            end
            
            
        end
        
        %plot autofluorescence
        key(i+1) = "Autofluorescence";
        x_range = (0:1:600) + autofluorescence_peak - af_distribution_peak;
        plot(x_range, autofluorescence_distribution, 'k');
        title('Fluorophore Emission Spectra');
        xlabel('Wavelength [nm]');
        ylabel('Relative Intensity');
        legend(key);
        set(gca,'fontname','times')
        

    end
    
    
    %TODO show detection channels on plot
   
    %get channel intensities for each fluorohpore
    for i=1:num_fluorophores
        channel_intensities(i, :) = get_channel_intensities(fluorophore_peaks(i), fluorophore_distribution, filter_cubes, 0);  
    end
    
    ideal_spectra = transpose(channel_intensities);
    
    if plots
        figure
        bar_plot = bar(channel_intensities./max(channel_intensities, [], 'all'),'LineStyle', 'none');
        xlabel('Fluorophore');
        ylabel('Relative Intensity')
        title('Synthetic Spectral Signatures')
        legend(["CH1", "CH2", "CH3", "CH4"]);
        ylim([0, 1.1]);
        set(gca,'fontname','times')
        for i = 1:size(channel_intensities, 2)
            bar_plot(i).FaceColor = cube_colors{i};
        end
    end
    
    
    %create fluorophore pattern
%     sparse_image = get_grid_of_fluorophores(num_fluorophores, cluster_size_ratio, image_size);
%     perc_background = sum(sparse_image==0, 'all')/(image_size^2);
%     
%     if plots
%         ground_truth = zeros(image_size, image_size, 3);
%         for x = 1:image_size
%             for y = 1:image_size
%                 fluor = sparse_image(x,y);
%                 if fluor ~=0
%                     ground_truth(x, y, :) = peak_rgb{fluor};
%                 end
%             end
%         end
%         figure;
%         imshow(ground_truth, []);   
%         write_tiff(ground_truth, "./ground_truth", "single_tiff")
%     end

%     sparse_image = ;

    %convert fluorophore pattern to multi-channel images
    maximum_signal = max(channel_intensities(:));
    multi_channel_image = new_get_multi_channel_image(image, snr, fluorophore_peaks, spectral_peak_stdev, fluorophore_distribution, filter_cubes, af_fraction);
    
    if plots
        figure
        for c=1:num_channels
           subplot(1,4,c);
           imshow(multi_channel_image(:, :, c), [0, max(multi_channel_image(:))]);
           title("4 channel "+ int2str(c));
           set(gca,'fontname','times')
        end
    %figure
%     imshow(multi_channel_image(:, :, 2:4)/4, [0 max(multi_channel_image(:))]);
    end
    %}
    
    ideal_image = image;%get_ideal_multi_channel_image(sparse_image, num_fluorophores);
    
    if false
        for c=1:num_fluorophores+1
            figure
            imshow(ideal_image(:, :, c), [0 max(ideal_image(:))]);
            title("All channel "+ int2str(c));
        end
    end
    
    %scale
    mixed_image = multi_channel_image/max(multi_channel_image(:));
    %close all;
    
    %save
    write_tiff(multi_channel_image, "./SimulatedData/" + img_name, save_type)
    %write_tiff(sparse_image_unmixed, sparse_image_save_path, save_type)
    
    
    write_tiff(mixed_image, "./simulated_mix_img/" + img_name, save_type);% "multiple_tiff")
    size(mixed_image);
end

function multi_channel_image = new_get_multi_channel_image(sparse_image, snr, fluorophore_peaks, spectral_peak_stdev, fluorophore_distribution, filter_cubes, af_fraction)
    num_channels = length(filter_cubes);
    [x_dim, y_dim, z_dim] = size(sparse_image);
    multi_channel_image = zeros(x_dim, y_dim, num_channels);
    total_signal=0;
    num_signal_pixels = 0;
    af_pixels_count = 0;
    
    background_pixels_total = sum(sparse_image(:)==0);
    af_pixels_total = background_pixels_total * af_fraction;
    
    % set autofluorescence pixel to -1
%     for i = 1:512
%         for j = 1:512
%             if af_pixels_count < af_pixels_total && sparse_image(i, j) == 0
%                af_pixels_count = af_pixels_count+1;
%                sparse_image(i, j) = -1;
%             end
%         end
%         for k = 1:512
%             if af_pixels_count < af_pixels_total && sparse_image(k, i) == 0
%                af_pixels_count = af_pixels_count+1;
%                sparse_image(k, i) = -1;
%             end
%         end      
%     end
    
    use_background_noise = true;
    
    if ~use_background_noise
       for warnings = 1:10
           disp("NO BACKGROUND NOISE")
       end 
    end
    
    for x=1:x_dim
        for y=1:y_dim
%             pixel_fluorophore = sparse_image(x, y);
%             if pixel_fluorophore>0
%                 num_signal_pixels=num_signal_pixels+1;
%                 channel_intensities = get_channel_intensities(fluorophore_peaks(pixel_fluorophore), fluorophore_distribution, filter_cubes, spectral_peak_stdev);
%                 total_signal = total_signal+sum(channel_intensities(:));
%             elseif pixel_fluorophore == -1 %autofluorescence
%                 channel_intensities = get_channel_intensities(-1, fluorophore_distribution, filter_cubes, spectral_peak_stdev);
%                 af_pixels_count = af_pixels_count + 1;
%             else
%                 channel_intensities = zeros(1, 4);
%             end
            channel_intensities = zeros(4, 4);% channel, fluorophore
            if sum(sparse_image(x,y,:))>0
                num_signal_pixels=num_signal_pixels+1;
                for i=1:4
                     channel_intensities(i, :) = get_channel_intensities(fluorophore_peaks(i), fluorophore_distribution, filter_cubes, spectral_peak_stdev);
                end
                total_signal = total_signal+sum(channel_intensities(:));
            end

            if use_background_noise
                background_noise = max(normrnd(2, 1, [4, 4]), 0);
            else
                background_noise = 0;
            end
            
            channel_intensities = max(channel_intensities, background_noise);
            
%             channel_intensities
            
            for i=1:num_channels
                tmp = sparse_image(x, y, i);
                for j=1:num_channels
                    if j ~= i
                        tmp = tmp + sparse_image(x, y, j) * channel_intensities(i, j) / channel_intensities(i, i);
                    end
                end
                multi_channel_image(x, y, i) = tmp;
            end

%             multi_channel_image(x, y, 1:num_channels) = sparse_image(x, y, 1:num_channels)
        end
    end
    
    mean_signal = total_signal/num_signal_pixels/num_channels;  %mean of non-background pixels
    
%     desired_mean = snr^2;
%     multi_channel_image = multi_channel_image/mean_signal*desired_mean;
    
    multi_channel_image = poissrnd(multi_channel_image); %adds poisson noise

    
    for c=1:num_channels
        multi_channel_image(:, :, c) = imgaussfilt(multi_channel_image(:, :, c));
        multi_channel_image(:, :, c) = medfilt2(multi_channel_image(:, :, c));
    end
    
    minimum_signal = 0;
    multi_channel_image = max(multi_channel_image, minimum_signal);
    saturation = false;
    if saturation
        maximum_signal = 512;
        multi_channel_image = min(multi_channel_image, maximum_signal);
    end
end

%turn sparse image into multi-channel image
function multi_channel_image = get_multi_channel_image(sparse_image, snr, fluorophore_peaks, spectral_peak_stdev, fluorophore_distribution, filter_cubes, af_fraction)
    num_channels = length(filter_cubes);
    [x_dim, y_dim] = size(sparse_image);
    multi_channel_image = zeros(x_dim, y_dim, num_channels);
    total_signal=0;
    num_signal_pixels = 0;
    af_pixels_count = 0;
    
    background_pixels_total = sum(sparse_image(:)==0);
    af_pixels_total = background_pixels_total * af_fraction;
    
    for i = 1:512
        for j = 1:512
            if af_pixels_count < af_pixels_total && sparse_image(i, j) == 0
               af_pixels_count = af_pixels_count+1;
               sparse_image(i, j) = -1;
            end
        end
        for k = 1:512
            if af_pixels_count < af_pixels_total && sparse_image(k, i) == 0
               af_pixels_count = af_pixels_count+1;
               sparse_image(k, i) = -1;
            end
        end      
    end
    
    use_background_noise = true;
    
    if ~use_background_noise
       for warnings = 1:10
           disp("NO BACKGROUND NOISE")
       end 
    end
    for x=1:x_dim
        for y=1:y_dim
            pixel_fluorophore = sparse_image(x, y);
            if pixel_fluorophore>0
                num_signal_pixels=num_signal_pixels+1;
                channel_intensities = get_channel_intensities(fluorophore_peaks(pixel_fluorophore), fluorophore_distribution, filter_cubes, spectral_peak_stdev);
                total_signal = total_signal+sum(channel_intensities(:));
            elseif pixel_fluorophore == -1 %autofluorescence
                channel_intensities = get_channel_intensities(-1, fluorophore_distribution, filter_cubes, spectral_peak_stdev);
                af_pixels_count = af_pixels_count + 1;
            else
                channel_intensities = zeros(1, 4);
            end

            if use_background_noise
                background_noise = max(normrnd(2, 1, [1, 4]), 0);
            else
                background_noise = 0;
            end
            
            channel_intensities = max(channel_intensities, background_noise);

            multi_channel_image(x, y, 1:num_channels) = channel_intensities;
        end
    end
    
    mean_signal = total_signal/num_signal_pixels/num_channels;  %mean of non-background pixels
    
    desired_mean = snr^2;
    multi_channel_image = multi_channel_image/mean_signal*desired_mean;
    
    multi_channel_image = poissrnd(multi_channel_image); %adds poisson noise

    
    for c=1:num_channels
        multi_channel_image(:, :, c) = imgaussfilt(multi_channel_image(:, :, c));
        multi_channel_image(:, :, c) = medfilt2(multi_channel_image(:, :, c));
    end
    
    minimum_signal = 0;
    multi_channel_image = max(multi_channel_image, minimum_signal);
    saturation = false;
    if saturation
        maximum_signal = 512;
        multi_channel_image = min(multi_channel_image, maximum_signal);
    end
end

%turn sparse image into ideal unmixed image
function multi_channel_image = get_ideal_multi_channel_image(sparse_image, num_fluorophores)
    [x_dim, y_dim] = size(sparse_image);
    multi_channel_image = zeros(x_dim, y_dim, num_fluorophores+1);
    for x=1:x_dim
        for y=1:y_dim
            pixel_fluorophore = sparse_image(x, y);
            multi_channel_image(x, y, pixel_fluorophore+1) = 1;
        end
    end
end

%Get channel intensities for a given pixel
function channel_intensities = get_channel_intensities(fluorophore_peak, fluorophore_distribution, filter_cubes, spectral_center_stdev)
    global autofluorescence_distribution autofluorescence_peak
    num_detection_channels = length(filter_cubes);
    
    
    if fluorophore_peak == -1 %flag for autofluorescence
        fluorophore_distribution = autofluorescence_distribution;
        fluorophore_peak = autofluorescence_peak + normrnd(0, spectral_center_stdev);
    else
        fluorophore_peak = fluorophore_peak + normrnd(0, spectral_center_stdev);  
    end
    
    [~, distribution_peak] = max(fluorophore_distribution);
    x_max = length(fluorophore_distribution);
    x_min = 1;
    for c=1:num_detection_channels
        filter_cube_range = filter_cubes{c};
        min_wavelength = int64(min(max(min(filter_cube_range)-fluorophore_peak+distribution_peak, x_min), x_max));
        max_wavelength = int64(max(min(max(filter_cube_range)-fluorophore_peak+distribution_peak, x_max), x_min));
        channel_intensities(c) = sum(fluorophore_distribution(min_wavelength:max_wavelength)); 
    end
end

%create automated test pattern that works for any number of fluorophores
function sparse_image = get_grid_of_fluorophores(num_fluorophores, cluster_size_ratio, image_size)
    sparse_image = zeros(image_size);
    width = floor(image_size/(num_fluorophores*2));
    %thin_width = width*cluster_size_ratio
    start = floor(width/2);
    small_cluster_number = num_fluorophores; %always have smallest cluster last %2*floor(num_fluorophores/4)+1;%round to nearest odd number so that it is always a column floor((num_fluorophores+1)/2);
    for i = 1:num_fluorophores
        start = start+width*1.2;
        small_cluster = i == small_cluster_number;
        %if small_cluster
        %    small_cluster_start = start;
        %else
            row = mod(i, 2) == 0;
            sparse_image = add_row_or_column(sparse_image, start, width, cluster_size_ratio, row, i, small_cluster, image_size); 
        %end
    end
    %Overlay smallest cluster on top of the rest so its number of pixels depends only on cluster size ratio
    %sparse_image = add_row_or_column(sparse_image, small_cluster_start, width, cluster_size_ratio, row, small_cluster_number, true, image_size);
    
    
end

%function to add a fluorophore as a single single row/column 
function sparse_image = add_row_or_column(sparse_image, start, width, cluster_size_ratio, row, fluorophore, small_cluster, image_size)
    start = int64(floor(start));
    %width = int64(floor(width));

    if small_cluster
        thin_width = ceil(max(width*cluster_size_ratio));
        length = floor(max(width*image_size*cluster_size_ratio/thin_width, 1));
        width = thin_width;
    else
        length = image_size;
    end
        
    if row
        sparse_image(start:start+int64(width), 1:int64(length))  = fluorophore;
    else
        sparse_image(1:int64(length), start:start+int64(width))  = fluorophore;
    end
end

function write_tiff(image, save_path, save_type)
    dims = size(image);
    
    if length(dims) >=3
        num_slices = dims(3);
        %num_timesteps = dims(3);
    else
        num_slices = 1;
        %num_timesteps = 1;  
    end
    if length(dims) >= 4
        num_channels = dims(4);
    else
        num_channels = 1;
    end
    if length(dims) >= 5
        %num_slices = dims(5);
        num_timesteps = dims(5);
    else
        num_timesteps = 1;
        %num_slices = 1;
    end
        
    
    if save_type == "single_tiff"
        save_path = strcat(save_path, ".tif");
        delete(save_path);
        for t = 1:num_timesteps
            for z = 1:num_slices
                for c = 1:num_channels
%                     disp(int2str(z) + " " + int2str(c) + " " +int2str(t));
                    %imwrite(double(squeeze(image(:, :, t, c, z))), save_path, 'WriteMode', 'append', 'Compression', 'none');
                    imwrite(double(squeeze(image(:, :, z, c, t))), save_path, 'WriteMode', 'append', 'Compression', 'none');
                end
            end
        end
    elseif save_type == "multiple_tiffs"
        mkdir(save_path)
        for t = 1:num_timesteps
            for z = 1:num_slices
                for c = 1:num_channels
                    fname_2d = strcat('s_C', sprintf('%03d', c), 'Z', sprintf('%03d', z), 'T', sprintf('%03d', t), '.tif');
                    fname_2d = fullfile(save_path, fname_2d);
                    delete(fname_2d)
                    imwrite(double(squeeze(image(:, :, z, c, t))), fname_2d, 'WriteMode', 'append', 'Compression', 'none');
                    %imwrite(double(squeeze(image(:, :, t, c, z))), fname_2d, 'WriteMode', 'append', 'Compression', 'none');
                end
            end
        end
    end
end
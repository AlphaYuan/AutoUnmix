clc; clear; close all;

addpath("../LUMoS/spectral_color");

n = 4;
noise = 0.25;
size = 0.2;
plots = false;
image_size = 512;
autofluorescence_strength = 0;
af_fractions = 0;

data_path = "data/img_all/";
file_ext = "*.jpg";

files = dir(fullfile(data_path, file_ext));
length(files)

for i=1:length(files)
    if i>=231 && i<=278
        continue
    end
    disp("processing image " + int2str(i));
%     file_name = strcat(data_path, files(i).name);
%     img = imread(file_name);
%     img = imresize(img, [img_size, img_size]);
    tic

    img = zeros(image_size, image_size, n);
    img(:,:,1) = imresize(imread(data_path + "mixed_img" + int2str(i) + "_blue.jpg"), [image_size, image_size]);
    img(:,:,2) = imresize(imread(data_path + "mixed_img" + int2str(i) + "_green.jpg"), [image_size, image_size]);
    img(:,:,3) = imresize(imread(data_path + "mixed_img" + int2str(i) + "_yellow.jpg"), [image_size, image_size]);
    img(:,:,4) = imresize(imread(data_path + "mixed_img" + int2str(i) + "_red.jpg"), [image_size, image_size]);

%     figure
%     for i=1:n
%         subplot(1,4,i);
%         imshow(img(:,:,i), [0 max(img(:))]);
%         title("channel " + int2str(i));
%     end

    [ideal_image, mixed_image, ideal_spectra] = create_simulated_data(n, noise, size, plots, img, "img_" + int2str(i), image_size, autofluorescence_strength, af_fractions);
    toc
end

from utils.image_io import *
from utils.Gauss_denoise_class import Denoise
import os
from os import listdir, getcwd
from os.path import join

def img_preprocess(img_name, sigma, times):
    img_org = prepare_image(img_name)
    sigma_ = sigma / 255.
    img_noise = []
    for i in range(times):
        _, temp_img_noise = get_noisy_image(img_org, sigma_)
        #_, temp_img_noise = get_salt_noisy_image(img_org, sigma)
        img_noise.append(temp_img_noise)
    return img_org, img_noise


def denoise(image_name, output_dir, img_org, img_noise, learning_rate=0.01, num_iter=6000, show_every=5000):
    print(output_dir)
    plot_during_training = True
    de = Denoise(image_name, output_dir, img_org, img_noise, learning_rate, num_iter, show_every, plot_during_training)
    de.optimize()
    de.finalize()



if __name__ == "__main__":
      # noise var
    output_dir = "/output/"
    number_img = 4
    num_iter = 5000
    show_every = 500000
    learning_rate = 0.01
    source_folder = "./dataset/input/"
    file_list = os.listdir(source_folder)
    sigma = 50
    print(sigma)
    for file_obj in file_list:
        img_name = file_obj
        print(img_name)
        img_org, img_noise = img_preprocess(source_folder + img_name, sigma, number_img)
        denoise("multi-" + img_name.split('.')[0], output_dir, img_org, img_noise, learning_rate, num_iter, show_every)

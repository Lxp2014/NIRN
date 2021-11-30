from utils.image_io import *
from utils.real_image_denoise_class import *
import os
from os import listdir, getcwd
from os.path import join

def img_preprocess(GT_name, noise_name):
    img_org = prepare_image(GT_name)
    img_noise = []
    for i in range(1):
        img_temp = prepare_image(noise_name)
        img_noise.append(img_temp)
    return img_org, img_noise


def denoise(image_name, output_dir, img_org, img_noise, learning_rate=0.01, num_iter=6000, show_every=5000):
    plot_during_training = True
    de = Denoise(image_name, output_dir, img_org, img_noise, learning_rate, num_iter, show_every, plot_during_training)
    de.optimize()
    de.finalize()


if __name__ == "__main__":

    noise_dir = "./noiseimage/"
    gt_dir = "./gtimage/"
    output_dir = "./output/"
    GT_name = "./gtimage/1.jpg"
    num_iter = 5000
    show_every = 500000
    learning_rate = 1e-2

    file_list = os.listdir(noise_dir)
    for file_obj in file_list:
        img_name = file_obj
        print(img_name)
        img_org, img_noise = img_preprocess(GT_name, noise_dir + img_name)
        denoise(img_name.split('.')[0], output_dir, img_org, img_noise, learning_rate, num_iter, show_every)

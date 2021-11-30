from utils.image_io import *
from utils.true_denoise_class import Denoise




def img_preprocess(input_dir, times):
    img_noise = []
    for i in range(times):
        img_name =str(i+1) + ".jpg"
        img_org = prepare_image(input_dir + img_name)
        img_noise.append(img_org)
    return img_noise


def denoise(image_name, output_dir, img_noise, learning_rate = 0.01, num_iter=6000, show_every=5000):
    plot_during_training = True
    de = Denoise(image_name, output_dir, img_noise, learning_rate, num_iter, show_every, plot_during_training)
    de.optimize()
    de.finalize()




if __name__ == "__main__":
    number_img = 4
    img_name = "night.png"
    input_dir = "./multi-input/"
    output_dir = "./output/"

    num_iter = 5000
    show_every = 500000
    learning_rate = 0.01

    img_noise = img_preprocess(input_dir, number_img)

    denoise(img_name.split('.')[0], output_dir, img_noise, learning_rate, num_iter, show_every)



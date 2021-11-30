from utils.image_io import *
from utils.real_image_denoise_class import Denoise




def img_preprocess(GT_name, input_dir, times):
    img_noise = []
    img_org = prepare_image(GT_name)
    for i in range(times):
        img_name =str(i+1) + ".jpg"
        print(img_name)
        img_temp = prepare_image(input_dir + img_name)
        img_noise.append(img_temp)
    return img_org, img_noise


def denoise(image_name, output_dir, img_org, img_noise, learning_rate = 0.01, num_iter=6000, show_every=5000):
    plot_during_training = True
    de = Denoise(image_name, output_dir, img_org, img_noise, learning_rate, num_iter, show_every, plot_during_training)
    de.optimize()
    de.finalize()




if __name__ == "__main__":
    number_img = 4
    img_name = "multi-1.jpg"
    input_dir = "./noiseimage/"
    output_dir = "./output/"
    GT_name = "./gtimage/1.jpg"

    num_iter = 5000
    show_every = 500000
    learning_rate = 1e-2

    img_org, img_noise = img_preprocess(GT_name, input_dir, number_img)

    denoise(img_name.split('.')[0], output_dir, img_org, img_noise, learning_rate, num_iter, show_every)



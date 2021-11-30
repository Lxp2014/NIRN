from BM3D.utils import add_gaussian_noise, symetrize
from BM3D.bm3d_1st_step import bm3d_1st_step
from BM3D.bm3d_2nd_step import bm3d_2nd_step
import numpy as np


def run_bm3d(noisy_im, sigma,
             n_H, k_H, N_H, p_H, tauMatch_H, useSD_H, tau_2D_H, lambda3D_H,
             n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W):
    k_H = 8 if (tau_2D_H == 'BIOR' or sigma < 40.) else 12
    k_W = 8 if (tau_2D_W == 'BIOR' or sigma < 40.) else 12

    noisy_im_p = symetrize(noisy_im, n_H)  #pad
    img_basic = bm3d_1st_step(sigma, noisy_im_p, n_H, k_H, N_H, p_H, lambda3D_H, tauMatch_H, useSD_H, tau_2D_H)
    img_basic = img_basic[n_H: -n_H, n_H: -n_H]

    # assert not np.any(np.isnan(img_basic))
    # img_basic_p = symetrize(img_basic, n_W)
    # noisy_im_p = symetrize(noisy_im, n_W)
    # img_denoised = bm3d_2nd_step(sigma, noisy_im_p, img_basic_p, n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W)
    # img_denoised = img_denoised[n_W: -n_W, n_W: -n_W]
    img_denoised = 0

    return img_basic, img_denoised

# NIRN
NIRN: Self-supervised Noisy Image Reconstruction Network for Real-World Image Denoising
## Environments
- Ubuntu 16.04
- CUDA 10.0 & cuDNN 7.5.0
- Python 3.7.6

Environment configuration:

conda env create -f NIRN.yml

conda activate NIRN

The package comprises these functions：

*) single_free_denoise.py        : Denoise single-frame real-world noisy image 

*) multi_free_denoise.py       : Denoise multi-frame real-world noisy images（frame = 4）

*) single_real_image_denoise.py       : Denoise single-frame real-world noisy image， we provide noise-free image to get PSNR result 

*) multi_real_image_denoise.py        : Denoise multi-frame real-world noisy images（frame = 4）， we provide noise-free image to get PSNR result 

*) single_Gauss_denoise.py      : Denoise single-frame image with Gaussian noise (sigma = 50) 

*) multi_Gauss_denoise.py      : Denoise multi-frame image with Gaussian noise (sigma = 50) （frame = 4）

Implementation Details:
Here we save the output from both subnetworks IGM and NGM with the suffixes '_CIGM_denoised.png' and '_NGM_denoised.png'
When changing the picture, it is better to use the name of our picture, otherwise an error may occur.

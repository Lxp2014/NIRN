import numpy as np


def get_transport_mat(im_s, k):
    temp = np.zeros((im_s, (im_s - k + 1) * k), dtype=np.int)
    for i in range(k):
        temp[i, i] = 1
    Trans = temp.copy()
    for i in range(1, im_s - k + 1):
        dT = np.roll(temp, i, axis=0)
        dT = np.roll(dT, i * k, axis=1)
        Trans += dT
    return Trans


# def image2patches(im, k, p):
#     '''
#     :param im:
#     :param k: patch size
#     :param p: step
#     :return:
#     '''
#     assert im.ndim == 2
#     assert im.shape[0] == im.shape[1]
#     im_s = im.shape[0]
#
#     Trans = get_transport_mat(im_s, k)
#     repetition = Trans.T @ im @ Trans
#     repetition = repetition.reshape((im_s - k + 1, k, im_s - k + 1, k))
#     repetition = repetition.transpose((0, 2, 1, 3))
#     repetition = repetition.reshape((-1, k, k))
#     return repetition

def image2patches(im, patch_h, patch_w):
    im_h, im_w = im.shape[0], im.shape[1]
    patch_table = np.zeros((im_h - patch_h + 1, im_w - patch_w + 1, patch_h, patch_w), dtype=np.float64)
    for i in range(im_h - patch_h + 1):
        for j in range(im_w - patch_w + 1):
            patch_table[i][j] = im[i:i + patch_h, j:j + patch_w]

    return patch_table




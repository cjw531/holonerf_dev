import numpy as np
import math
import cv2
import math
import torch
import lpips
from PIL import Image
import torchvision.transforms.functional as TF


def mse(img1, img2):
    mse = np.mean(np.subtract(img1, img2) ** 2)
    return mse

def sq_err(img1, img2):
    return np.sum(np.subtract(img1, img2))

# reference: https://dsp.stackexchange.com/a/50704
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python
def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_lpips(im0_addr, im1_addr, resize=1):
    loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
    # loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

    img0 = torch.zeros(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
    img1 = torch.zeros(1,3,64,64)
    print(img0.shape)

    img0 = Image.open(im0_addr).convert('RGB')
    # width, height = img0.size
    # img0 = img0.resize((width // resize, height // resize))
    im0 = TF.to_tensor(img0)
    img1 = Image.open(im1_addr).convert('RGB')
    # width, height = img1.size
    # img1 = img1.resize((width // resize, height // resize))
    im1 = TF.to_tensor(img1)

    d = loss_fn_alex(im0, im1)

    return d[0][0][0][0]

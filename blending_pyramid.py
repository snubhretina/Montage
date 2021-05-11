import sys
import os
import numpy as np
import cv2
import scipy
from scipy.stats import norm
from scipy.signal import convolve2d
import math
import matplotlib.pyplot as plt

'''split rgb image to its channels'''


def split_rgb(image):
    red = None
    green = None
    blue = None
    (blue, green, red) = cv2.split(image)
    return red, green, blue


'''generate a 5x5 kernel'''


def generating_kernel(a):
    w_1d = 1/float(a) * np.array([1,5,8,5,1])
    return np.outer(w_1d, w_1d)

'LPF with gasussian filter'
def LPF_Gauss(image):
    out = None
    # kernel = generating_kernel(10)
    outimage = cv2.GaussianBlur(image,(5,5),0)
    return outimage



'''reduce image by 1/2'''


def ireduce(image):
    out = None
    kernel = generating_kernel(40)
    outimage = scipy.signal.convolve2d(image, kernel, 'same')
    out = outimage[::2, ::2]
    return out


'''expand image by factor of 2'''


def iexpand(image):
    out = None
    kernel = generating_kernel(40)
    outimage = np.zeros((image.shape[0] * 2, image.shape[1] * 2), dtype=np.float64)
    outimage[::2, ::2] = image[:, :]
    out = 4*scipy.signal.convolve2d(outimage, kernel, 'same')
    return out


'''create a gaussain pyramid of a given image'''


def gauss_pyramid(image, levels):
    output = []
    output.append(image)
    tmp = image
    for i in range(0, levels):
        tmp = ireduce(tmp)
        output.append(tmp)
    return output


'''build a laplacian pyramid'''


def lapl_pyramid(gauss_pyr):
    output = []
    k = len(gauss_pyr)
    for i in range(0, k - 1):
        gu = gauss_pyr[i]
        egu = iexpand(gauss_pyr[i + 1])
        if egu.shape[0] > gu.shape[0]:
            egu = np.delete(egu, (-1), axis=0)
        if egu.shape[1] > gu.shape[1]:
            egu = np.delete(egu, (-1), axis=1)
        output.append(gu - egu)
    output.append(gauss_pyr.pop())
    return output

def convert_weighted_mask(image):
    nonzero_image = np.nonzero(image)
    mask_width = nonzero_image[1].max() - nonzero_image[1].min()
    mask_height = nonzero_image[0].max() - nonzero_image[0].min()
    # center_pt = ((nonzero_image[0].max() + nonzero_image[0].min())/2,(nonzero_image[1].max() + nonzero_image[1].min())/2)
    center_pt = (mask_height/ 2, mask_width/ 2)
    # image_shape = image.shape
    # max_edge = max([mask_height, mask_width])
    # if mask_height%2 == 0:
    #     mask_height -= 1
    # if mask_width%2 == 0:
    #     mask_width -= 1

    x_img = np.repeat(np.abs(np.arange(mask_width) - center_pt[1])[np.newaxis, :], mask_height, axis=0)
    y_img = np.repeat(np.abs(np.arange(mask_height) - center_pt[0])[:, np.newaxis], mask_width, axis=1)
    x_img = np.power(x_img, 2)
    y_img = np.power(y_img, 2)
    center_norm = np.sqrt(center_pt[0]*center_pt[0] + center_pt[1]*center_pt[1])
    dist = np.tile(np.sqrt(x_img + y_img)[:, :, None], [1, 1, 3])/center_norm
    dist = 1 - np.clip(dist, 0, 1)
    dst = np.zeros([image.shape[0],image.shape[1],3])
    dst[nonzero_image[0].min():nonzero_image[0].max(), nonzero_image[1].min():nonzero_image[1].max()] = dist
    dst = cv2.bitwise_and(dst,dst,mask = image)
    #imwir
    return dst


def convert_twin_weighted_mask(src1,src2, pow_opt):

    overlap_region = cv2.bitwise_and(src1,src2)
    image1 = convert_weighted_mask(src1)
    image2 = convert_weighted_mask(src2)
    if pow_opt != 0:
        image1 = cv2.pow(image1, pow_opt)
        image2 = cv2.pow(image2, pow_opt)
    image_summed_weight = image1+image2 + 0.00000001
    result = image1/image_summed_weight
    result2 = image2/image_summed_weight
    # src1 = src1 / 255.
    # src2 = src2 / 255.
    # src1[overlap_region>0][0] = result[overlap_region>0][0]
    # src2[overlap_region>0] = result2[overlap_region>0][0]

    return result, result2

'''Blend the two laplacian pyramids by weighting them according to the mask.'''


def blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask, gauss_pyr_mask2):
    blended_pyr = []
    k = len(gauss_pyr_mask)
    for i in range(0, k):
        p1 = (gauss_pyr_mask[i]) * lapl_pyr_white[i]
        p2 = (gauss_pyr_mask2[i]) * lapl_pyr_black[i]
        overlap = cv2.bitwise_and(gauss_pyr_mask[i],gauss_pyr_mask2[i])
        non_overlap_idx = cv2.bitwise_and((p1 + p2),(p1 + p2),mask = (1 - (overlap>0)*1).astype(np.uint8))

        overlap_idx = cv2.bitwise_and((p1 + p2),(p1 + p2), mask = cv2.threshold(overlap,0,1,cv2.THRESH_BINARY)[1].astype(np.uint8))
        #if save mask for visualizing change 1 to 255
        blended_pyr_idx = non_overlap_idx + overlap_idx
        cv2.imwrite('./tmp/' + str(i) + '_p1.png', p1)
        cv2.imwrite('./tmp/' + str(i) + '_p2.png', p2)
        cv2.imwrite('./tmp/' + str(i) + '_gauss_1.png', gauss_pyr_mask[i]*255)
        cv2.imwrite('./tmp/' + str(i) + '_gauss_2.png', gauss_pyr_mask2[i]*255)
        blended_pyr.append(blended_pyr_idx)
    return blended_pyr


'''Reconstruct the image based on its laplacian pyramid.'''


def collapse(lapl_pyr):
    output = None
    output = np.zeros((lapl_pyr[0].shape[0], lapl_pyr[0].shape[1]), dtype=np.float64)
    for i in range(len(lapl_pyr) - 1, 0, -1):
        lap = iexpand(lapl_pyr[i])
        lapb = lapl_pyr[i - 1]
        if lap.shape[0] > lapb.shape[0]:
            lap = np.delete(lap, (-1), axis=0)
        if lap.shape[1] > lapb.shape[1]:
            lap = np.delete(lap, (-1), axis=1)
        tmp = lap + lapb
        lapl_pyr.pop()
        lapl_pyr.pop()
        lapl_pyr.append(tmp)
        output = tmp
    return output


def main():
    image1 = cv2.imread('./0000_splined_disc_FP.png').astype(np.float64)
    image2 = cv2.imread('./1_ori.png').astype(np.float64)
    mask1 = cv2.imread('./0000_bspline_disc_FP_mask.png',cv2.IMREAD_GRAYSCALE).astype(np.float64)/255.
    mask2 = cv2.imread('./1_mask_ori.png', cv2.IMREAD_GRAYSCALE).astype(np.float64)/255.
    Fmask1 = LPF_Gauss(mask1)
    Fmask2 = 255-LPF_Gauss(mask2)
    GaussImage1 = np.zeros(image1.shape)
    GaussImage2 = np.zeros(image2.shape)
    GaussImage1[:, :, 0] = (image1[:, :, 0] * Fmask1)
    GaussImage1[:, :, 1] = (image1[:, :, 1] * Fmask1)
    GaussImage1[:, :, 2] = (image1[:, :, 2] * Fmask1)
    GaussImage2[:, :, 0] = (image1[:, :, 0] * Fmask2)
    GaussImage2[:, :, 1] = (image1[:, :, 1] * Fmask2)
    GaussImage2[:, :, 2] = (image1[:, :, 2] * Fmask2)
    Result1 = GaussImage1 + GaussImage2
    # cv2.imwrite('./Gaussimage1.png', GaussImage1)
    xormask = cv2.bitwise_xor(Fmask1,Fmask2)
    Fmask1 = LPF_Gauss(mask2)

def pyramid_spline(image1,image2,mask_ori,mask2_ori, pow = 0):
    # image1 = cv2.imread('./0000_splined_disc_FP.png')
    # image2 = cv2.imread('./1_ori.png')
    # mask_ori = cv2.erode(cv2.threshold(cv2.imread('./0000_bspline_disc_FP_mask.png',cv2.IMREAD_GRAYSCALE), 0, 255, cv2.THRESH_BINARY)[1], np.ones((3,3),np.uint8), iterations=7)
    # mask2_ori = cv2.erode(cv2.threshold(cv2.imread('./1_mask_ori.png',cv2.IMREAD_GRAYSCALE), 0, 255, cv2.THRESH_BINARY)[1], np.ones((3,3),np.uint8), iterations=7)
    # image1 = cv2.imread(image)
    # image2 = cv2.imread(image2)
    mask_ori = cv2.erode(
        cv2.threshold(mask_ori, 0, 255, cv2.THRESH_BINARY)[
            1], np.ones((3, 3), np.uint8), iterations=7)
    mask2_ori = cv2.erode(
        cv2.threshold(mask2_ori, 0, 255, cv2.THRESH_BINARY)[1],
        np.ones((3, 3), np.uint8), iterations=7)
    # image1 = cv2.bitwise_and(image1, image1, mask=mask_ori)
    # image2 = cv2.bitwise_and(image2, image2, mask=mask2_ori)
    # mask_ori = cv2.threshold(mask_ori, 0, 255, cv2.THRESH_BINARY)[1]
    # mask2_ori = cv2.threshold(mask2_ori, 0, 255, cv2.THRESH_BINARY)[1]
    # mask = convert_weighted_mask(mask_ori)
    # mask2 = convert_weighted_mask(mask2_ori)
    mask, mask2 = convert_twin_weighted_mask(mask_ori,mask2_ori, pow)

    # mask = convert_weighted_mask(mask_ori)
    # mask2 = convert_weighted_mask(mask2_ori)
    # mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)/255.
    # mask2 = cv2.cvtColor(mask2,cv2.COLOR_GRAY2BGR)/255.
    r1 = None
    g1 = None
    b1 = None
    r2 = None
    g2 = None
    b2 = None
    rm = None
    gm = None
    bm = None

    (r1, g1, b1) = split_rgb(image1)
    (r2, g2, b2) = split_rgb(image2)
    (rm, gm, bm) = split_rgb(mask)
    (rm2, gm2, bm2) = split_rgb(mask2)

    r1 = r1.astype(float)
    g1 = g1.astype(float)
    b1 = b1.astype(float)

    r2 = r2.astype(float)
    g2 = g2.astype(float)
    b2 = b2.astype(float)

    # rm = rm.astype(float) / 255
    # gm = gm.astype(float) / 255
    # bm = bm.astype(float) / 255
    #
    # rm2 = rm2.astype(float) / 255
    # gm2= gm2.astype(float) / 255
    # bm2 = bm2.astype(float) / 255
    # Automatically figure out the size
    min_size = min(r1.shape)
    depth = int(math.floor(math.log(min_size, 2))) + 2  # at least 16x16 at the highest level.

    gauss_pyr_maskr = gauss_pyramid(rm, depth)
    gauss_pyr_maskg = gauss_pyramid(gm, depth)
    gauss_pyr_maskb = gauss_pyramid(bm, depth)

    gauss_pyr_maskr2 = gauss_pyramid(rm2, depth)
    gauss_pyr_maskg2 = gauss_pyramid(gm2, depth)
    gauss_pyr_maskb2 = gauss_pyramid(bm2, depth)

    gauss_pyr_image1r = gauss_pyramid(r1, depth)
    gauss_pyr_image1g = gauss_pyramid(g1, depth)
    gauss_pyr_image1b = gauss_pyramid(b1, depth)

    gauss_pyr_image2r = gauss_pyramid(r2, depth)
    gauss_pyr_image2g = gauss_pyramid(g2, depth)
    gauss_pyr_image2b = gauss_pyramid(b2, depth)

    lapl_pyr_image1r = lapl_pyramid(gauss_pyr_image1r)
    lapl_pyr_image1g = lapl_pyramid(gauss_pyr_image1g)
    lapl_pyr_image1b = lapl_pyramid(gauss_pyr_image1b)

    lapl_pyr_image2r = lapl_pyramid(gauss_pyr_image2r)
    lapl_pyr_image2g = lapl_pyramid(gauss_pyr_image2g)
    lapl_pyr_image2b = lapl_pyramid(gauss_pyr_image2b)

    outpyrr = blend(lapl_pyr_image2r, lapl_pyr_image1r, gauss_pyr_maskr2, gauss_pyr_maskr)
    outpyrg = blend(lapl_pyr_image2g, lapl_pyr_image1g, gauss_pyr_maskg2, gauss_pyr_maskg)
    outpyrb = blend(lapl_pyr_image2b, lapl_pyr_image1b, gauss_pyr_maskb2, gauss_pyr_maskb)

    outimgr = collapse(outpyrr)
    outimgg = collapse(outpyrg)
    outimgb = collapse(outpyrb)
    # blending sometimes results in slightly out of bound numbers.
    outimgr[outimgr < 0] = 0
    outimgr[outimgr > 255] = 255
    outimgr = outimgr.astype(np.uint8)

    outimgg[outimgg < 0] = 0
    outimgg[outimgg > 255] = 255
    outimgg = outimgg.astype(np.uint8)

    outimgb[outimgb < 0] = 0
    outimgb[outimgb > 255] = 255
    outimgb = outimgb.astype(np.uint8)

    result = np.zeros(image1.shape, dtype=image1.dtype)
    tmp = []
    tmp.append(outimgb)
    tmp.append(outimgg)
    tmp.append(outimgr)
    result = cv2.merge(tmp, result)
    return result
    # cv2.imwrite('./blended.jpg', result)
    # cv2.imwrite('./blended.png', result)


if __name__ == '__main__':
    main()
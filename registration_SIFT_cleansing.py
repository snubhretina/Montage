import numpy as np
import skimage
import skimage.transform
import skimage.morphology
import skimage.filters
import cv2, os, inspect, sys
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
from registration_BSpline import regist_BSpline
from numpy.linalg import inv
sys.path.append('./blending_ref')
sys.path.append('./blending_ref/image_spline')
from spline import do_spline
from blending_pyramid import *
from skimage.morphology import skeletonize


class regist_SIFT_montage():
    def __init__(self, img_exist_list, img_None_list, mask, hess_thresh=30, using_detection=False, center_pt = False):
        """
        initalize parameter and if using_detectin is False, registration image set to use registration_series fuction
        :param img_exist_list: fundus image list and vessel image list in dictionary, exist macular or disc image
        :param img_none_list: fundus image list and vessel image list in dictionary, none macular and disc image
        :param mask: image array, field of view image
        :param hess_thresh: int, defined threshold hessian, default 30
        :param using_detection: Bool, if True, using img_exist_list, img_none_list, else using only img_none_list that doesn't detection macular or disc in fundus image
        :param center_pt:
        """

        self.img_list, self.img_vessel_list = img_exist_list
        self.img_none_list, self.img_vessel_none_list = img_None_list
        self.registered_image_count = 0
        self.hess_thresh = hess_thresh
        self.mask = mask
        self.surf_info = {'kpt_list' : [], 'des_list' : []}
        self.surf_no_fp_info = {'kpt_list': [], 'des_list': []}
        self.branching_error_list = []
        self.branching_pt_list = []
        self.matching_distance = 0.7
        self.using_VPmap = True
        #before source base image listed in first
        self.base_idx = 0
        if using_detection == False:
            try:
                self.base_idx, self.img_list, self.img_vessel_list = self.registration_series(img_exist_list, img_None_list)
            except:
                print('no registration')
            self.base_img = self.img_list[self.base_idx]
            self.base_kpt = self.surf_info['kpt_list'][self.base_idx]
            self.base_vessel = self.img_vessel_list[self.base_idx]

            self.img_list.pop(self.base_idx)
            self.img_vessel_list.pop(self.base_idx)

            self.base_mask = mask

        else:

            self.center_pt = center_pt
            self.base_img = self.img_list[self.base_idx]
            self.base_vessel = self.img_vessel_list[self.base_idx]

            self.img_list.pop(self.base_idx)
            self.regist_img_list = self.img_list

            self.img_vessel_list.pop(self.base_idx)
            self.vessel_img_list = self.img_vessel_list

            self.base_mask = mask

        self.surf_CEM_list = []
        self.surf_mask = []
        self.surf_CEM_value = 0

        self.bspline_CEM_list = []
        self.bspline_mask = []
        self.bspline_CEM_value = 0

        self.regist_count = 0


    def registration_series(self, img_exist_list, img_None_list):
        """
        registration series set
        :param img_exist_list: fundus photo image list, have optic disc or fovea
        :param img_None_list: fundus photo image list, didn't have optic disc or fovea
        :return: sorted image list with comapring feature point
        """

        all_image = [img_exist_list[0]+img_None_list[0],img_exist_list[1]+img_None_list[1]]

        self.matching_list = [[0 for i in range(all_image[0].__len__())] for j in range(all_image[0].__len__())]
        self.matching_list_num = [[0 for i in range(all_image[0].__len__())] for j in range(all_image[0].__len__())]

        for idx, img_idx in enumerate(all_image[0]):
             if self.using_VPmap == True:
                kps, descs, sift_kpt_img, _ = self.get_sift_kpt(img_idx, self.mask, is_VesselMask=self.img_vessel_list[idx],
                                                                hess_thresh=self.hess_thresh)
             else:
                kps, descs, sift_kpt_img, _ = self.get_sift_kpt(img_idx, self.mask, is_VesselMask = None, hess_thresh=self.hess_thresh)
             self.surf_info['kpt_list'].append(kps)
             self.surf_info['des_list'].append(descs)

        for idx in range(all_image[0].__len__()):
            for idx_2 in range(all_image[0].__len__()):
                if idx == idx_2:
                    continue
                good = self.matching(self.surf_info['kpt_list'][idx], self.surf_info['des_list'][idx],
                    self.surf_info['kpt_list'][idx_2], self.surf_info['des_list'][idx_2], self.mask, self.mask, distance_m =  self.matching_distance)
                try:
                    h, good_homography = self.find_homography(self.surf_info['kpt_list'][idx], self.surf_info['kpt_list'][idx_2], good)
                except:
                    h, good_homography = 0,[0]

                self.matching_list[idx][idx_2] = good_homography
                self.matching_list_num[idx][idx_2] = good_homography.__len__()

        self.matching_list_num = np.array(self.matching_list_num)

        base_idx = np.sum(self.matching_list_num, axis=1).argmax()
        self.good_list = self.matching_list[:][self.base_idx]
        return base_idx, all_image[0], all_image[1]


    def get_sift_kpt(self, img, org_mask, is_VesselMask=False, hess_thresh = 30):
        """
        calculate keypoint
        :param img: funuds image
        :param org_mask: mask of fundus image 
        :param is_VesselMask: if none, using kpt in org_mask, else using kpt in vessel mask
        :param hess_thresh: hessain threshold of surf algorithm
        :return: keypoint and decriptor list
        """
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray_img = img.reshape(img.shape[:-1])
        else:
            gray_img = img.copy()

        kernel = np.ones((11, 11), np.ubyte)
        mask = skimage.morphology.erosion(org_mask, kernel).astype(np.ubyte)
        mask = skimage.morphology.erosion(mask, kernel).astype(np.ubyte)
        mask = skimage.morphology.erosion(mask, kernel).astype(np.ubyte)
        FOV_mask = mask.copy()

        gray_img_norm = gray_img.astype(np.float32)

        if type(is_VesselMask).__name__ == 'ndarray':
            bg_mask = cv2.bitwise_and(org_mask, 255 - is_VesselMask)
            bg_mask = np.clip(bg_mask, 0, 50)
            gray_img_norm += bg_mask

        gray_img_norm = ((gray_img_norm - gray_img_norm[org_mask != 0].min()) / float(gray_img_norm.max() - gray_img_norm[org_mask != 0].min()) + 0.1) * 255.
        gray_img_norm = np.clip(gray_img_norm, 0, 255)
        gray_img_norm = gray_img_norm.astype(np.ubyte)

        surf = cv2.xfeatures2d.SURF_create(hess_thresh, nOctaves=4, nOctaveLayers=3,  upright = 1)

        kps, descs = surf.detectAndCompute(gray_img_norm, mask=org_mask)

        excluded_kps = []
        excluded_descs = []

        for kps_idx in range(kps.__len__()):
            x, y = kps[kps_idx].pt
            if FOV_mask[int(y), int(x)] == 255:
                excluded_kps.append(kps[kps_idx])
                excluded_descs.append(descs[kps_idx])

        return excluded_kps, np.array(excluded_descs)



    def matching(self, kps1, descs1, kps2, descs2, max_pt_dist=50000, distance_m = 0.8):
        """
        keypoint matching brute force matching
        :param kps1: kepoint list in image1
        :param descs1: descriptor list in image1
        :param kps2: kepoint list in image2
        :param descs2: descriptor list in image2
        :param max_pt_dist: limited point distance with source image
        :param distance_m: limited point distance with keypoint
        :return: mathced good descriptor list
        """
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descs1, descs2, k=2)
        good = []
        for m, n in matches:
            pt_dist = np.sqrt(
                (kps1[m.queryIdx].pt[0] - kps2[m.trainIdx].pt[0]) * (kps1[m.queryIdx].pt[0] - kps2[m.trainIdx].pt[0]) + \
                (kps1[m.queryIdx].pt[1] - kps2[m.trainIdx].pt[1]) * (kps1[m.queryIdx].pt[1] - kps2[m.trainIdx].pt[1]))
            if m.distance < distance_m * n.distance and pt_dist < max_pt_dist:
                good.append([m])


        return good

    def CEM(self, centerline_1, centerline_2, overlap_region):
        """
        calculate center line error
        :param centerline_1: center vessel line of image1
        :param centerline_2: center vessel line of image2
        :param overlap_region: overlap mask between image1 and image2
        :return: center line error value
        """
        centerline_1 = cv2.bitwise_and(centerline_1, centerline_1, mask = overlap_region)
        centerline_2 = cv2.bitwise_and(centerline_2, centerline_2, mask = overlap_region)
        centerline_1_pt = cv2.findNonZero(centerline_1[:,:,2])
        centerline_2_pt = cv2.findNonZero(centerline_2[:,:,0])
        result_cem = 0
        for input_venus_nonzero_idx in centerline_1_pt:
            tmp_edge_nonzero = centerline_2_pt - input_venus_nonzero_idx
            min_distance = np.sqrt(np.sum(cv2.pow(tmp_edge_nonzero, 2), axis=2)).min()
            result_cem += min_distance
        return result_cem / float(centerline_1_pt.__len__())


    def make_CEM_image(self, vessel_img1, vessel_img2, mask1, mask2):
        """
        make center line image and overlap mask
        :param vessel_image1:
        :param vessel_image2:
        :param mask1: FOV of image1
        :param mask2: FOV of image2
        :return: centerline image of vessel_image1, centerline image of vessel_image2,
        """
        overlap_region = cv2.bitwise_and(mask1, mask2)
        otsu_vessel_img1 = cv2.threshold(vessel_img1, 30, 255, cv2.THRESH_BINARY)[1]
        otsu_vessel_img1 = cv2.bitwise_and(otsu_vessel_img1, otsu_vessel_img1, mask = mask1)
        centerline_img1_org = skeletonize(otsu_vessel_img1 / 255).astype(np.uint8) * 255

        otsu_vessel_img2 = cv2.threshold(vessel_img2, 30, 255, cv2.THRESH_BINARY)[1]
        otsu_vessel_img2 = cv2.bitwise_and(otsu_vessel_img2, otsu_vessel_img2, mask = mask2)
        centerline_img2_org = skeletonize(otsu_vessel_img2 / 255).astype(np.uint8) * 255

        centerline_img1 = cv2.cvtColor(centerline_img1_org, cv2.COLOR_GRAY2BGR)
        centerline_img2 = cv2.cvtColor(centerline_img2_org, cv2.COLOR_GRAY2BGR)

        overlap_region1 = cv2.bitwise_and(centerline_img1, centerline_img1, mask = mask2)
        centerline_img1[:,:,:2] = centerline_img1[:,:,:2] - overlap_region1[:,:,:2]

        overlap_region2 = cv2.bitwise_and(centerline_img2, centerline_img2, mask = mask1)
        centerline_img2[:, :, 1:] = centerline_img2[:, :, 1:] - overlap_region2[:,:,1:]

        overlap_pixel =  (overlap_region > 0).sum()
        all_pixel = (cv2.bitwise_or(mask1, mask2) > 0).sum()
        overlap_IOU = overlap_pixel / float(all_pixel)

        return centerline_img1, centerline_img2, overlap_region, overlap_IOU


    def do_registration(self, img1_info, img2_info, h):
        """
        rigid registration
        :param img1_info: image1 and mask1
        :param img2_info: image2 and mask2
        :param h: homography 3x3 matrix
        :return: registratoin result(if False no registration), [warped image1, warped image2], [warped image1, warped image2]
        """
        img1, mask1 = img1_info
        img2, mask2 = img2_info

        if type(h).__name__ == 'NoneType':
            return False, [[],[]], [[],[]]


        self.h1, self.w1 = img1.shape[:2]
        self.h2, self.w2 = img2.shape[:2]
        pts1 = np.reshape([[0, 0], [0, self.h1], [self.w1, self.h1], [self.w1, 0]], [-1, 1, 2]).astype(np.float)
        pts2 = np.reshape([[0, 0], [0, self.h2], [self.w2, self.h2], [self.w2, 0]], [-1, 1, 2]).astype(np.float)
        res_pts = np.concatenate((cv2.perspectiveTransform(pts2, h), pts1), axis=0)
        res_min_pt = (res_pts.min(axis=0)[0] - 0.5).astype(np.int)
        res_max_pt = (res_pts.max(axis=0)[0] + 0.5).astype(np.int)



        t = -res_min_pt
        ht = np.asarray([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

        self.h = h
        self.ht = ht
        self.t = t
        self.res_max_pt = res_max_pt
        self.res_min_pt = res_min_pt

        if self.res_max_pt[0] - self.res_min_pt[0] > 20000 or  self.res_max_pt[1] - self.res_min_pt[1] > 20000:
            print('image size is big')
            return False, [[], []], [[], []]

        mask1_expended, mask2_expended = self.registrationFromMatrix(mask1, mask2)

        img1_expended, im_out_canvas = self.registrationFromMatrix(img1, img2)
        img2_expended = im_out_canvas.copy()

        return True, [img1_expended, mask1_expended], [img2_expended, mask2_expended]



    def registrationFromMatrix(self, source, target):
        """
        warping image target to source
        :param source: source image
        :param target: target image
        :return: warped source image, warped target image
        """
        im_canvas_target = cv2.warpPerspective(target, np.dot(self.h, self.ht), (
            self.res_max_pt[0] - self.res_min_pt[0], self.res_max_pt[1] - self.res_min_pt[1]))

        im_canvas_source = np.zeros(im_canvas_target.shape).astype(np.uint8)
        im_canvas_source[self.t[1]:self.h1 + self.t[1], self.t[0]:self.w1 + self.t[0]] = source

        return im_canvas_source, im_canvas_target

    def TRE_branch(self, source_branch, target_branch):
        """
        Target registration error of vessel branch
        :param source_branch: source vessel image branch
        :param target_branch: target vessel image branch
        :return: target registration value
        """
        TRE = 0
        source_branch_pts = cv2.findNonZero(source_branch).reshape(-1,2)
        target_branch_pts = cv2.findNonZero(target_branch).reshape(-1,2)
        for target_branch_idx in target_branch_pts:
            min_error = 1000000
            for source_branch_idx in source_branch_pts:
                branch_error = np.sqrt(np.power(np.array(target_branch_idx) - np.array(source_branch_idx), 2).sum())
                if branch_error < min_error:
                    min_error = branch_error
            TRE += min_error
        return min_error

    def overlap_rate(self, source, target):
        """
        calculate overlap rate
        :param source: source mask
        :param target: target mask
        :return: overlap rate value
        """
        overlap_region = cv2.bitwise_and(source, target)
        all_region = cv2.bitwise_or(source, target)
        overlap_region_all = cv2.findNonZero(overlap_region).shape[0]
        all_region_all = cv2.findNonZero(all_region).shape[0]
        return overlap_region_all/float(all_region_all)

    def registration_pair(self):


        img_list = self.img_list + self.img_none_list
        self.img_vessel_list += self.img_vessel_none_list

        for idx, img_idx_org in enumerate(img_list):
            base_img = self.base_img.copy()
            base_mask = self.mask

            exist_mask = self.mask
            base_vessel = self.base_vessel.copy()
            target_vessel = self.img_vessel_list[idx].copy()
            img_idx = img_idx_org.copy()

            #make image size same
            if base_img.shape != img_idx.shape:
                tmp_img = np.zeros(base_img.shape).astype(np.uint8)
                tmp_img[:img_idx.shape[0], :img_idx.shape[1]] = img_idx
                img_idx = tmp_img.copy()

                tmp_img = np.zeros(base_img.shape[:2]).astype(np.uint8)
                tmp_img[:exist_mask.shape[0], :exist_mask.shape[1]] = exist_mask
                exist_mask = tmp_img.copy()

                tmp_img = np.zeros(base_img.shape[:2]).astype(np.uint8)
                tmp_img[:target_vessel['vessel'].shape[0], :target_vessel['vessel'].shape[1]] = \
                target_vessel[idx]['vessel']
                target_vessel[idx]['vessel'] = tmp_img.copy()

                tmp_img = np.zeros(base_img.shape[:2]).astype(np.uint8)
                tmp_img[:target_vessel[idx]['branch'].shape[0], :target_vessel[idx]['branch'].shape[1]] = \
                    target_vessel[idx]['branch']
                target_vessel[idx]['branch'] = tmp_img.copy()

            #get keypoint from image
            if self.using_VPmap == True:
                img_kps, img_descs, img_sift_kpt_img, _ = self.get_sift_kpt(img_idx, exist_mask,
                                                                            is_VesselMask=self.img_vessel_list[idx][
                                                                                'vessel'],
                                                                            hess_thresh=self.hess_thresh)
                base_kpt, base_descs, base_sift_kpt_img, kpt_img = self.get_sift_kpt(base_img, base_mask,
                                                                                     is_VesselMask=self.base_vessel[
                                                                                         'vessel'],
                                                                                     hess_thresh=self.hess_thresh)
            else:
                img_kps, img_descs, img_sift_kpt_img, _ = self.get_sift_kpt(img_idx, exist_mask, is_VesselMask=None,
                                                                            hess_thresh=self.hess_thresh)
                base_kpt, base_descs, base_sift_kpt_img, kpt_img = self.get_sift_kpt(base_img, base_mask,
                                                                                     is_VesselMask=None,
                                                                                     hess_thresh=self.hess_thresh)

            good = self.matching(base_kpt, base_descs,
                                 img_kps, img_descs,
                                 base_mask, self.mask, distance_m=self.matching_distance)

            h, good = self.find_homography(base_kpt, img_kps, good)

            result_flag, [result_base_img, result_base_mask], [result_img_idx, result_img_mask] = \
                self.do_registration([base_img, base_mask], [img_idx, self.mask], h)

            if result_flag == False:
                continue
            registration_result = self.registration_result(result_img_mask, self.mask)

            if 0.9 < registration_result < 1.1:

                mask_nonzero = np.nonzero(result_base_mask + result_img_mask)
                base_img = result_base_img
                base_mask = result_base_mask
                img_idx = result_img_idx
                img_mask = result_img_mask
            else:
                print('exist_disc regist pass : {}'.format(registration_result))
                continue

            self.surf_mask.append(
                [np.clip((base_mask.astype(np.int) + img_mask.astype(np.int)), 0, 255).astype(np.uint8),
                 '{}_{}_mask.png'.format(base_vessel['idx'],
                                         target_vessel['idx'])])
            base_vessel['vessel'], target_vessel['vessel'] = self.registrationFromMatrix(base_vessel['vessel'],
                                                                                         target_vessel['vessel'])
            base_vessel['branch'], _ = self.registrationFromMatrix(
                base_vessel['branch'], target_vessel['branch'])

            rigid_matrix = np.dot(self.h, self.ht)
            # target_branch = np.insert(self.vessel_img_list[idx]['branch'], 2, 1, axis=2).reshape(-1, 3, 1)
            target_branch = np.insert(cv2.findNonZero(target_vessel['branch'].astype(np.uint8)), 2, 1,
                                      axis=2).reshape(-1, 3, 1)
            branch_mapping = np.zeros(
                [base_vessel['vessel'].shape[0], base_vessel['vessel'].shape[1], 3])
            branch_mapping[:, :, 0] = base_vessel['branch']
            for target_branch_idx in target_branch:
                try:
                    branch_pt = np.dot(rigid_matrix, target_branch_idx)
                    branch_pt = branch_pt[:2] / branch_pt[2]
                    branch_mapping[int(branch_pt[1]), int(branch_pt[0]), 2] = 255
                except Exception as e:
                    print(e)

            # for surf_idx, surf_result_img in enumerate([base_img, img_idx, self.base_vessel, base_mask, img_mask, self.vessel_img_list[idx]]):
            #     Image.fromarray(surf_result_img.astype(np.uint8)).save("./montage_7etdrs/1_2_SIFT_result/{}_{}.jpg".format('surf_result', surf_idx))

            [img_idx, img_vessel, img_mask, bsp] = self.do_bspline(img_idx, base_vessel['vessel'],
                                                                   target_vessel['vessel'], base_mask,
                                                                   img_mask)
            target_vessel_skleton = pcv.morphology.skeletonize(mask=img_vessel)
            source_vessel_skleton = pcv.morphology.skeletonize(mask=img_vessel)
            # target_branch_bspline = bsp.registrationFromMatrix(branch_mapping[:,:,2]).astype(np.uint)
            overlap_region = cv2.bitwise_and(base_mask, img_mask)
            branch_mapping = cv2.bitwise_and(branch_mapping, branch_mapping, mask=overlap_region).astype(np.uint8)

            overlap_region_rate = self.overlap_rate(base_mask, img_mask)

            surf_branch_mapping = branch_mapping.copy()
            surf_TRE = self.TRE_branch(branch_mapping[:, :, 0], branch_mapping[:, :, 2])

            branch_mapping = branch_mapping.astype(np.uint8)
            target_branch_bspline = bsp.get_displacement_vector_pt(branch_mapping[:, :, 2]).astype(np.uint)
            bspline_branch_mapping = branch_mapping.copy()
            bspline_branch_mapping[:, :, 2] = 0
            bspline_branch_mapping[:, :, 2] = target_branch_bspline[:, :, 2]
            bspline_branch_mapping = bspline_branch_mapping.astype(np.uint8)

            bspline_TRE = self.TRE_branch(bspline_branch_mapping[:, :, 0],
                                                                   bspline_branch_mapping[:, :, 2])

            source_pt = cv2.findNonZero(bspline_branch_mapping[:, :, 0]).reshape(-1,2)
            target_pt = cv2.findNonZero(bspline_branch_mapping[:, :, 2]).reshape(-1, 2)

            surf_center1, surf_center2, surf_overlap_region, surf_overlap_IOU = self.make_CEM_image(
                base_vessel['vessel'],
                target_vessel['vessel'],
                base_mask, result_base_mask)


            self.surf_CEM_value = self.CEM(surf_center1, surf_center2, surf_overlap_region)

            base_img = pyramid_spline(base_img, img_idx, base_mask, img_mask, 6)

            center1, center2, overlap_region, overlap_IOU = self.make_CEM_image(base_vessel['vessel'],
                                                                                                 img_vessel, base_mask,
                                                                                                 img_mask)
            self.bspline_mask.append([np.clip((base_mask.astype(np.int) + img_mask.astype(np.int)), 0, 255).astype(np.uint8), '{}_{}_mask.png'.format(base_vessel['idx'],
                                                                                      target_vessel['idx'])])
            self.bspline_CEM_value = self.CEM(center1, center2, overlap_region)

            branch_image = cv2.cvtColor(base_vessel['vessel'], cv2.COLOR_GRAY2BGR)
            branch_image[:, :, 2] = base_vessel['vessel'] - cv2.bitwise_and(base_vessel['vessel'],
                                                                                 base_vessel['vessel'],
                                                                                 mask=bspline_branch_mapping[:, :, 0])
            branch_image[:, :, 0] = base_vessel['vessel'] - cv2.bitwise_and(base_vessel['vessel'],
                                                                                 base_vessel['vessel'],
                                                                                 mask=bspline_branch_mapping[:, :, 2])
            branch_image[:, :, 1] = base_vessel['vessel'] - cv2.bitwise_and(base_vessel['vessel'],
                                                                                 base_vessel['vessel'],
                                                                                 mask=cv2.cvtColor(
                                                                                     bspline_branch_mapping,
                                                                                     cv2.COLOR_BGR2GRAY))
            self.branching_pt_list.append([branch_image, '{}_{}_branching.png'.format(base_vessel['idx'],
                                                                                      target_vessel[
                                                                                          'idx'])])

            self.branching_error_list.append([bspline_tar_error_image, '{}_{}_target_error.png'.format(base_vessel['idx'],
                                                                                               target_vessel['idx'])])

            cv2.imwrite("./{}_mask.jpg".format(self.regist_count), base_img)
            cv2.imwrite("./{}_montage.jpg".format(self.regist_count), img_mask)
            self.regist_count += 1
        self.base_img = base_img


    def registration_non_detection(self):

        #set all img list(0 : fp, 1: vessel, pts, branch)
        img_list = self.img_list
        img_vessel_list = self.img_vessel_list

        base_img = self.base_img
        base_mask = self.base_mask.copy()

        while (img_list.__len__()):
            base_kps, base_descs, base_sift_kpt_img, base_kpt_img = self.get_sift_kpt(base_img, base_mask,
                                                                                      is_VesselMask=self.base_vessel['vessel'],
                                                                                      hess_thresh=self.hess_thresh)
            # if self.using_VPmap == True:
            #     base_kps, base_descs, base_sift_kpt_img, kpt_img = self.get_sift_kpt(base_img, base_mask, is_VesselMask = self.base_vessel, hess_thresh = self.hess_thresh)
            # else:
            #     base_kps, base_descs, base_sift_kpt_img, kpt_img = self.get_sift_kpt(base_img, base_mask, is_VesselMask=None, hess_thresh = self.hess_thresh)

            matching_good = 0
            fp_img = 0
            fp_kpt, img_vessel = 0, 0

            for no_fp_idx in range(img_list.__len__()):
                no_fp_img = np.zeros(base_img.shape).astype(np.uint8)
                # no_fp_vessel = np.zeros(base_img.shape[:2]).astype(np.uint8)
                # no_fp_vessel[:no_fp_list[no_fp_idx].shape[0], :no_fp_list[no_fp_idx].shape[1]] = self.img_vessel_none_list[no_fp_idx]
                fp_mask = np.zeros(base_img.shape[:2]).astype(np.uint8)
                fp_mask[:img_list[no_fp_idx].shape[0], :img_list[no_fp_idx].shape[1]] = self.mask
                no_fp_img[:img_list[no_fp_idx].shape[0], :img_list[no_fp_idx].shape[1]] = img_list[no_fp_idx]

                no_fp_vessel = np.zeros(base_img.shape[:2]).astype(np.uint8)
                no_fp_vessel[:img_list[no_fp_idx].shape[0], :img_list[no_fp_idx].shape[1]] = self.img_vessel_list[no_fp_idx]['vessel']

                np_fp_kpt, no_fp_des, no_fp_kpt_img, kpt_img = self.get_sift_kpt(no_fp_img, fp_mask,
                                                                                 is_VesselMask = no_fp_vessel,
                                                                                 hess_thresh=self.hess_thresh)

                good = self.matching(base_kps, base_descs, np_fp_kpt, no_fp_des, self.base_mask,
                                     fp_mask, distance_m=self.matching_distance)
                try:
                    h, good = self.find_homography(base_kps, np_fp_kpt, good)
                except:
                    continue
                if matching_good == 0:
                    fp_img = no_fp_img.copy()
                    fp_kpt = np_fp_kpt
                    img_vessel = self.img_vessel_list[no_fp_idx]
                    fp_vessel = no_fp_vessel
                    matching_good = [no_fp_idx, good, h]
                else:
                    if matching_good[1].__len__() < good.__len__():
                        fp_img = no_fp_img.copy()
                        fp_kpt = np_fp_kpt
                        img_vessel = self.img_vessel_list[no_fp_idx]
                        matching_good = [no_fp_idx, good, h]
                # matching_img = cv2.drawMatchesKnn(base_img, base_kps, no_fp_img, np_fp_kpt, good, flags=2, outImg=None)
                # matching_img2 = cv2.drawMatchesKnn(self.base_vessel, base_kps2, no_fp_vessel, np_fp_kpt2, good2, flags=2, outImg=None)


                # Image.fromarray(matching_img2).save("./montage_7etdrs/{}_vessel.jpg".format(no_fp_idx))

            # fp_vessel = np.zeros(base_img.shape[:2])
            # fp_vessel[:img_vessel['vessel'].shape[0], :img_vessel['vessel'].shape[1]] = img_vessel['vessel']

            matching_img = cv2.drawMatchesKnn(base_img, base_kps,
                                              fp_img, fp_kpt, matching_good[1], flags=2,
                                              outImg=None)

            result_flag, [result_base_img, result_base_mask], [result_registed_img,
                                                               result_registed_mask] = self.do_registration(
                [base_img, base_mask],
                [fp_img, fp_mask], matching_good[2])
            Image.fromarray(matching_img).save("./montage_7etdrs/{}.jpg".format(matching_good[0]))






            if result_flag == False:
                img_list.pop(matching_good[0])
                img_vessel_list.pop(matching_good[0])
                if img_list.__len__() == 0:
                    break
                continue
            registration_result = self.registration_result(result_registed_mask, fp_mask)

            if 0.8 < registration_result < 1.2:
                mask_nonzero = np.nonzero(result_base_mask + result_registed_mask)
                # min_x, min_y, max_x, max_y = max(mask_nonzero[1].min() - 50, 0), max(mask_nonzero[0].min() - 50,
                #                                                                      0), min(
                #     mask_nonzero[1].max() + 50, mask_nonzero[1].max()), min(mask_nonzero[0].max() + 50,
                #                                                             mask_nonzero[0].max())
                # result_base_mask = result_base_mask[min_y:max_y, min_x:max_x]
                # result_registed_mask = result_registed_mask[min_y:max_y, min_x:max_x]
                #
                # result_base_img = result_base_img[min_y:max_y, min_x:max_x]
                # result_registed_img = result_registed_img[min_y:max_y, min_x:max_x]
                base_img = result_base_img
                base_mask = result_base_mask
                registed_img = result_registed_img
                registed_mask = result_registed_mask
            else:

                img_list.pop(matching_good[0])
                img_vessel_list.pop(matching_good[0])
                print('no disc : regist pass {}'.format(registration_result))
                if img_list.__len__() == 0:
                    break
                continue


            self.base_vessel['vessel'], registed_img_vessel = self.registrationFromMatrix(self.base_vessel['vessel'],
                                                                                          fp_vessel)

            surf_center1, surf_center2, surf_overlap_region, surf_overlap_IOU = self.make_CEM_image(
                self.base_vessel['vessel'], registed_img_vessel,
                base_mask, result_registed_mask)

            self.surf_CEM_value += self.CEM(surf_center1, surf_center2, surf_overlap_region)




            bspline_overlap_region = bspline_class(registed_img, self.base_vessel['vessel'],
                                                                                     registed_img_vessel,
                                                                                     base_mask, registed_mask)
            bspline_img, img_vessel['vessel'], bspline_mask = bspline_overlap_region.regi_disc_FP, bspline_overlap_region.regi_FPVessel, bspline_overlap_region.regi_target_mask
            # [img_idx, img_vessel, img_mask, bsp] = self.do_bspline(img_idx, self.base_vessel['vessel'], self.vessel_img_list[idx]['vessel'], base_mask, img_mask)
            self.bspline_displaent.append([bspline_overlap_region.get_vector_field(),
                                              '{}_{}_displacment.png'.format(self.base_vessel['idx'],
                                                                             img_vessel['idx'])])

            base_img = pyramid_spline(base_img, bspline_img, base_mask, bspline_mask, 6)

            self.registered_image_count +=1
            center1, center2, overlap_region, bspline_overlap_IOU = self.make_CEM_image(self.base_vessel[
                                                                                                             'vessel'],
                                                                                                         img_vessel[
                                                                                                             'vessel'],
                                                                                                         base_mask,
                                                                                                         bspline_mask)
            self.bspline_CEM_value += self.CEM(center1, center2, overlap_region)
            base_mask = np.clip((base_mask.astype(np.int) + bspline_mask.astype(np.int)), 0, 255).astype(np.uint8)
            self.base_vessel['vessel'] = np.clip(
                (self.base_vessel['vessel'].astype(np.int) + img_vessel['vessel'].astype(np.int)), 0, 255).astype(
                np.uint8)


            mask_nonzero = np.nonzero(base_mask)
            max_x, max_y = max(mask_nonzero[1].max() + 100, self.mask.shape[1]), max(mask_nonzero[0].max() + 100, self.mask.shape[0])
            temp_image = np.zeros([max_y, max_x])
            temp_image[0:base_mask.shape[0], 0:base_mask.shape[1]] = base_mask
            base_mask = temp_image

            temp_image = np.zeros([max_y, max_x])
            temp_image[0:self.base_vessel['vessel'].shape[0], 0:self.base_vessel['vessel'].shape[1]] = self.base_vessel['vessel']
            self.base_vessel['vessel'] = temp_image

            temp_image = np.zeros([max_y, max_x,3])
            temp_image[0:base_img.shape[0], 0:base_img.shape[1]] = base_img
            base_img = temp_image


            # cv2.imwrite("./{}_mask.jpg".format(self.regist_count), base_img)
            # cv2.imwrite("./{}_montage.jpg".format(self.regist_count), bspline_mask)
            self.regist_count += 1

            img_list.pop(matching_good[0])
            img_vessel_list.pop(matching_good[0])
            if img_list.__len__() == 0:
                break
        return base_img, self.base_vessel

    def find_homography(self, src_kpt, des_kpt, good):
        """
        :param src_kpt: source keypoint
        :param des_kpt: target keypoint
        :param good: matched good descriptor list
        :return: homography matrix, good list from ransan result
        """
        pts_src = []  # img2
        pts_dst = []  # img1

        for i, match in enumerate(good):
            pts_src.append([src_kpt[match[0].queryIdx].pt[0], src_kpt[match[0].queryIdx].pt[1]])
            pts_dst.append([des_kpt[match[0].trainIdx].pt[0], des_kpt[match[0].trainIdx].pt[1]])

        pts_src = np.array(pts_src)
        pts_dst = np.array(pts_dst)

        h, status = cv2.findHomography(pts_dst, pts_src, cv2.RANSAC, ransacReprojThreshold=30)
        good_re = []
        for idx, good_idx in enumerate(status):
            if good_idx[0] == True:
                good_re.append(good[idx])
        return h, good_re

    def registration_exist_fp_repeat(self):
        base_img = self.base_img
        base_mask = self.mask

        img_list = self.regist_img_list

        for idx, img_idx in enumerate(img_list):

            if self.center_pt != False:
                exist_mask = self.mask
            else:
                try:
                    exist_mask = self.mask.copy()
                    cv2.circle(exist_mask, tuple(self.center_pt[idx]), 50, (0, 0, 0), -1)
                except:
                    print('exist_mask')
                    exist_mask = self.mask

            if base_img.shape != img_idx.shape:
                tmp_img = np.zeros(base_img.shape).astype(np.uint8)
                tmp_img[:img_idx.shape[0], :img_idx.shape[1]] = img_idx
                img_idx = tmp_img.copy()

                tmp_img = np.zeros(base_img.shape[:2]).astype(np.uint8)
                tmp_img[:exist_mask.shape[0], :exist_mask.shape[1]] = exist_mask
                exist_mask = tmp_img.copy()

                tmp_img = np.zeros(base_img.shape[:2]).astype(np.uint8)
                tmp_img[:self.img_vessel_list[idx]['vessel'].shape[0], :self.img_vessel_list[idx]['vessel'].shape[1]] = self.img_vessel_list[idx]['vessel']
                self.img_vessel_list[idx]['vessel'] = tmp_img.copy()

                tmp_img = np.zeros(base_img.shape[:2]).astype(np.uint8)
                tmp_img[:self.img_vessel_list[idx]['branch'].shape[0], :self.img_vessel_list[idx]['branch'].shape[1]] = \
                self.img_vessel_list[idx]['branch']
                self.img_vessel_list[idx]['branch'] = tmp_img.copy()


            if self.using_VPmap == True:
                img_kps, img_descs, img_sift_kpt_img, _ = self.get_sift_kpt(img_idx, exist_mask,
                                                                is_VesselMask=self.img_vessel_list[idx]['vessel'],
                                                                hess_thresh=self.hess_thresh)
                base_kpt, base_descs, base_sift_kpt_img, kpt_img = self.get_sift_kpt(base_img, base_mask,
                                                                                          is_VesselMask=self.base_vessel['vessel'],
                                                                                          hess_thresh=self.hess_thresh)
            else:
                img_kps, img_descs, img_sift_kpt_img, _ = self.get_sift_kpt(img_idx, exist_mask, is_VesselMask=None,
                                                                hess_thresh=self.hess_thresh)
                base_kpt, base_descs, base_sift_kpt_img, kpt_img = self.get_sift_kpt(base_img, base_mask,
                                                                                          is_VesselMask=None,
                                                                                          hess_thresh=self.hess_thresh)

            # Image.fromarray(img_sift_kpt_img).save("./montage_7etdrs/1_2_SIFT_result/{}.jpg".format(idx))






            good = self.matching(base_kpt, base_descs,
                                 img_kps, img_descs,
                                 base_mask, self.mask, distance_m=self.matching_distance)



            h, good_homography = self.find_homography(base_kpt, img_kps, good)


            result_flag, [result_base_img, result_base_mask], [result_img_idx, result_img_mask] = \
                self.do_registration([base_img, base_mask], [img_idx, self.mask], h)

            if result_flag == False:
                continue
            registration_result = self.registration_result(result_img_mask, self.mask)



            if 0.9 < registration_result < 1.1:

                base_img = result_base_img
                base_mask = result_base_mask
                img_idx = result_img_idx
                img_mask = result_img_mask
            else:
                print('exist_disc regist pass : {}'.format(registration_result))
                continue


            self.base_vessel['vessel'], self.vessel_img_list[idx]['vessel'] = self.registrationFromMatrix(self.base_vessel['vessel'], self.vessel_img_list[idx]['vessel'])
            self.base_vessel['branch'], _ = self.registrationFromMatrix(self.base_vessel['branch'], self.vessel_img_list[idx]['branch'])

            rigid_matrix = np.dot(self.h, self.ht)
            # target_branch = np.insert(self.vessel_img_list[idx]['branch'], 2, 1, axis=2).reshape(-1, 3, 1)
            target_branch = np.insert(cv2.findNonZero(self.vessel_img_list[idx]['branch'].astype(np.uint8)), 2, 1, axis=2).reshape(-1, 3, 1)
            branch_mapping = np.zeros(
                [self.base_vessel['vessel'].shape[0], self.base_vessel['vessel'].shape[1], 3])
            branch_mapping[:, :, 0] = self.base_vessel['branch']
            try:
                for target_branch_idx in target_branch:
                    branch_pt = np.dot(rigid_matrix, target_branch_idx)
                    branch_pt = branch_pt[:2] / branch_pt[2]
                    branch_mapping[int(branch_pt[1]), int(branch_pt[0]), 2] = 255
            except:
                print('error')



            # for surf_idx, surf_result_img in enumerate([base_img, img_idx, self.base_vessel, base_mask, img_mask, self.vessel_img_list[idx]]):
            #     Image.fromarray(surf_result_img.astype(np.uint8)).save("./montage_7etdrs/1_2_SIFT_result/{}_{}.jpg".format('surf_result', surf_idx))

            before = datetime.now()
            [img_idx, img_vessel, img_mask, bsp] = self.do_bspline(img_idx, self.base_vessel['vessel'], self.vessel_img_list[idx]['vessel'], base_mask, img_mask)

            # bsp = bspline_class(img_idx, self.base_vessel['vessel'], self.vessel_img_list[idx]['vessel'], base_mask, img_mask)
            # img_idx, img_vessel, img_mask = bsp.regi_disc_FP, bsp.regi_FPVessel, bsp.regi_target_mask
            after = datetime.now()
            print((after - before).total_seconds())
            # bsp.get_displacement_vector_field()[1]


            # target_branch_bspline = bsp.registrationFromMatrix(branch_mapping[:,:,2]).astype(np.uint)
            overlap_region = cv2.bitwise_and(base_mask, img_mask)
            branch_mapping = cv2.bitwise_and(branch_mapping, branch_mapping, mask = overlap_region)

            overlap_region_rate = self.overlap_rate(base_mask, img_mask)

            surf_branch_mapping = branch_mapping.copy()

            branch_mapping = branch_mapping.astype(np.uint8)
            target_branch_bspline = bsp.get_displacement_vector_pt(branch_mapping[:, :, 2]).astype(np.uint)
            bspline_branch_mapping = branch_mapping.copy()
            bspline_branch_mapping[:,:,2] = 0
            bspline_branch_mapping[:,:,2] = target_branch_bspline[:,:,2]
            bspline_branch_mapping = bspline_branch_mapping.astype(np.uint8)


            surf_center1, surf_center2, surf_overlap_region, surf_overlap_IOU = self.make_CEM_image(
                self.base_vessel['vessel'],
                self.vessel_img_list[idx]['vessel'],
                base_mask, img_mask)

            self.surf_CEM_value += self.CEM(surf_center1, surf_center2, surf_overlap_region)
            # bsp.get_displacement_vector_field()


            base_img = pyramid_spline(base_img, img_idx,  base_mask, img_mask, 6)
            self.registered_image_count += 1


            center1, center2, overlap_region, overlap_IOU = self.make_CEM_image(self.base_vessel['vessel'], img_vessel, base_mask, img_mask)
            self.bspline_CEM_value += self.CEM(center1, center2, overlap_region)
            base_mask = np.clip((base_mask.astype(np.int) + img_mask.astype(np.int)),0,255).astype(np.uint8)

            result_vessel = cv2.threshold(np.clip((self.base_vessel['vessel'].astype(np.int) + img_vessel.astype(np.int)),0,255).astype(np.uint8),125,255, cv2.THRESH_BINARY)[1]
            self.base_vessel['vessel'] = cv2.bitwise_and(result_vessel, result_vessel, mask = base_mask)

            branch_image = cv2.cvtColor(self.base_vessel['vessel'], cv2.COLOR_GRAY2BGR)
            branch_image[:, :, 2] = self.base_vessel['vessel'] - cv2.bitwise_and(self.base_vessel['vessel'], self.base_vessel['vessel'], mask = bspline_branch_mapping[:,:,0])
            branch_image[:, :, 0] = self.base_vessel['vessel'] - cv2.bitwise_and(self.base_vessel['vessel'], self.base_vessel['vessel'], mask = bspline_branch_mapping[:,:,2])
            branch_image[:, :, 1] = self.base_vessel['vessel'] - cv2.bitwise_and(self.base_vessel['vessel'], self.base_vessel['vessel'], mask = cv2.cvtColor(bspline_branch_mapping, cv2.COLOR_BGR2GRAY))
            self.branching_pt_list.append([branch_image, '{}_{}_branching.png'.format(self.base_vessel['idx'],
                                                                                        self.img_vessel_list[idx][
                                                                                            'idx'])])


            cv2.imwrite("./{}_mask.jpg".format(self.regist_count), base_img)
            cv2.imwrite("./{}_montage.jpg".format(self.regist_count), img_mask)
            self.regist_count +=1


            mask_nonzero = np.nonzero(base_mask)
            min_x, min_y, max_x, max_y = mask_nonzero[1].min(), mask_nonzero[0].min(), mask_nonzero[1].max(), mask_nonzero[0].max()

            width = max_x-min_x
            if width < self.mask.shape[1]:
                width = self.mask.shape[1]
                min_x = 0
                max_x = self.mask.shape[1]

            height = max_y-min_y
            if height < self.mask.shape[0]:
                height = self.mask.shape[0]
                min_y = 0
                max_y = self.mask.shape[0]

            copy_image = np.zeros([height + 100, width + 100, 3]).astype(np.uint8)
            copy_mask = np.zeros([height + 100, width + 100]).astype(np.uint8)
            copy_image[50:-50, 50:-50] = base_img[min_y:max_y, min_x:max_x]
            base_img = copy_image

            temp_image = copy_mask.copy()
            temp_image[50:-50, 50:-50]= base_mask[min_y:max_y, min_x:max_x]
            base_mask = temp_image

            temp_image = copy_mask.copy()
            temp_image[50:-50, 50:-50] = self.base_vessel['vessel'][min_y:max_y, min_x:max_x]
            self.base_vessel['vessel'] = temp_image

            temp_image = copy_mask.copy()
            temp_image[50:-50, 50:-50] = self.base_vessel['branch'][min_y:max_y, min_x:max_x]
            self.base_vessel['branch'] = temp_image
        self.base_mask = base_mask
        self.base_img = base_img




    def registration_result(self, before, after):

        before_mask_sum = (before > 0).sum()

        after_mask_sum = (after > 0).sum()
        try:
            before_mask_area = np.nonzero(before)
            before_area = (before_mask_area[0].max()- before_mask_area[0].min()) * (before_mask_area[1].max()- before_mask_area[1].min())
            after_mask_area = np.nonzero(after)
            after_area = (after_mask_area[0].max()- after_mask_area[0].min()) * (after_mask_area[1].max()- after_mask_area[1].min())
        except:
            return 100

        if 0.8 > after_area/before_area.astype(np.float) or after_area/before_area.astype(np.float) > 1.2:
            return 100

        result = before_mask_sum/float(after_mask_sum) + 0.00001
        return result


    def do_bspline(self, target_fp,  base_vessel, target_vessel, base_mask, target_mask):

        overlap_region = cv2.bitwise_and(base_mask, target_mask)
        base_vessel_overlap = cv2.bitwise_and(base_vessel, base_vessel, mask=overlap_region)
        target_vessel_overlap = cv2.bitwise_and(target_vessel, target_vessel, mask=overlap_region)


        bsp = regist_BSpline(base_vessel_overlap, target_vessel_overlap)

        # regi_fpvessel : bsplined disc vessel mask, regi_disc_FOV
        before = datetime.now()
        regi_FPVessel = bsp.do_registration()
        after = datetime.now()
        regi_FPVessel = bsp.registrationFromMatrix(target_vessel).astype(np.uint8)

        regi_disc_FP = np.zeros(target_fp.shape, np.uint8)
        regi_disc_FP[:, :, 0] += bsp.registrationFromMatrix(target_fp[:, :, 0]).astype(np.uint8)
        regi_disc_FP[:, :, 1] += bsp.registrationFromMatrix(target_fp[:, :, 1]).astype(np.uint8)
        regi_disc_FP[:, :, 2] += bsp.registrationFromMatrix(target_fp[:, :, 2]).astype(np.uint8)
        regi_disc_FOV = bsp.registrationFromMatrix(target_mask).astype(np.uint8)

        print((after-before).total_seconds())
        return regi_disc_FP, regi_FPVessel, regi_disc_FOV, bsp

    def registration_no_fp_refeat(self):
        no_fp_list = self.img_none_list
        # no_fp_vessel_list = self.img_vessel_none_list
        # base_crop_nonzero = np.nonzero(self.base_mask)
        # base_crop = [base_crop_nonzero[0].min(), base_crop_nonzero[0].max(), base_crop_nonzero[1].min(), base_crop_nonzero[1].max()]
        base_img = self.base_img
        base_mask = self.base_mask.copy()
        while(no_fp_list.__len__()):
            if self.using_VPmap == True:
                base_kps, base_descs, base_sift_kpt_img, kpt_img = self.get_sift_kpt(base_img, base_mask, is_VesselMask = self.base_vessel['vessel'], hess_thresh = self.hess_thresh)
            else:
                base_kps, base_descs, base_sift_kpt_img, kpt_img = self.get_sift_kpt(base_img, base_mask, is_VesselMask=None, hess_thresh = self.hess_thresh)


            matching_good = 0
            fp_img = 0
            fp_kpt, img_vessel = 0, 0
            for no_fp_idx in range(no_fp_list.__len__()):

                fp_mask = np.zeros(base_img.shape[:2]).astype(np.uint8)
                fp_mask[:no_fp_list[no_fp_idx].shape[0], :no_fp_list[no_fp_idx].shape[1]] = self.mask
                no_fp_img = np.zeros(base_img.shape).astype(np.uint8)
                no_fp_img[:no_fp_list[no_fp_idx].shape[0], :no_fp_list[no_fp_idx].shape[1]] = no_fp_list[no_fp_idx]
                no_fp_vessel = np.zeros(base_img.shape[:2]).astype(np.uint8)
                no_fp_vessel[:no_fp_list[no_fp_idx].shape[0], :no_fp_list[no_fp_idx].shape[1]] = self.img_vessel_none_list[no_fp_idx]['vessel']

                # np_fp_kpt, no_fp_des, no_fp_kpt_img, kpt_img = self.get_sift_kpt(no_fp_img, fp_mask,
                #                                                                  is_VesselMask=None, hess_thresh=self.hess_thresh)
                if self.using_VPmap == True:
                    np_fp_kpt, no_fp_des, no_fp_kpt_img, kpt_img = self.get_sift_kpt(no_fp_img, fp_mask,
                                                                                     is_VesselMask=no_fp_vessel, hess_thresh=50)
                else:
                    np_fp_kpt, no_fp_des, no_fp_kpt_img, kpt_img = self.get_sift_kpt(no_fp_img, fp_mask,
                                                                                     is_VesselMask=None, hess_thresh=50)

                good = self.matching(base_kps, base_descs, np_fp_kpt, no_fp_des, base_mask,
                                     fp_mask, distance_m= self.matching_distance)

                # good2 = self.matching(base_kps2, base_descs2, np_fp_kpt2, no_fp_des2, self.base_mask,
                #                      fp_mask)
                h, good = self.find_homography(base_kps, np_fp_kpt, good)
                if matching_good == 0:
                    fp_img = no_fp_img.copy()
                    fp_kpt = np_fp_kpt
                    fp_vessel = no_fp_vessel
                    matching_good = [no_fp_idx, good, h]
                else:
                    if matching_good[1].__len__() < good.__len__():
                        fp_img = no_fp_img.copy()
                        fp_kpt = np_fp_kpt
                        fp_vessel = no_fp_vessel
                        matching_good = [no_fp_idx, good, h]

            matching_img = cv2.drawMatchesKnn(base_img, base_kps,
                                              fp_img, fp_kpt, matching_good[1], flags=2,
                                              outImg=None)

            # matching_img2 = cv2.drawMatchesKnn(self.base_vessel, base_kps2, no_fp_vessel, np_fp_kpt2, good2, flags=2, outImg=None)
            Image.fromarray(matching_img).save("./montage_7etdrs/{}.jpg".format(matching_good[0]))

                # Image.fromarray(matching_img2).save("./montage_7etdrs/{}_vessel.jpg".format(no_fp_idx))

            # fp_vessel = np.zeros(base_img.shape[:2])
            # fp_vessel[:img_vessel['vessel'].shape[0], :img_vessel['vessel'].shape[1]] = img_vessel['vessel']
            img_vessel = self.img_vessel_none_list[matching_good[0]]

            try:
                result_flag, [result_base_img, result_base_mask], [result_registed_img, result_registed_mask] = self.do_registration(
                    [base_img, base_mask],
                    [fp_img, fp_mask], matching_good[2])

                if result_flag == False:
                    no_fp_list.pop(matching_good[0])
                    self.img_vessel_none_list.pop(matching_good[0])
                    if no_fp_list.__len__() == 0:
                        break
                    continue


                registration_result = self.registration_result(result_registed_mask, fp_mask)
            except Exception as e:
                print(e)
                registration_result = 0



            if  0.9 < registration_result < 1.1:
                # mask_nonzero = np.nonzero(result_base_mask + result_registed_mask)
                # min_x, min_y, max_x, max_y = max(mask_nonzero[1].min() - 50, 0), max(mask_nonzero[0].min() - 50,
                #                                                                      0), min(
                #     mask_nonzero[1].max() + 50, mask_nonzero[1].max()), min(mask_nonzero[0].max() + 50,
                #                                                             mask_nonzero[0].max())

                base_img = result_base_img
                base_mask = result_base_mask
                registed_img = result_registed_img
                registed_mask = result_registed_mask
            else:

                no_fp_list.pop(matching_good[0])
                self.img_vessel_none_list.pop(matching_good[0])
                print('no disc : regist pass {}'.format(registration_result))
                if no_fp_list.__len__() == 0:
                    break
                continue

            self.base_vessel['vessel'], registed_img_vessel = self.registrationFromMatrix(self.base_vessel['vessel'], fp_vessel)
            self.base_vessel['branch'], _ = self.registrationFromMatrix(self.base_vessel['branch'],
                                                                        img_vessel['branch'])



            surf_center1, surf_center2, surf_overlap_region, surf_overlap_IOU = self.make_CEM_image(self.base_vessel['vessel'], registed_img_vessel,
                                                                                                        base_mask, result_registed_mask)

            self.surf_CEM_value += self.CEM(surf_center1, surf_center2, surf_overlap_region)


            [bspline_img, img_vessel['vessel'], bspline_mask, bsp] = self.do_bspline(registed_img, self.base_vessel['vessel'], registed_img_vessel, base_mask, registed_mask)


            base_img = pyramid_spline(base_img, bspline_img, base_mask, bspline_mask, 6)

            self.registered_image_count += 1

            center1, center2, overlap_region, bspline_overlap_IOU = self.make_CEM_image(self.base_vessel['vessel'], img_vessel['vessel'], base_mask,
                                                                    bspline_mask)
            self.bspline_CEM_value += self.CEM(center1, center2, overlap_region)
            base_mask = np.clip((base_mask.astype(np.int) + bspline_mask.astype(np.int)), 0, 255).astype(np.uint8)
            self.base_vessel['vessel'] = np.clip((self.base_vessel['vessel'].astype(np.int) + img_vessel['vessel'].astype(np.int)), 0, 255).astype(np.uint8)
            cv2.imwrite("./{}_mask.jpg".format(self.regist_count), base_img)
            cv2.imwrite("./{}_montage.jpg".format(self.regist_count), bspline_mask)
            self.regist_count += 1

            no_fp_list.pop(matching_good[0])

            mask_nonzero = np.nonzero(base_mask)
            min_x, min_y, max_x, max_y = mask_nonzero[1].min(), mask_nonzero[0].min(), mask_nonzero[1].max(), \
                                         mask_nonzero[0].max()

            width = max(max_x - min_x, self.mask.shape[1])
            if width < self.mask.shape[1]:
                min_x = 0
                max_x = self.mask.shape[1]

            height = max(max_y - min_y, self.mask.shape[0])
            if height < self.mask.shape[0]:
                min_x = 0
                max_x = self.mask.shape[0]

            width = max_x - min_x
            height = max_y - min_y
            copy_image = np.zeros([height + 100, width + 100, 3]).astype(np.uint8)
            copy_mask = np.zeros([height + 100, width + 100]).astype(np.uint8)
            copy_image[50:-50, 50:-50] = base_img[min_y:max_y, min_x:max_x]
            base_img = copy_image

            temp_image = copy_mask.copy()
            temp_image[50:-50, 50:-50] = base_mask[min_y:max_y, min_x:max_x]
            base_mask = temp_image

            temp_image = copy_mask.copy()
            temp_image[50:-50, 50:-50] = self.base_vessel['vessel'][min_y:max_y, min_x:max_x]
            self.base_vessel['vessel'] = temp_image

            temp_image = copy_mask.copy()
            temp_image[50:-50, 50:-50] = self.base_vessel['branch'][min_y:max_y, min_x:max_x]
            self.base_vessel['branch'] = temp_image



            self.img_vessel_none_list.pop(matching_good[0])
            if no_fp_list.__len__() == 0:
                break
        self.base_img = base_img
        self.base_mask = base_mask
        return base_img, self.base_vessel


class bspline_class():
    def __init__(self, target_fp, base_vessel, target_vessel, base_mask, target_mask):
        # super(regist_SIFT, self).__init__()
        padding = 400
        half_pad = 200
        self.half_pad = 200
        overlap_region = cv2.bitwise_and(base_mask, target_mask)
        base_vessel_overlap = cv2.bitwise_and(base_vessel, base_vessel, mask=overlap_region)
        target_vessel_overlap = cv2.bitwise_and(target_vessel, target_vessel, mask=overlap_region)

        self.min_x, self.min_y, self.max_x, self.max_y = self.extract_to_overlap(overlap_region, target_mask)

        # extracted image
        extract_base_overlap = base_vessel_overlap[self.min_y:self.max_y, self.min_x:self.max_x]
        extract_target_overlap = target_vessel_overlap[self.min_y:self.max_y, self.min_x:self.max_x]

        tmp_EBO = np.zeros([self.max_y - self.min_y + padding, self.max_x - self.min_x + padding]).astype(np.uint8)
        tmp_ETO = np.zeros([self.max_y - self.min_y + padding, self.max_x - self.min_x + padding]).astype(np.uint8)
        tmp_EBO[half_pad:-half_pad, half_pad:-half_pad] = extract_base_overlap
        tmp_ETO[half_pad:-half_pad, half_pad:-half_pad] = extract_target_overlap

        # registration
        self.bsp = regist_BSpline(tmp_EBO, tmp_ETO)

        # doing registration and create displacement vector
        regi_FPVessel = self.bsp.do_registration()

        # regi_fpvessel : bsplined disc vessel mask, regi_disc_FOV
        extract_target_vessel = target_vessel[self.min_y:self.max_y, self.min_x:self.max_x]

        tmp_ETV = np.zeros([self.max_y - self.min_y + padding, self.max_x - self.min_x + padding]).astype(np.uint8)
        tmp_ETV[half_pad:-half_pad, half_pad: -half_pad] = extract_target_vessel

        extract_FPV_vessel = self.bsp.registrationFromMatrix(tmp_ETV).astype(np.uint8)

        # regi_disc_FP : bsplined fundus photo
        extract_target_fp = target_fp[self.min_y:self.max_y, self.min_x:self.max_x]
        tmp_ETF = np.zeros([self.max_y - self.min_y + padding, self.max_x - self.min_x + padding, 3]).astype(np.uint8)
        tmp_ETF[half_pad: -half_pad, half_pad: -half_pad] = extract_target_fp

        regi_disc_FP = np.zeros(tmp_ETF.shape, np.uint8).astype(np.uint8)
        regi_disc_FP[:, :, 0] += self.bsp.registrationFromMatrix(tmp_ETF[:, :, 0]).astype(np.uint8)
        regi_disc_FP[:, :, 1] += self.bsp.registrationFromMatrix(tmp_ETF[:, :, 1]).astype(np.uint8)
        regi_disc_FP[:, :, 2] += self.bsp.registrationFromMatrix(tmp_ETF[:, :, 2]).astype(np.uint8)

        # extract_target_mask : bsplined mask
        extract_target_mask = target_mask[self.min_y:self.max_y, self.min_x:self.max_x]
        tmp_ETM = np.zeros([self.max_y - self.min_y + padding, self.max_x - self.min_x + padding]).astype(np.uint8)
        tmp_ETM[half_pad:-half_pad, half_pad: -half_pad] = extract_target_mask
        regi_disc_FOV = self.bsp.registrationFromMatrix(tmp_ETM).astype(np.uint8)

        target_fp[self.min_y:self.max_y, self.min_x:self.max_x] = regi_disc_FP[half_pad:-half_pad, half_pad: -half_pad]
        target_vessel[self.min_y:self.max_y, self.min_x:self.max_x] = extract_FPV_vessel[half_pad:-half_pad, half_pad: -half_pad]
        target_mask[self.min_y:self.max_y, self.min_x:self.max_x] = regi_disc_FOV[half_pad:-half_pad, half_pad: -half_pad]

        self.regi_disc_FP = target_fp
        self.regi_FPVessel = target_vessel
        self.regi_target_mask = target_mask

    def get_vector_field(self):
        displacement_image = self.bsp.get_displacement_vector_field()[1]
        return_image = np.zeros(self.regi_disc_FP.shape)
        return_image[self.min_y:self.max_y, self.min_x:self.max_x] = displacement_image[self.half_pad:-self.half_pad, self.half_pad: -self.half_pad]
        return return_image

    def get_vector_pt(self, image):
        return_image = np.zeros(list(image.shape) + [3])
        extract_image = image[self.min_y:self.max_y, self.min_x:self.max_x]
        tmp_extract_image = np.zeros(list(np.array(extract_image.shape[:2]) + 400)).astype(np.uint8)
        tmp_extract_image[self.half_pad:-self.half_pad, self.half_pad:-self.half_pad] = extract_image
        displacement_pt_image = self.bsp.get_displacement_vector_pt(tmp_extract_image)
        return_image[self.min_y:self.max_y, self.min_x:self.max_x] = displacement_pt_image[self.half_pad:-self.half_pad, self.half_pad: -self.half_pad]
        return return_image



    def do_bspline(self, target_fp, base_vessel, target_vessel, base_mask, target_mask):
        overlap_region = cv2.bitwise_and(base_mask, target_mask)
        base_vessel_overlap = cv2.bitwise_and(base_vessel, base_vessel, mask=overlap_region)
        target_vessel_overlap = cv2.bitwise_and(target_vessel, target_vessel, mask=overlap_region)

        bsp = regist_BSpline(base_vessel_overlap, target_vessel_overlap)

        # regi_fpvessel : bsplined disc vessel mask, regi_disc_FOV
        before = datetime.now()
        regi_FPVessel = bsp.do_registration()
        after = datetime.now()
        regi_FPVessel = bsp.registrationFromMatrix(target_vessel).astype(np.uint8)

        regi_disc_FP = np.zeros(target_fp.shape, np.uint8)
        regi_disc_FP[:, :, 0] += bsp.registrationFromMatrix(target_fp[:, :, 0]).astype(np.uint8)
        regi_disc_FP[:, :, 1] += bsp.registrationFromMatrix(target_fp[:, :, 1]).astype(np.uint8)
        regi_disc_FP[:, :, 2] += bsp.registrationFromMatrix(target_fp[:, :, 2]).astype(np.uint8)
        regi_disc_FOV = bsp.registrationFromMatrix(target_mask).astype(np.uint8)

        print((after - before).total_seconds())
        return regi_disc_FP, regi_FPVessel, regi_disc_FOV, bsp

    #overlap mask : overlap region mask, target_mask : target image region mask, return : axis information
    def extract_to_overlap(self, overlap_mask, target_mask):


        overlap_mask_nonzero = np.nonzero(overlap_mask)
        target_mask_nonzero = np.nonzero(target_mask)

        overlap_min_x, overlap_min_y, overlap_max_x, overlap_max_y = overlap_mask_nonzero[1].min(), overlap_mask_nonzero[0].min(), overlap_mask_nonzero[1].max(), overlap_mask_nonzero[0].max()
        target_min_x, target_min_y, target_max_x, target_max_y = target_mask_nonzero[1].min(), target_mask_nonzero[0].min(), target_mask_nonzero[1].max(), \
                                     target_mask_nonzero[0].max()
        min_x = min(overlap_min_x, target_min_x)
        min_y = min(overlap_min_y, target_min_y)
        max_x = max(overlap_max_x, target_max_x)
        max_y = max(overlap_max_y, target_max_y)

        return min_x, min_y, max_x, max_y









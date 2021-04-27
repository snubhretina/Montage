#!/usr/bin/env python

from __future__ import print_function

import SimpleITK as sitk
import sys
import os
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import cv2


class regist_BSpline():
    def __init__(self, img1, img2, iteration = 30):
        self.img1 = img1.astype(np.float32)
        self.img2 = img2.astype(np.float32)
        self.iteration = 15

        self.itk_img1 = sitk.GetImageFromArray(self.img1)
        self.itk_img2 = sitk.GetImageFromArray(self.img2)
        self.outTx = None
    def do_registration(self):
        transformDomainMeshSize = [8] * self.itk_img2.GetDimension()
        tx = sitk.BSplineTransformInitializer(self.itk_img1,
                                              transformDomainMeshSize)

        R2 = sitk.ImageRegistrationMethod()
        R2.SetMetricAsCorrelation()
        R2.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
                               numberOfIterations=self.iteration,
                               maximumNumberOfCorrections=5,
                               maximumNumberOfFunctionEvaluations=1000,
                               costFunctionConvergenceFactor=1e+7)
        R2.DebugOff()
        R2.GlobalWarningDisplayOff()
        R2.SetInitialTransform(tx, True)
        R2.SetInterpolator(sitk.sitkLinear)

        # R = sitk.ImageRegistrationMethod()
        # R.SetMetricAsJointHistogramMutualInformation()

        # R.SetOptimizerAsGradientDescentLineSearch(5.0,
        #                                           50,
        #                                           convergenceMinimumValue=1e-6,
        #                                           convergenceWindowSize=10)
        #
        # R.SetInterpolator(sitk.sitkLinear)
        #
        # R.SetInitialTransformAsBSpline(tx,
        #                                inPlace=True,
        #                                scaleFactors=[1, 2, 5])
        # R.SetShrinkFactorsPerLevel([4, 2, 1])
        # R.SetSmoothingSigmasPerLevel([4, 2, 1])


        # R = sitk.ImageRegistrationMethod()
        # R.SetMetricAsJointHistogramMutualInformation()
        #
        # R.SetOptimizerAsGradientDescentLineSearch(5.0,
        #                                           100,
        #                                           convergenceMinimumValue=1e-4,
        #                                           convergenceWindowSize=5)
        #
        # R.SetInterpolator(sitk.sitkLinear)
        #
        # R.SetInitialTransformAsBSpline(tx,
        #                                inPlace=True,
        #                                scaleFactors=[1, 2, 5])
        # R.SetShrinkFactorsPerLevel([4, 2, 1])
        # R.SetSmoothingSigmasPerLevel([4, 2, 1])

        # outTx = R.Execute(self.itk_img1, self.itk_img2)
        # self.outTx = outTx
        # self.resampler = sitk.ResampleImageFilter()
        # self.resampler.SetReferenceImage(self.itk_img1)
        # self.resampler.SetInterpolator(sitk.sitkLinear)
        # self.resampler.SetDefaultPixelValue(0)
        # self.resampler.SetTransform(outTx)
        #
        # out = self.resampler.Execute(self.itk_img2)

        outTx2 = R2.Execute(self.itk_img1, self.itk_img2)
        self.outTx2 = outTx2
        self.resampler2 = sitk.ResampleImageFilter()
        self.resampler2.SetReferenceImage(self.itk_img1)
        self.resampler2.SetInterpolator(sitk.sitkLinear)
        self.resampler2.SetDefaultPixelValue(0)
        self.resampler2.SetTransform(outTx2)
        out = self.resampler2.Execute(self.itk_img2)


        out = sitk.GetArrayFromImage(out)

        return out

    def registrationFromMatrix(self, img):
        # resampler = sitk.ResampleImageFilter()
        # resampler.SetReferenceImage(self.itk_img1)
        # resampler.SetInterpolator(sitk.sitkLinear)
        # resampler.SetDefaultPixelValue(0)
        # self.resampler.SetTransform(outTx)

        out = self.resampler2.Execute(sitk.GetImageFromArray(img.astype(np.float32).copy()))
        out = sitk.GetArrayFromImage(out)

        return out


    def get_displacement_vector_field(self):
        all_disp = np.zeros(list(self.img1.shape) + [2], dtype=np.int)
        draw_disp = np.zeros(list(self.img1.shape) + [3], dtype=np.ubyte)
        # disp[:, :, 0] = self.img1
        draw_disp[:, :, 1] = self.img1
        draw_disp[:, :, 2] = self.img2
        for y in range(self.img1.shape[0]):
            for x in range(self.img1.shape[1]):
                mov = self.outTx2.TransformPoint([x, y])
                mov = list(np.array(mov, dtype=int))
                d = [mov[0] - x, mov[1] - y]
                all_disp[y, x] = d
                if y%20==0 and x%20==0:
                    if mov[0] > 0 and mov[1] > 0 and mov[0] < self.img1.shape[1] and mov[1] < self.img1.shape[0]:
                        cv2.arrowedLine(draw_disp, (x + d[0] * 5, y + d[1] * 5), (x, y), (255, 0, 0), 1, tipLength=0.5)
                        cv2.arrowedLine(draw_disp, (x + d[0] * 5, y + d[1] * 5), (x, y), (255, 0, 0), 1, tipLength=0.5)

        return all_disp, draw_disp

    def get_displacement_vector_pt(self, pt_map):
        draw_disp = np.zeros(list(self.img1.shape) + [3], dtype=np.ubyte)
        all_disp = np.zeros(list(self.img1.shape) + [2], dtype=np.int)
        for pt_idx in cv2.findNonZero(pt_map):
            x, y = int(pt_idx[0][0]), int(pt_idx[0][1])
            mov = self.outTx2.TransformPoint([x, y])
            # mov = list(np.array(mov, dtype=int))
            mov = list(np.around(mov).astype(np.int))
            d = [mov[0] - x, mov[1] - y]

            if mov[0] > 0 and mov[1] > 0 and mov[0] < self.img1.shape[1] and mov[1] < self.img1.shape[0]:
                all_disp[y, x] = d
                draw_disp[y - d[1], x - d[0]] = 255
        # try:
        #     for pt_idx in cv2.findNonZero(pt_map):
        #         x,y = int(pt_idx[0][0]), int(pt_idx[0][1])
        #         mov = self.outTx2.TransformPoint([x, y])
        #         # mov = list(np.array(mov, dtype=int))
        #         mov = list(np.around(mov).astype(np.int))
        #         d = [mov[0] - x, mov[1] - y]
        #
        #         if mov[0] > 0 and mov[1] > 0 and mov[0] < self.img1.shape[1] and mov[1] < self.img1.shape[0]:
        #             all_disp[y, x] = d
        #             draw_disp[y - d[1], x - d[0]] = 255
        # except Exception as e:
        #     print(e)

        return draw_disp


# def command_iteration(method) :
#     print("{0:3} = {1:10.5f}".format(method.GetOptimizerIteration(),
#                                      method.GetMetricValue()))
#
# # if len ( sys.argv ) < 4:
# #     print( "Usage: {0} <fixedImageFilter> <movingImageFile> <outputTransformFile>".format(sys.argv[0]))
# #     sys.exit ( 1 )
#
#
# # fixed = sitk.ReadImage(sys.argv[1], sitk.sitkFloat32)
# #
# # moving = sitk.ReadImage(sys.argv[2], sitk.sitkFloat32)
# img1 = np.asanyarray(Image.open('img_FA_02_MutInform.png'), dtype=np.float32)[:,:]
# img2 = np.asanyarray(Image.open('img_FA_04_MutInform.png'), dtype=np.float32)[:,:]
# fixed = sitk.GetImageFromArray(img1)
#
# moving = sitk.GetImageFromArray(img2)
#
# transformDomainMeshSize=[8]*moving.GetDimension()
# tx = sitk.BSplineTransformInitializer(fixed,
#                                       transformDomainMeshSize )
#
# print("Initial Parameters:");
# print(tx.GetParameters())
#
# R = sitk.ImageRegistrationMethod()
# R.SetMetricAsCorrelation()
#
# R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
#                        numberOfIterations=10,
#                        maximumNumberOfCorrections=5,
#                        maximumNumberOfFunctionEvaluations=1000,
#                        costFunctionConvergenceFactor=1e+7)
# R.SetInitialTransform(tx, True)
# R.SetInterpolator(sitk.sitkLinear)
#
# # R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )
#
# outTx = R.Execute(fixed, moving)
#
# print("-------")
# print(outTx)
# print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
# print(" Iteration: {0}".format(R.GetOptimizerIteration()))
# print(" Metric value: {0}".format(R.GetMetricValue()))
#
# # sitk.WriteTransform(outTx,  './tf.tf')
#
# # if ( not "SITK_NOSHOW" in os.environ ):
#
# resampler = sitk.ResampleImageFilter()
# resampler.SetReferenceImage(fixed);
# resampler.SetInterpolator(sitk.sitkLinear)
# resampler.SetDefaultPixelValue(0)
# resampler.SetTransform(outTx)
#
# out = resampler.Execute(moving)
# img = sitk.GetArrayFromImage(out)
# Image.fromarray(img.astype(np.ubyte)).save('check.png')
# # simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
# # simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
# # cimg = sitk.Compose(simg1, simg2, simg1//2.+simg2//2.)
# # sitk.Show( cimg, "ImageRegistration1 Composition" )
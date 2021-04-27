from registration_SIFT_cleansing import regist_SIFT_montage
import find_disc_fovea_Faster_RCNN
import cv2, os, glob, csv, argparse
from datetime import datetime
import numpy as np
import utils
from PIL import Image
from plantcv import plantcv as pcv

parser = argparse.ArgumentParser(description='Fully Leveraging Deep Learning Methods for Constructing Retinal Fundus Photomontages')
parser.add_argument('--path', default="./data/", type=str, help='dir path for input')
parser.add_argument('--res_path', default="./res/", type=str, help='dir path for output')

args = parser.parse_args()


def cut_background(img):
    """
    image cropping of center, cut black space
    :param img_path: image
    :return: cut image
    """
    height, width = img.shape[0], img.shape[1]
    tmp_img = img[:, int(width / 2 - height / 2): int(width / 2 + height / 2)]
    return tmp_img

def disc_sort_img(image_list, img_vessel_list, base_label):
    """
    image sorting have disc and fovea image
    :param image_list: fundus photo image list with disc or fovea
    :param img_vessel_list: vessel image of image_list
    :param base_label: image that disc position is center
    :return: sorted image list
    """

    return_vessel_list = []
    return_img_list = []
    for idx, vessel_idx in enumerate(img_vessel_list):
        if vessel_idx['idx'] == base_label:
            return_img_list.append(image_list[idx])
            return_vessel_list.append(vessel_idx)
            img_vessel_list.pop(idx)
            image_list.pop(idx)

    disc_exist_image_list = []
    disc_exist_vessel_list = []
    # OM_info 0 is fovea, 1 is disc


    for idx, vessel_idx in enumerate(img_vessel_list):
        tmp_vessel_idx = vessel_idx.copy()
        if tmp_vessel_idx['OM_info'][1] != None:
            tmp_vessel_idx['OM_info'] = np.sqrt(np.power(np.array(tmp_vessel_idx['OM_info'][1]) - np.array(return_vessel_list[0]['OM_info'][1]),2).sum())
            disc_exist_image_list.append([image_list[idx], tmp_vessel_idx['OM_info']])
            disc_exist_vessel_list.append(tmp_vessel_idx)


    disc_exist_image_list = sorted(disc_exist_image_list, key= lambda image_info : image_info[1])
    disc_exist_vessel_list = sorted(disc_exist_vessel_list, key=lambda vessel_info : vessel_info['OM_info'])

    fovea_exist_image_list = []
    fovea_exist_vessel_list = []
    for idx, vessel_idx in enumerate(img_vessel_list):
        tmp_vessel_idx = vessel_idx.copy()
        if tmp_vessel_idx['OM_info'][1] == None and tmp_vessel_idx['OM_info'][0] != None:
            tmp_vessel_idx['OM_info'] = np.sqrt(
                np.power(np.array(tmp_vessel_idx['OM_info'][0]) - np.array(return_vessel_list[0]['OM_info'][0]), 2).sum())
            fovea_exist_image_list.append([image_list[idx], tmp_vessel_idx['OM_info']])
            fovea_exist_vessel_list.append(tmp_vessel_idx)

    fovea_exist_image_list = sorted(fovea_exist_image_list, key=lambda image_info: image_info[1])
    fovea_exist_vessel_list = sorted(fovea_exist_vessel_list, key=lambda vessel_info : vessel_info['OM_info'])

    for idx, disc_exist_vessel_list_idx in enumerate(disc_exist_vessel_list):
        return_img_list.append(disc_exist_image_list[idx][0])
        return_vessel_list.append(disc_exist_vessel_list_idx)

    for idx, fovea_exist_vessel_list_idx in enumerate(fovea_exist_vessel_list):
        return_img_list.append(fovea_exist_image_list[idx][0])
        return_vessel_list.append(fovea_exist_vessel_list_idx)

    return return_img_list, return_vessel_list

def make_dir(dir_path):
    """
    make directory in python2, if you use python3, it probably replace os.makedir function
    :param dir_path: path
    :return: no return
    """
    if os.path.isdir(dir_path) == False:
        os.mkdir(dir_path)

def mapped(center_pt, mapping):
    """
    mapping position of faster rcnn input scale to image scale
    :param center_pt:
    :param mapping:
    :return: if center pt not none, return center pt, no return None
    """
    if center_pt != None:
        center_pt[0] = int(center_pt[0] * mapping[0] / 512.)
        center_pt[1] = int(center_pt[1] * mapping[1] / 512.)
        return [center_pt[1], center_pt[0]]
    else:
        return None

def Mosaic_FP(input_path, result_path, preprocessing = True, detection = True):
    #input_path = "../jooyoung/DB_extract/example_seven_standard_extension/"

    img_path_list = sorted(glob.glob(input_path + "/*"))
    fpmodel = utils.get_model()

    result_path = "{}{}".format(result_path, input_path.split("/")[-1])
    if preprocessing == True:
        result_path = result_path + "_preprocessing"
    if detection == True:
        result_path = result_path + "_detection"
    else:
        result_path = result_path + "_nodetection"
    print(result_path)
    make_dir(result_path)
    csv_name = open("{}/result_TRE.csv".format(result_path), 'w')
    file_csv = csv.writer(csv_name)
    file_csv.writerow(['pair_list', 'overlap', 'surf_TRE', 'bsplined_TRE'])

    for input_path_idx, input_path in enumerate(img_path_list):
        idx_result_path = "{}/{}_{}/".format(result_path, input_path_idx, input_path.split("/")[-1])

        img_list = sorted(glob.glob(input_path + "/*"))

        ####20200804

        make_dir(idx_result_path)

        laterality_result_path = idx_result_path + "/"

        make_dir(laterality_result_path)
        disc_exist = []
        disc_no = []
        disc_exist_vessel = []
        disc_no_vessel = []
        center_pt = []

        exist_value = -1


        #make mask
        for img_idx, i in enumerate(img_list):
            if i[-3:] != 'jpg':
                continue
            img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
            if img_idx == 0:
                mask = img.astype(np.int)
            else:
                mask += img

        mask = np.clip(mask, 0, 255).astype(np.uint8)
        mask = cv2.threshold(mask, 6 * img_list.__len__(), 255, cv2.THRESH_BINARY)[1]
        mask_label_num, mask_label_img, mask_img_stats, _ = cv2.connectedComponentsWithStats(mask)
        mask_label_mask_index = mask_img_stats[1:,4].argmax() + 1
        mask = np.array((mask_label_img == mask_label_mask_index) * 255).astype(np.uint8)
        mask = cv2.threshold(mask, 125, 255, cv2.THRESH_BINARY)[1]
        mask = cut_background(mask)


        calculate_time = 0
        for img_idx, image_path in enumerate(img_list):

            if image_path[-3:] != 'jpg':
                continue
            img = Image.open(image_path)
            img = Image.fromarray(cut_background(np.array(img)))
            org_img = img
            if img.size[0] != mask.shape[1] or img.size[1] != mask.shape[0]:
                img = np.array(img.resize((mask.shape[1],mask.shape[0]), Image.LANCZOS))
            else:
                img = np.array(img)

            img = cv2.bitwise_and(img, img, mask = mask)

            img_to_vessel = utils.FP_vessel_prediction2(fpmodel, np.array(org_img).astype(np.uint8), mask = None,
                                       size=[img.shape[0], img.shape[1]])
            img_to_vessel = cv2.threshold(img_to_vessel, 30, 255, cv2.THRESH_BINARY)[1]
            #using processing
            if result_path.find("processing")>-1:
                label_num, label_img, img_stats, _ = cv2.connectedComponentsWithStats(img_to_vessel)
                second_label, max_label = img_stats[1:,4].argsort()[-2:]
                root_vessel = np.zeros(label_img.shape).astype(np.uint8)
                for label_idx in range(label_num):
                    if label_idx in [second_label, max_label]:
                        if img_stats[label_idx+1][2] * img_stats[label_idx+1][3] < 10000:
                            continue


                        root_vessel += np.array((label_img == label_idx + 1) * 255).astype(np.uint8)

                    else:
                        root_vessel += np.array((label_img == label_idx + 1) * 0).astype(np.uint8)
            # no processing
            else:
                root_vessel = img_to_vessel

            img_to_vessel = root_vessel

            mapping_shape = list(mask.shape)
            if result_path.find("detection") > -1:
                detection_before = datetime.now()
                [fovea, disc], disc_length = find_disc_fovea_Faster_RCNN.find_disc_fovea(org_img)


                if fovea != None or disc != None:
                    disc = mapped(disc, mapping_shape)
                    fovea = mapped(fovea, mapping_shape)
                    exist_value += 1
                    if fovea != None and disc != None:
                        center_pt.append([fovea, img_idx])


                    disc_exist.append(img)
                    disc_exist_vessel.append({'vessel' : img_to_vessel, 'idx' : img_idx, 'OM_info' : [fovea, disc]})
                else:
                    disc_no.append(img)
                    disc_no_vessel.append({'vessel': img_to_vessel, 'idx': img_idx})

                detection_after = datetime.now()
                calculate_time += (detection_after - detection_before).total_seconds()
            else:
                disc_exist.append(img)
                disc_exist_vessel.append({'vessel': img_to_vessel, 'idx': img_idx})


        #center_pt 0 is fovea center, 1 is image idx
        if result_path.find("detection") > -1:

            distance_between_imgcenter_macular = 10000000
            disc_fovea_exist = None
            # if macular center image none, this is perform
            if center_pt.__len__() == 0:
                for disc_exist_vessel_idx in disc_exist_vessel:
                    min_distance = np.sqrt(
                        np.power(disc_exist_vessel_idx['OM_info'][1] - np.array(list(mask.shape)) / 2, 2).sum())
                    if distance_between_imgcenter_macular > min_distance:
                        distance_between_imgcenter_macular = min_distance
                        disc_fovea_exist = disc_exist_vessel_idx['idx']
            for center_pt_idx in center_pt:
                #ominfo 0 x 1 y
                min_distance = np.sqrt(np.power(np.array(center_pt_idx[0]) - np.array(list(mask.shape))/2, 2).sum())
                if distance_between_imgcenter_macular > min_distance:
                    distance_between_imgcenter_macular = min_distance
                    disc_fovea_exist = center_pt_idx[1]

            montage_image_list, montage_vessel_list = disc_sort_img(disc_exist, disc_exist_vessel, disc_fovea_exist)


        if result_path.find("pair") > -1:
            montage1 = regist_SIFT_montage([disc_exist, disc_exist_vessel], [disc_no, disc_no_vessel],
                                           mask, using_detection=False,
                                           center_pt=center_pt)
            montage1.registration_pair()
        else:
            if result_path.find("detection") > -1:
                montage1 = regist_SIFT_montage([montage_image_list, montage_vessel_list],
                                               [disc_no, disc_no_vessel], mask, using_detection=True,
                                               center_pt=center_pt)
                montage1.registration_exist_fp_repeat()
                mosaic_fp, mosaic_vessel = montage1.registration_no_fp_refeat()
            else:
                montage1 = regist_SIFT_montage([disc_exist, disc_exist_vessel],
                                               [disc_no, disc_no_vessel], mask, using_detection=False,
                                               center_pt=center_pt)
                mosaic_fp, mosaic_vessel = montage1.registration_non_detection()
            Image.fromarray(mosaic_fp).save(laterality_result_path + "mosaic_result.png")
            Image.fromarray(mosaic_vessel['vessel']).save(laterality_result_path + "mosaic_vessel_result.png")

        try:
            for value_length in range(montage1.TRE_pair.__len__()):
                file_csv.writerow([str(input_path_idx) + "_" + montage1.TRE_pair[value_length][0],
                                   montage1.TRE_pair[value_length][1], montage1.TRE_pair[value_length][2], montage1.TRE_pair[value_length][3],
                                   montage1.TRE_pair[value_length][4], montage1.TRE_pair[value_length][5]])
        except:
            continue



Mosaic_FP("../jooyoung/DB_extract/example_seven_standard_extension","./", True, True)


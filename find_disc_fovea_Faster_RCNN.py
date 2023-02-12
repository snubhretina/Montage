import os
import numpy as np
import torch
from PIL import Image, ImageDraw
import os, glob, cv2, sys, csv, time, math, shutil
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import sys

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# load a model pre-trained pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 3  # 1 class (person) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)
model.load_state_dict(torch.load("./12.pth"))
model.eval()

def visualize_result(result, img):
    draw = ImageDraw.Draw(img)
    for result_idx, result_boxes in enumerate(result['boxes']):
        x1, y1, x2, y2 = result_boxes
        labels = result['labels'][result_idx]
        scores = result['scores'][result_idx]
        # print(labels, scores)
        # disc_length = 0
        if labels == 1:
            draw.rectangle((int(x1), int(y1), int(x2), int(y2)), fill = (255, 255, (labels - 1) * 255), outline = (255, 255, (labels - 1) * 255))
        elif labels == 2:
            draw.rectangle((int(x1), int(y1), int(x2), int(y2)), fill=(255, 255, (labels - 1) * 255),
                           outline=(255, 255, (labels - 1) * 255))
    return img


def find_disc_fovea_example(img):
    if type(img).__name__  == 'str':
        image = Image.open(img).convert('L').convert('RGB')
    else:
        image = img
    resized_image = image.resize((512, 512))
    # center_box_imgage = np.array(resized_image)
    result = model([TF.to_tensor(resized_image).to(device)])[0]
    result_list = {'boxes': result['boxes'].tolist(), 'labels': result['labels'].tolist(),
                   'scores': result['scores'].tolist()}

    # intermidate_result = model.backbone(model.transform([TF.to_tensor(resized_image).to(device)])[0].tensors)
    # # last_intermidate_output = intermidate_result['pool'][0].mul(255).byte().permute(1,2,0).cpu().data.numpy()
    # last_intermidate_output = intermidate_result[3].mul(255).byte().cpu().data.numpy().squeeze(0).transpose((1,2,0))
    # for i in range(last_intermidate_output.shape[2]):
    #     tmp = last_intermidate_output[:,:,i]
    #     tmp = cv2.resize(tmp, (512,512))
    #     cv2.imwrite("./intermidate_output/{}.png".format(i), tmp)

    # cv2_img = np.array(resized_image)
    # cv2_img_all = np.array(resized_image)
    fovea = None
    disc = None
    disc_length = 0
    for result_idx, result_boxes in enumerate(result_list['boxes']):
        x1, y1, x2, y2 = result_boxes
        labels = result_list['labels'][result_idx]
        scores = result_list['scores'][result_idx]
        # print(labels, scores)
        # disc_length = 0
        if scores > 0.8:
            #label 1 : fovea 2 : disc
            if labels == 1:
                fovea = [int((y2 + y1) / 2.), int((x2 + x1) / 2.)]
            elif labels == 2:
                disc = [int((y2 + y1) / 2.), int((x2 + x1) / 2.)]
                # disc_length = max(y2-y1, x2-x1)/2.
            # binary_str = "{0:02b}".format(labels)
            # resized_image.rectangle([(int(x1),int(y1)), (int(x2),int(y2))], 2, (255, 255, (labels - 1) * 255))
    return [fovea, disc], [disc_length]
def mapped(center_pt, mapping):
    if center_pt != None:
        center_pt[0] = int(center_pt[0] * mapping[0] / 512.)
        center_pt[1] = int(center_pt[1] * mapping[1] / 512.)
        return [center_pt[1], center_pt[0]]
    else:
        return None

def find_disc_fovea(img):
    if type(img).__name__  == 'str':
        image = Image.open(img).convert('L').convert('RGB')
    else:
        image = Image.fromarray(img)
    mapping_shape = list(np.array(image).shape)
    resized_image = image.resize((512, 512))
    # center_box_imgage = np.array(resized_image)
    result = model([TF.to_tensor(resized_image).to(device)])[0]
    result_list = {'boxes': result['boxes'].tolist(), 'labels': result['labels'].tolist(),
                   'scores': result['scores'].tolist()}

    # intermidate_result = model.backbone(model.transform([TF.to_tensor(resized_image).to(device)])[0].tensors)
    # # last_intermidate_output = intermidate_result['pool'][0].mul(255).byte().permute(1,2,0).cpu().data.numpy()
    # last_intermidate_output = intermidate_result[3].mul(255).byte().cpu().data.numpy().squeeze(0).transpose((1,2,0))
    # for i in range(last_intermidate_output.shape[2]):
    #     tmp = last_intermidate_output[:,:,i]
    #     tmp = cv2.resize(tmp, (512,512))
    #     cv2.imwrite("./intermidate_output/{}.png".format(i), tmp)

    # cv2_img = np.array(resized_image)
    # cv2_img_all = np.array(resized_image)
    fovea = None
    disc = None
    disc_length = 0
    for result_idx, result_boxes in enumerate(result_list['boxes']):
        x1, y1, x2, y2 = result_boxes
        labels = result_list['labels'][result_idx]
        scores = result_list['scores'][result_idx]
        # print(labels, scores)
        # disc_length = 0
        if scores > 0.8:
            #label 1 : fovea 2 : disc
            if labels == 1:
                fovea = [int((y2 + y1) / 2.), int((x2 + x1) / 2.)]
            elif labels == 2:
                disc = [int((y2 + y1) / 2.), int((x2 + x1) / 2.)]
                # rect_img = cv2.rectangle(np.array(resized_image).astype(np.uint8), (int(x1),int(y1)), (int(x2),int(y2)), 255, 1)
                # cv2.imwrite("tmp.png", rect_img)
                # disc_length = max(y2-y1, x2-x1)/2.
            # binary_str = "{0:02b}".format(labels)
            # resized_image.rectangle([(int(x1),int(y1)), (int(x2),int(y2))], 2, (255, 255, (labels - 1) * 255))
    disc = mapped(disc, mapping_shape)
    fovea = mapped(fovea, mapping_shape)
    return [fovea, disc], disc_length


# # img_list = glob.glob("../10_Wide_fundus_photo_disc-fovea_center_extraction/DB_color/resized/*")
# img_list = ["KakaoTalk_20220711_142440895.jpg", "00340734_20200130_OD_1_fp.png"]
# for img_idx in sorted(img_list):
#     img = Image.open(img_idx)
#     result_img = np.array(img)
#     mapping_shape = list(result_img.shape)
#     [fovea, disc], disc_length = find_disc_fovea_example(img)
#     disc = mapped(disc, mapping_shape)
#     fovea = mapped(fovea, mapping_shape)
#     try:
#         circle_img = cv2.circle(result_img, tuple(disc), 5, (255, 0, 0), 5)
#     except:
#         pass
#     try:
#         circle_img = cv2.circle(circle_img, tuple(fovea), 5, (0, 0, 255), 5)
#     except:
#         pass

#     Image.fromarray(circle_img).save("./test_image/{}.jpg".format(img_idx.split("/")[-1][:-4]))
# Fully Leveraging Deep Learning Methods for Constructing Retinal Fundus Photomontages


## Introduction
This repository described in the paper "Fully Leveraging Deep Learning Methods for Constructing Retinal Fundus Photomontages" (https://www.mdpi.com/2076-3417/11/4/1754)
![example](https://user-images.githubusercontent.com/64057617/120308126-b6dfa280-c30e-11eb-9535-2490e32c61aa.PNG)
## Usage

### Installation
```
git clone git@github.com:snubhretina/Montage.git
cd Montage
pip3 install -r requirements.txt
```

* Download the pretrained Vessel extraction models form [here]. This model is trained DRIVE Database. Our model can't provide cause trained our SNUBH internel DB.
* Also, Our Faster RCNN model for detecting disc and fovea center can't provide same issue. So we provide our training code for detecting disc and fovea in Faster_RCNN_train.py
* Unzip and move the pretrained parameters to models/

### Run

```
python main.py --input_path="./data" --output_path="./res/" --seg_model_path = "./model/seg_model.pth" --detection_model_path = "./model/detection_model.pth"
```
you can choose whether to use detection model if detection_model_path argument is not, process image sorting with keypoint matching .

**Train optic disc and fovea detection**
```
python Faster_RCNN_train.py --input_path="./" --output_path="./res/"
```

train faster rcnn is based on pytorch. you can find additional information in this cite [here] https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html. 


## Citation
```
@article{kim2021fully,
  title={Fully Leveraging Deep Learning Methods for Constructing Retinal Fundus Photomontages},
  author={Kim, Jooyoung and Go, Sojung and Noh, Kyoung Jin and Park, Sang Jun and Lee, Soochahn},
  journal={Applied Sciences},
  year={2021}
}
```

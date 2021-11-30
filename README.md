# BALLAD
This is the official code repository for [*A Simple Long-Tailed Rocognition Baseline via Vision-Language Model.*](https://arxiv.org/pdf/2111.14745.pdf)

![image](https://github.com/gaopengcuhk/BALLAD/blob/main/figure.PNG)

## Requirements
* Python3
* Pytorch(1.7.1 recommended)
* yaml
* other necessary packages

## Datasets
* ImageNet_LT
* Places_LT

Download the [ImageNet_2014](http://image-net.org/index) and [Places_365](http://places2.csail.mit.edu/download.html).

Modify the data_root in [main.py](main.py) to refer to your own dataset path.

## Training

#### Phase A
```
python main.py --cfg ./config/ImageNet_LT/clip_A_rn50.yaml
```

#### Phase B
```
python main.py --cfg ./config/ImageNet_LT/clip_B_rn50.yaml
```

## Testing
```
python main.py --cfg ./config/ImageNet_LT/test.yaml --test
```

## Acknowledgments

The codes is based on [https://github.com/zhmiao/OpenLongTailRecognition-OLTR](https://github.com/zhmiao/OpenLongTailRecognition-OLTR) and motivated by [https://github.com/facebookresearch/classifier-balancing](https://github.com/facebookresearch/classifier-balancing).



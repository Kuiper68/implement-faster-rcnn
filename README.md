# Implement Faster-RCNN

딥러닝 모델 ***Faster R-CNN (Object Detection)*** 구현 프로젝트 입니다. 아직 모델의 성능적인 면에서 이슈가 있습니다. 이에 대해서는 차후 개선해 나갈 예정입니다.

---
## Description

<img src="src/object_detection_images.png" width="auto">

Faster R-CNN 은 객체를 경계상자로 표시하는 object detection 방법론 중 하나이며 객체 영역을 추천하는데 ``Selective Search`` 가 아닌 ``Region Proposal Network`` 으로 불리는 네트워크(모델)를 사용하여 이전의 Fast R-CNN 보다 훨씬 빠르게 추천할 수 있게 되었다는 점에서 의미가 있는 딥러닝 모델입니다.

---
## Environment

- OS: Ubuntu 20.04.4 LTS 64bit
- Language: python 3.8.10

---
## Prerequisite

다음 파이썬 패키지가 필요합니다.

- tensorflow 2.7.0
- opencv-python 4.5.5
- numpy 1.21.3
- tqdm 4.64.0

---

## How to use

##### 1. Download Data

데이터를 다운로드합니다. 본 프로젝트에서는 모델 훈련에 PASCAL VOC2007 데이터를 사용합니다.

```shell
sh pascal_voc_2007.sh
```
<br>

##### 2. Train Model

Faster R-CNN 모델을 훈련합니다.

```shell
python train.py
```
<br>

##### 3. Test Model
훈련된 모델을 테스트합니다.

```shell
# Test image
python test.py image --path input/image/path

# Test Video
python test.py video --path input/video/path

# Test WebCam
python test.py webcam
```

---

## Test Sample

<img src="src/test_1.gif" width="640">
<br><br>

<img src="src/test_2.gif" width="640">
<br><br>

---

## Reference
- **Paper**: [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
- **Data**: [PASCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/)
- **Video1**: [Video Detection with Tensorflow](https://youtu.be/Q3lKlzi_cEw)
- **Video2**: [Masaka Kids Africana Dancing To Jerusalema By Master KG Feat Nomcebo & Burna Boy](https://youtu.be/TH4V-yHbJXk)

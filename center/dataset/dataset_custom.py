from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pycocotools.coco as coco

import torch.utils.data as data
import numpy as np
import cv2
import os
from utils.heatmap import draw_umich_gaussian
import albumentations as A

class DATASET_CUSTOM(data.Dataset):


  def __init__(self, config, split):
    """

    :param opt:
    :param split: train/val
    """
    super(DATASET_CUSTOM, self).__init__()
    self.data_dir = config['dataset']['data_dir']
    self.img_dir = os.path.join(self.data_dir, 'images')
    self.input_h = config['model']['input_h']
    self.input_w = config['model']['input_h']
    self.pad = config['model']['pad']
    self.down_ratio = config['model']['down_ratio']
    self.mean = config['dataset']['mean']
    self.std = config['dataset']['std']
    self.max_objs = config['dataset']['max_object']
    self.num_classes = config['dataset']['num_classes']
    self.radius = config['dataset']['radius']

    self.annot_path = os.path.join(
          self.data_dir, 'annotations',
          '{}.json').format(split)
    # print(self.data_dir)

    self.class_name = ['__background__'] + config['dataset']['label_name']
    self._valid_ids = [_id for _id in range(1, self.num_classes+1)]    # [1,2,..self.num_classes]
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}        # {1:0,

    self.split = split

    print('==> initializing {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

    self.output_h = self.input_h // self.down_ratio  # 512/4 = 128
    self.output_w = self.input_w // self.down_ratio

    self.transform_train = A.Compose(
                                [
                                    A.OneOf([
                                            A.RandomBrightnessContrast(brightness_limit=0.5,
                                                                          contrast_limit=0.4),
                                            A.RandomGamma(gamma_limit=(50, 150)),
                                            A.NoOp()
                                        ]),
                                    A.OneOf([
                                            A.RGBShift(r_shift_limit=20, b_shift_limit=15,
                                                          g_shift_limit=15),
                                            A.HueSaturationValue(hue_shift_limit=5,
                                                                    sat_shift_limit=5),
                                            A.NoOp()
                                        ]),
                                    A.HorizontalFlip(p=0.5),    #OK
                                    A.ShiftScaleRotate(shift_limit=[0.1, 0.1], scale_limit=[0,0], rotate_limit=[-45, 45], p=0.5, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255)),    #OK
                                    A.Downscale(scale_min=0.1, scale_max=0.2, p=0.3),      # OK
                                    # A.CoarseDropout(max_holes=5, max_height=100, max_width=100, min_holes=3, min_height=64, min_width=64, p=0.5),   # error
                                    A.CLAHE(p=0.5),
                                    A.Resize(height=self.input_h, width=self.input_w, interpolation=cv2.INTER_LINEAR, always_apply=True),
                                    A.Normalize(mean=self.mean, std=self.std, always_apply=True)
                                ],
                                keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels'])
    )

    self.transform_heatmap = A.Compose(
                                [
                                    A.Resize(height=self.output_h, width=self.output_w, interpolation=cv2.INTER_LINEAR, always_apply=True)
                                ]
                                , keypoint_params=A.KeypointParams(format='xy')
    )

    self.transform_test = A.Compose(
                                [
                                    A.Resize(height=self.input_h, width=self.input_w, interpolation=cv2.INTER_LINEAR, always_apply=True),
                                    A.Normalize(mean=self.mean, std=self.std, always_apply=True)
                                ],
                                keypoint_params=A.KeypointParams(format='xy')
    )


  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def _cocobox_to_center(self, box):
      center = np.array([box[0] + box[2] // 2, box[1] + box[3] // 2], dtype=np.uint16)  # [x_center, y_center]

      return center

  def __getitem__(self, index):
      """
      img_size -(1)-> input_size -(2)-> output_size ( for heatmap)
      1: only resize : new keypoint, new input size
      2: only resize
      :param index:
      :return:
      """
      img_id = self.images[index]
      file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
      img_path = os.path.join(self.img_dir, file_name)
      ann_ids = self.coco.getAnnIds(imgIds=[img_id])
      anns = self.coco.loadAnns(ids=ann_ids)
      num_objs = min(len(anns), self.max_objs)

      keypoints = []
      cls_ids = []
      for k in range(num_objs):
          ann = anns[k]
          keypoint = self._cocobox_to_center(ann['bbox'])
          keypoints.append(keypoint)
          cls_ids.append(int(self.cat_ids[ann['category_id']]))
          # cls_ids.append(0)           # if your dataset have only one lable

      img = cv2.imread(img_path)  # BGR
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

      if self.split == 'train':
         # Resolve loose keypoint when transform
         res = self.transform_train(image=img, keypoints=keypoints, class_labels=cls_ids)
         inp, resized_keypoints, resized_labels = res['image'], res['keypoints'], res['class_labels']

         if len(resized_keypoints) != len(keypoints):
             cls_ids = []
             for k in range(len(resized_keypoints)):
                 cls_ids.append(resized_labels[k])

      else:
         res = self.transform_test(image=img, keypoints=keypoints)
         inp, resized_keypoints = res['image'], res['keypoints']


      # Create heatmap
      res = self.transform_heatmap(image=inp, keypoints=resized_keypoints)
      heatmap_keypoints = res['keypoints']

      hm = np.zeros((self.num_classes, self.output_h, self.output_w), dtype=np.float32)
      reg = np.zeros((self.max_objs, 2), dtype=np.float32)
      ind = np.zeros((self.max_objs), dtype=np.int64)
      reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
      for k in range(len(heatmap_keypoints)):
          cls_id = cls_ids[k]
          ct = np.array(heatmap_keypoints[k])  # center
          ct_int = ct.astype(np.int32)  # center integer
          draw_umich_gaussian(hm[cls_id], ct_int, self.radius)
          ind[k] = ct_int[1] * self.output_w + ct_int[0]
          reg[k] = ct - ct_int
          reg_mask[k] = 1

      inp = inp.transpose(2, 0, 1)  # input image
      ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'reg': reg}

      return ret

  def __len__(self):
      return self.num_samples
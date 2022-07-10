# Implement Faster-RCNN Project
# Editor: Lee, Jongmo
# Filename: /test.py

# Reference URL
# https://arxiv.org/abs/1506.01497

import argparse
from ast import literal_eval
from configparser import ConfigParser
import cv2
import numpy as np
import tensorflow as tf
from custom import utils
from custom.models import BackBone
from custom.models import Proposal
from custom.models import Detection

config_file = 'config.ini'
products_file = 'products.ini'

def parser_config(config_file):
    """
    Set initial as global variables

    Arguments)
    1. config_file => 'str'
    - Filename with extension '*.ini'
    """

    config = ConfigParser()
    config.read(config_file)
    checkpoint_parser = config.get('path', 'checkpoint')
    image_input_size = literal_eval(config.get('variables', 'image_input_size'))
    number_of_proposals = config.getint('variables', 'number_of_proposals')
    anchor_side_per_feature = literal_eval(config.get('variables',
        'anchor_side_per_feature'))
    anchor_size_per_feature = list(map(utils.square, anchor_side_per_feature))
    anchor_aspect_ratio_per_feature = literal_eval(config.get('variables',
        'anchor_aspect_ratio_per_feature'))
    shorter_side_scale = config.getint('variables', 'shorter_side_scale')

    ckpt_config = ConfigParser()
    ckpt_config.read(checkpoint_parser)
    max_to_keep = ckpt_config.getint('variables', 'max_to_keep')
    backbone_dir = ckpt_config.get('network', 'backbone')
    proposal_dir = ckpt_config.get('network', 'proposal')
    detection_dir = ckpt_config.get('network', 'detection')

    products_config = ConfigParser()
    products_config.read(products_file)
    number_of_anchors_per_feature = products_config.getint('variables',
        'number_of_anchors_per_feature')
    number_of_features = products_config.getint('variables',
        'number_of_features')
    feature_map_size = literal_eval(products_config.get('variables',
        'feature_map_size'))
    pool_size = literal_eval(products_config.get('variables', 'pool_size'))
    number_of_objects = products_config.getint('variables', 'number_of_objects')
    object_list = literal_eval(products_config.get('variables', 'object_list'))
    object_color_list = literal_eval(products_config.get('variables',
        'object_color_list'))

    globals().update(locals())


def parse_argument():

    parser = argparse.ArgumentParser(description='Test Faster-R-CNN!!!')
    subparsers = parser.add_subparsers(dest='data_type',
        help='image, video or None(Webcam)')

    img = subparsers.add_parser('image')
    img.add_argument('--path', help='image path', required=True)

    vdo = subparsers.add_parser('video')
    vdo.add_argument('--path', help='video path', required=True)

    cam = subparsers.add_parser('webcam')

    return parser.parse_args()


def main():

    parser_config(config_file)
    external_arguments = parse_argument()

    backbone_model, *_ = utils.load_network_manager(BackBone, backbone_dir,
        max_to_keep, input_shape=image_input_size)
    proposal_model, *_ = utils.load_network_manager(Proposal, proposal_dir,
        max_to_keep, input_shape=feature_map_size,
        number_of_anchors_per_feature=number_of_anchors_per_feature)
    detection_model, *_ = utils.load_network_manager(Detection, detection_dir,
        max_to_keep, input_shape=feature_map_size, pool_size=pool_size,
        number_of_proposals=number_of_proposals,
        number_of_objects=number_of_objects)

    if (external_arguments.data_type == 'image'):
        image_path = external_arguments.path

        test_image(backbone_model, proposal_model, detection_model,
            image_input_size, feature_map_size, anchor_size_per_feature,
            anchor_aspect_ratio_per_feature, number_of_anchors_per_feature,
            shorter_side_scale, number_of_features, number_of_proposals,
            number_of_objects, object_list, object_color_list, image_path)

    elif (external_arguments.data_type == 'video'):
        video_path = external_arguments.path

        test_video(backbone_model, proposal_model, detection_model,
            image_input_size, feature_map_size, anchor_size_per_feature,
            anchor_aspect_ratio_per_feature, number_of_anchors_per_feature,
            shorter_side_scale, number_of_features, number_of_proposals,
            number_of_objects, object_list, object_color_list, video_path)


    elif (external_arguments.data_type == 'webcam'):

        test_video(backbone_model, proposal_model, detection_model,
            image_input_size, feature_map_size, anchor_size_per_feature,
            anchor_aspect_ratio_per_feature, number_of_anchors_per_feature,
            shorter_side_scale, number_of_features, number_of_proposals,
            number_of_objects, object_list, object_color_list)


if __name__ == '__main__':
    main()

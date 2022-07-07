# Implement Faster-RCNN Project
# Editor: Lee, Jongmo
# Filename: /custom/utils.py

# Reference URL
# https://www.tensorflow.org/guide/checkpoint
# https://www.tensorflow.org/guide/autodiff

import os
import math
import pickle
import random
from glob import glob
from time import time
from ast import literal_eval
from configparser import ConfigParser
from xml.etree.ElementTree import parse
import cv2
from tqdm import tqdm
import tensorflow as tf

number_of_cls, number_of_reg = 2, 4

def CRLF():
    """
    Line break
    """

    print()


def load_network_manager(model_class,
                         ckpt_path,
                         max_to_keep,
                         **kwargs):
    """
    Load objects to manage model

    Arguments)
    1. model_class => 'type'
    - Class of model defined in models.py

    2. ckpt_path => 'str'
    - Checkpoint path of model

    3. max_to_keep => 'int'
    - Max number of checkpoints to keep

    4. **kwargs
    - Arguments for init model_class

    Returns)
    1. model => 'custom.models.*'
    - Custom model for Faster-RCNN-object-detection

    2. ckpt => 'tensorflow.python.training.tracking.util.Checkpoint'
    - Model checkpoint

    3. ckpt_manager => 'tensorflow.python.training.checkpoint_management.
                        CheckpointManager'
    - Model checkpoint manager
    """

    model = model_class(**kwargs)

    if os.path.exists(ckpt_path):
        latest = tf.train.latest_checkpoint(ckpt_path)
        model.load_weights(latest)
        print("%s Restored from %s" % (model_class.__name__, latest))

    else:
        print("%s Initialized" % model_class.__name__)

    ckpt = tf.train.Checkpoint(model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep)

    return model, ckpt, ckpt_manager


def apply_weight_decay(model,
                       l2):
    """
    Apply weight decay option to model

    Arguments)
    1. model => 'custom.models.*'
    - Custom model to apply weight decay

    2. l2 => 'float'
    - Weight decay factor

    Returns)
    1. model => 'custom.models.*'
    - Custom model with weight decay applied
    """

    for layer in model.layers:

        if hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = tf.keras.regularizers.L2(l2)

    return model


def get_input_image_and_original_shape(image_path,
                                       image_input_size):
    """
    Get tensor array from image file

    Arguments)
    1. image_path => 'str'
    - Image file path

    2. image_input_size => 'tuple'
    - Fixed shape when input image to model

    Returns)
    1. image_tensor => 'tensorflow.python.framework.ops.EagerTensor'
    - Image tensor from which to extract feature maps

    2. image_shape => 'tuple'
    - Original shape(height, width, depth) of image
    """

    image_array = cv2.imread(image_path)
    image_shape = image_array.shape
    image_array = cv2.resize(image_array, image_input_size[:2]) / 255.
    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)

    return (image_tensor,
            image_shape)


def generate_anchors(image_shape,
                     image_input_size,
                     feature_map_size,
                     anchor_size_per_feature,
                     anchor_aspect_ratio_per_feature,
                     number_of_anchors_per_feature,
                     shorter_side_scale,
                     number_of_features):
    """
    Generate anchors projected on image_input_size from original image_shape

    Arguments)
    1. image_shape => 'tuple'
    - Original shape(height, width, depth) of image

    2. image_input_size => 'tuple'
    - Fixed shape when input image to model

    3. feature_map_size => 'tuple'
    - Extracted feature map shape from image by backbone

    4. anchor_size_per_feature => 'list'
    - Anchor area defined

    5. anchor_aspect_ratio_per_feature => 'list'
    - Anchor width-height ratio defined

    6. number_of_anchors_per_feature => 'int'
    - Number of anchors corresponding to a feature

    7. shorter_side_scale => 'int'
    - Fix shorter side scale when generate anchors

    8. number_of_features => 'int'
    - Number of features in a feature map

    Returns)
    1. anchors => 'tensorflow.python.framework.ops.EagerTensor'
    - Generated anchors in input image
    """

    if (image_shape[0] < image_shape[1]):
        height = float(shorter_side_scale)
        width = image_shape[1] / image_shape[0] * shorter_side_scale

    else:
        height = image_shape[0] / image_shape[1] * shorter_side_scale
        width = float(shorter_side_scale)

    stride_x = image_input_size[1] / feature_map_size[1]
    stride_y = image_input_size[0] / feature_map_size[0]

    centered_x = ((tf.cast(tf.range(feature_map_size[1]), dtype=tf.float32)
        + 0.5) * stride_x)
    centered_y = ((tf.cast(tf.range(feature_map_size[0]), dtype=tf.float32)
        + 0.5) * stride_y)

    mesh_x, mesh_y = tf.meshgrid(centered_x, centered_y)
    mesh_x = tf.reshape(mesh_x, (-1,))
    mesh_y = tf.reshape(mesh_y, (-1,))
    tile_coords = tf.stack([mesh_x, mesh_y], -1)

    coords = tf.reshape(tf.tile(tile_coords,
        [1, number_of_anchors_per_feature]),
        (number_of_features * number_of_anchors_per_feature, 2))

    tile_anchors_shape = []

    for area in anchor_size_per_feature:

        for ratio in anchor_aspect_ratio_per_feature:
            anchor_width = math.sqrt(area / ratio)
            anchor_height = anchor_width * ratio
            anchor_width *= image_input_size[1] / width
            anchor_height *= image_input_size[0] / height

            tile_anchors_shape.append([anchor_width, anchor_height])

    tile_anchors_shape = tf.convert_to_tensor(tile_anchors_shape,
        dtype=tf.float32)
    anchors_shape = tf.tile(tile_anchors_shape, [number_of_features, 1])
    anchors = tf.concat([coords, anchors_shape], -1)

    return anchors


def load_pascal_voc_faster_rcnn_dataset(images_path,
                                        metadata_path,
                                        anchor_size_per_feature,
                                        anchor_aspect_ratio_per_feature,
                                        number_of_anchors_per_feature,
                                        shorter_side_scale,
                                        image_input_size,
                                        feature_map_size,
                                        products_config):
    """
    Load PascalVOC2007 dataset for training Faster-RCNN object detection Model

    Arguments)
    1. images_path => 'str'
    - image file path in dataset

    2. metadata_path => 'str'
    - *.xml file path containing metadata

    3. anchor_size_per_feature => 'list'
    - Anchor area defined

    4. anchor_aspect_ratio_per_feature => 'list'
    - Anchor width-height ratio defined

    5. number_of_anchors_per_feature => 'int'
    - Number of anchors corresponding to a feature

    6. shorter_side_scale => 'int'
    - Fix shorter side scale when generate anchors

    7. image_input_size => 'tuple'
    - Fixed shape when input image to model

    8. feature_map_size => 'tuple'
    - Extracted feature map shape from image by backbone

    9. products_config => 'configparser.ConfigParser'
    - Parser that save products generated during processing

    Returns)
    1. images_list => 'list'
    - Images list parsed from dataset

    2. object_dict => 'dict'
    - Object labels information, {'background': 0, ...}

    3. anchors_list => 'list'
    - Anchors list generated from images list

    4. ground_truth_bboxes_list => 'list'
    - Ground truth bounding box list for all images list

    5. ground_truth_labels_list => 'list'
    - Ground truth object labels list for all images list

    6. number_of_objects => 'int'
    - Number of objects to predict

    7. products_config => 'configparser.ConfigParser'
    - Parser that save products generated during processing
    """

    def ground_truth_parser(root_tag,
                            object_dict,
                            object_label,
                            image_input_size,
                            image_shape):
        """
        Parse ground truth data from *.xml file

        Arguments)
        1. root_tag => 'xml.etree.ElementTree.Element'
        - Data(*.xml) include ground truth information

        2. object_dict => 'dict'
        - Object labels information, {'background': 0, ...}

        3. object_label => 'int'
        - Object label max number in current

        4. image_input_size => 'tuple'
        - Fixed shape when input image to model

        5. image_shape => 'tuple'
        - Original shape(height, width, depth) of image

        Returns)
        1. ground_truth_bboxes => 'tensorflow.python.framework.ops.EagerTensor'
        - Ground truth bounding boxes in an image

        2. ground_truth_labels => 'tensorflow.python.framework.ops.EagerTensor'
        - Ground truth labels in an image

        3. object_dict => 'dict'
        - Object labels information, {'background': 0, ...}
        """

        ground_truth_bboxes = []
        ground_truth_labels = []

        for object_tag in root_tag.findall('object'):
            object_name = object_tag.find('name').text

            if (object_dict.get(object_name) == None):
                object_dict[object_name] = object_label
                object_label += 1

            width_multiplier = image_input_size[1] / image_shape[1]
            height_multiplier = image_input_size[0] / image_shape[0]

            xmin_g = int(object_tag.find("bndbox/xmin").text) * width_multiplier
            ymin_g = int(object_tag.find("bndbox/ymin").text) * height_multiplier
            xmax_g = int(object_tag.find("bndbox/xmax").text) * width_multiplier
            ymax_g = int(object_tag.find("bndbox/ymax").text) * height_multiplier

            x_g = (xmin_g + xmax_g) / 2
            y_g = (ymin_g + ymax_g) / 2
            w_g = xmax_g - xmin_g
            h_g = ymax_g - ymin_g

            ground_truth_bboxes.append([x_g, y_g, w_g, h_g])
            ground_truth_labels.append(object_dict[object_name])

        ground_truth_bboxes = tf.convert_to_tensor(ground_truth_bboxes,
            dtype=tf.float32)
        ground_truth_labels = tf.convert_to_tensor(ground_truth_labels,
            dtype=tf.int32)

        return (ground_truth_bboxes,
                ground_truth_labels,
                object_dict)


    tqdm_bar_format = '{percentage:3.0f}%|{bar:30}{r_bar}'

    object_dict = {'background': 0}; object_label = 1

    images_list = []
    anchors_list = []
    ground_truth_bboxes_list = []
    ground_truth_labels_list = []

    print('Load Data...')

    metadataset = glob(metadata_path + '*.xml')
    number_of_features = feature_map_size[0] * feature_map_size[1]
    products_config['variables']['number_of_data'] = str(len(metadataset))
    products_config['variables']['number_of_features'] = str(number_of_features)

    for metadata_file in tqdm(metadataset, bar_format=tqdm_bar_format):
        tree = parse(metadata_file)
        root_tag = tree.getroot()

        image_path = images_path + root_tag.find('filename').text
        (image_tensor, image_shape) = get_input_image_and_original_shape(
            image_path, image_input_size)
        images_list.append(image_tensor)

        anchors = generate_anchors(image_shape, image_input_size,
            feature_map_size, anchor_size_per_feature,
            anchor_aspect_ratio_per_feature, number_of_anchors_per_feature,
            shorter_side_scale, number_of_features)

        anchors_list.append(anchors)

        (ground_truth_bboxes, ground_truth_labels, object_dict) = \
            ground_truth_parser(root_tag, object_dict, object_label,
                image_input_size, image_shape)

        ground_truth_bboxes_list.append(ground_truth_bboxes)
        ground_truth_labels_list.append(ground_truth_labels)

    CRLF()

    number_of_objects = len(object_dict) - 1 	# Omit backgound
    object_list = []

    for object_name in object_dict:
        object_list.append(object_name)

    products_config['variables']['object_list'] = str(object_list)
    products_config['variables']['number_of_objects'] = str(number_of_objects)

    return (images_list,
            object_dict,
            anchors_list,
            ground_truth_bboxes_list,
            ground_truth_labels_list,
            number_of_objects,
            products_config)


def load_train_dataset(products_file,
                       train_dataset_file):
    """
    Load data from file saved

    Arguments)
    1. products_file => 'str'
    - Product filename('products.ini')

    2. train_dataset_file => 'str'
    - Dataset filename('train_dataset.pickle')

    Returns)
    1. train_dataset => 'dict'
    - Dataset for models training

    2. number_of_objects => 'int'
    - Number of objects to predict

    3. products_config => 'configparser.ConfigParser'
    - Parser that save products generated during processing
    """

    products_config = ConfigParser()
    products_config.read(products_file)

    number_of_objects = products_config.getint('variables',
        'number_of_objects')

    with open(train_dataset_file, 'rb') as f:
        train_dataset = pickle.load(f)

    return (train_dataset,
            number_of_objects,
            products_config)


def make_cache(products_file,
               train_dataset_file,
               pool_size,
               images_path,
               metadata_path,
               anchor_size_per_feature,
               anchor_aspect_ratio_per_feature,
               number_of_anchors_per_feature,
               shorter_side_scale,
               image_input_size,
               feature_map_size):
    """
    Make cache file for loading time save

    Arguments)
    1. products_file => 'str'
    - Product filename('products.ini')

    2. train_dataset_file => 'str'
    - Dataset filename('train_dataset.pickle')

    3. pool_size => 'tuple'
    - Return size of pooled feature map after operate ROIPooling

    4. images_path => 'str'
    - image file path in dataset

    5. metadata_path => 'str'
    - *.xml file path containing metadata

    6. anchor_size_per_feature => 'list'
    - Anchor area defined

    7. anchor_aspect_ratio_per_feature => 'list'
    - Anchor width-height ratio defined

    8. number_of_anchors_per_feature => 'int'
    - Number of anchors corresponding to a feature

    9. shorter_side_scale => 'int'
    - Fix shorter side scale when generate anchors

    10. image_input_size => 'tuple'
    - Fixed shape when input image to model

    11. feature_map_size => 'tuple'
    - Extracted feature map shape from image by backbone

    Returns)
    1. train_dataset => 'dict'
    - Dataset for models training

    2. number_of_objects => 'int'
    - Number of objects to predict

    3. products_config => 'configparser.ConfigParser'
    - Parser that save products generated during processing
    """

    def save_products(products_config,
                      products_file):
        """
        Save variables producted

        Arguments)
        1. products_config => 'configparser.ConfigParser'
        - Parser that save products generated during processing

        2. products_file => 'str'
        - Product filename('products.ini')
        """

        with open(products_file, 'a') as f:
            products_config.write(f)


    def save_train_dataset(train_dataset,
                           train_dataset_file):
        """
        Save train dataset loaded

        Arguments)
        1. train_dataset => 'dict'
        - Dataset for models training

        2. train_dataset_file => 'str'
        - Dataset filename('train_dataset.pickle')
        """

        with open(train_dataset_file, 'wb') as f:
            pickle.dump(train_dataset, f, pickle.HIGHEST_PROTOCOL)


    if (os.path.isfile(products_file)):
        os.remove(products_file)

    if (os.path.isfile(train_dataset_file)):
        os.remove(train_dataset_file)

    products_config = ConfigParser()
    products_config['variables'] = {}
    products_config['variables']['number_of_anchors_per_feature'] = \
        str(number_of_anchors_per_feature)
    products_config['variables']['feature_map_size'] = str(feature_map_size)
    products_config['variables']['pool_size'] = str(pool_size)

    (images_list, object_dict, anchors_list, ground_truth_bboxes_list,
        ground_truth_labels_list, number_of_objects, products_config) = \
            load_pascal_voc_faster_rcnn_dataset(images_path, metadata_path,
                anchor_size_per_feature, anchor_aspect_ratio_per_feature,
                number_of_anchors_per_feature, shorter_side_scale,
                image_input_size, feature_map_size, products_config)

    train_dataset = {'images_list': images_list,
        'anchors_list': anchors_list,
        'ground_truth_bboxes_list': ground_truth_bboxes_list,
        'ground_truth_labels_list': ground_truth_labels_list}

    save_products(products_config, products_file)
    save_train_dataset(train_dataset, train_dataset_file)

    return (train_dataset,
            number_of_objects,
            products_config)


def shuffle_list(*args):
    """
    Apply the same shuffle to all list arguments

    Arguments)
    1. *args
    - Lists with same length

    Returns)
    2. zip(*l)
    - Shuffled lists
    """

    l = list(zip(*args))
    random.shuffle(l)

    return zip(*l)


def convert_centered_to_square_and_split(centered_bboxes):
    """
    Convert and split centered coordinates to square coordinates

    Arguments)
    1. centered_bboxes => 'tensorflow.python.framework.ops.EagerTensor'
    - Coordinates defined (x, y, w, h) , where x, y represent the centered

    Returns)
    1. bboxes_coordinates => 'tuple'
    - Returns the coordinates defined as a tensor in tuple.
    """

    x, y, w, h = tf.split(centered_bboxes, 4, -1)

    xmin = x - 0.5 * w
    ymin = y - 0.5 * h
    xmax = x + 0.5 * w
    ymax = y + 0.5 * h

    bboxes_coordinates = (xmin, ymin, xmax, ymax)

    return bboxes_coordinates


def convert_square_to_centered(square_bboxes):
	"""
	Convert square coordinates to centered coordinates

	Arguments)
	1. square_bboxes => 'tensorflow.python.framework.ops.EagerTensor'
	- Coordinates defined (xmin, ymin, xmax, ymax)

	Returns)
	1. centered_bboxes => 'tensorflow.python.framework.ops.EagerTensor'
	- Coordinates defined (x, y, w, h)
	"""

	xy_min, xy_max = tf.split(square_bboxes, 2, -1)
	xy = (xy_min + xy_max) / 2
	wh = xy_max - xy_min
	centered_bboxes = tf.concat([xy, wh], -1)

	return centered_bboxes


def get_iou_map(centered_bboxes_1,
                centered_bboxes_2):
    """
    Operate iou map from two centered bounding boxes (example: anchors and
    ground truth bboxes)

    !!Caution!! => Batch process operation not applied yet...

    Arguments)
    1. centered_bboxes_1 => 'tensorflow.python.framework.ops.EagerTensor'
    - Centered bounding boxes (Use current bounding boxes you have)

    2. centered_bboxes_2 => 'tensorflow.python.framework.ops.EagerTensor'
    - Centered bounding boxes (Use ground truth bounding boxes)

    Returns)
    1. iou_map => 'tensorflow.python.framework.ops.EagerTensor'
    - IoU map (matrix) between two bounding boxes
    """

    def intersection_area_map(square_bboxes_1,
                              square_bboxes_2):
        """
        Operate intersection area map from two square bounding boxes (example:
        anchors and ground truth bboxes)

        Arguments)
        1. square_bboxes_1 => 'tensorflow.python.framework.ops.EagerTensor'
        - Square bounding boxes (Use current bounding boxes you have)

        2. square_bboxes_2 => 'tensorflow.python.framework.ops.EagerTensor'
        - Square bounding boxes (Use ground truth bounding boxes)

        Returns)
        1. area_i => 'tensorflow.python.framework.ops.EagerTensor'
        - Intersection area map (matrix) between two bounding boxes
        """

        xmin_1, ymin_1, xmax_1, ymax_1 = square_bboxes_1
        xmin_2, ymin_2, xmax_2, ymax_2 = square_bboxes_2

        xmin_i = tf.maximum(xmin_1, tf.transpose(xmin_2, [0, 2, 1]))
        ymin_i = tf.maximum(ymin_1, tf.transpose(ymin_2, [0, 2, 1]))
        xmax_i = tf.minimum(xmax_1, tf.transpose(xmax_2, [0, 2, 1]))
        ymax_i = tf.minimum(ymax_1, tf.transpose(ymax_2, [0, 2, 1]))

        area_i = tf.maximum(xmax_i - xmin_i, 0) * tf.maximum(ymax_i - ymin_i, 0)

        return area_i


    def union_area_map(area_1,
                       area_2,
                       area_i):
        """
        Operate union area map from two square bounding boxes (example: anchors
        and ground truth bboxes)

        Arguments)
        1. area_1 => 'tensorflow.python.framework.ops.EagerTensor'
        - Bounding boxes (Current bounding boxes) area map

        2. area_2 => 'tensorflow.python.framework.ops.EagerTensor'
        - Bounding boxes (Ground truth bounding boxes) area map

        3. area_i => 'tensorflow.python.framework.ops.EagerTensor'
        - Intersection area map between two bounding boxes

        Returns)
        1. area_u => 'tensorflow.python.framework.ops.EagerTensor'
        - Union area map (matrix) between two bounding boxes
        """

        area_u =  (tf.expand_dims(area_1, -1) + tf.expand_dims(area_2, 1)
            - area_i)

        return area_u


    square_bboxes_1 = convert_centered_to_square_and_split(centered_bboxes_1)
    x_1, y_1, w_1, h_1 = tf.split(centered_bboxes_1, number_of_reg, -1)

    square_bboxes_2 = convert_centered_to_square_and_split(centered_bboxes_2)
    x_2, y_2, w_2, h_2 = tf.split(centered_bboxes_2, number_of_reg, -1)

    area_1 = tf.squeeze(w_1 * h_1, axis=-1)
    area_2 = tf.squeeze(w_2 * h_2, axis=-1)

    area_i = intersection_area_map(square_bboxes_1, square_bboxes_2)
    area_u = union_area_map(area_1, area_2, area_i)

    iou_map = area_i / area_u

    return iou_map


def centered_to_regression_labels(max_iou_ground_truth_bboxes,
                                  current_bboxes):
    """
    Encoding bounding boxes for operating regression loss

    Arguments)
    1. max_iou_ground_truth_bboxes => 'tensorflow.python.framework.ops.
                                       EagerTensor'
    - Label IoU max ground truth bounding boxes

    2. current_bboxes => 'tensorflow.python.framework.ops.EagerTensor'
    - Bounding boxes you have (example: anchors, decoded regression predict from
      region proposal network)

    Returns)
    1. reg_label => 'tensorflow.python.framework.ops.EagerTensor'
    - Regression values of bounding boxes
    """

    xy_g, wh_g = tf.split(max_iou_ground_truth_bboxes, 2, -1)
    xy_c, wh_c = tf.split(current_bboxes, 2, -1)
    xy_reg = (xy_g - xy_c) / wh_c
    wh_reg = tf.math.log(wh_g / wh_c)
    reg_label = tf.concat([xy_reg, wh_reg], 2)

    return reg_label


def regression_to_centered_predict(regression_predict,
                                   current_bboxes):
    """
    Decoding bounding boxes

    Arguments)
    1. regression_predict => 'tensorflow.python.framework.ops.EagerTensor'
    - Regression bounding boxes from model (example: region proposal network)

    2. current_bboxes => 'tensorflow.python.framework.ops.EagerTensor'
    - Bounding boxes you have (example: anchors, decoded regression predict from
      region proposal network)

    Returns)
    1. centered_bboxes_predict => 'tensorflow.python.framework.ops.EagerTensor'
	- Centered coordinates bounding boxes
    """

    xy_offset, wh_offset = tf.split(regression_predict, 2, -1)
    xy_c, wh_c = tf.split(current_bboxes, 2, -1)
    xy_p = xy_offset * wh_c + xy_c
    wh_p = tf.math.exp(wh_offset) * wh_c
    centered_bboxes_predict = tf.concat([xy_p, wh_p], 2)

    return centered_bboxes_predict


def print_step_message(step,
                       steps,
                       step_start,
                       loss,
                       progress_bar_length):
    """
    Print message during a step

    Arguments)
    1. step => 'int'
    - Current step in epoch

    2. steps => 'int'
    - Total step in epoch

    3. step_start => 'float'
    - Step start time

    4. loss => 'tensorflow.python.framework.ops.EagerTensor'
    - Loss value calculated

    5. progress_bar_length => 'int'
    - Length when Visualize progress bar
    """

    terminal_width, _ = os.get_terminal_size()
    progress = int(progress_bar_length * (step + 1) / steps)
    progress_bar = (progress * "●").ljust(progress_bar_length, "○")
    step_runtime = time() - step_start
    step_message = \
        "\rtrain: %4d/%4d[%s] - multi_task_loss: %.4e - time: %.4fsec" \
        % (step + 1, steps, progress_bar, loss, step_runtime)
    print(step_message.ljust(terminal_width, " "), end="")


def print_epoch_message(epoch_start,
                        train_loss_list,
                        valid_loss_list):
    """
    Print message during an epoch

    Arguments)
    1. epoch_start => 'float'
    - Epoch start time

    2. train_loss_list => 'list'
    - List of train loss values

    3. valid_loss_list => 'list'
    - List of validation loss values
    """

    terminal_width, _ = os.get_terminal_size()
    epoch_runtime = time() - epoch_start
    epoch_message = \
        "\rtrain_loss_mean: %.4e - valid_loss_mean: %.4e - time: %.4fsec" \
        % (tf.reduce_mean(train_loss_list), tf.reduce_mean(valid_loss_list),
        epoch_runtime)
    print(epoch_message.ljust(terminal_width, " "))


def save_weights(epoch,
                 save_cycle,
                 *ckpt_managers):
    """
    Save model's weights per cycle epoch

    Arguments)
    1. epoch => 'int'
    - Current epoch

    2. save_cycle => 'int'
    - Cycle that save model

    3. *ckpt_managers
    - Checkpoint managers of models
    """

    if ((epoch % save_cycle) == (save_cycle - 1)):
        print("Weights Saved!!")

        for ckpt_manager in ckpt_managers:
            ckpt_manager.save()


def train_rpn(train_dataset,
              products_config,
              epochs,
              train_valid_split_rate,
              valid_steps_max,
              backbone_model,
              proposal_model,
              backbone_trainable,
              sparse_categorical_cross_entropy,
              huber,
              rpn_loss_lambda,
              stochastic_gradient_descent,
              number_of_sampled_region,
              progress_bar_length,
              save_cycle,
              *ckpt_managers):
    """
    Train region proposal network

    Arguments)
    1. train_dataset => 'dict'
    - Dataset for models training

    2. products_config => 'configparser.ConfigParser'
    - Parser that save products generated during processing

    3. epochs => 'int'
    - Total epochs when training

    4. train_valid_split_rate => 'float'
    - Split rate between train and validation of dataset when training

    5. valid_steps_max => 'int'
    - Max of validation dataset when validate during training

    6. backbone_model => 'custom.models.BackBone'
    - Feature extract model for image

    7. proposal_model => 'custom.models.Proposal'
    - Region proposal using feature map

    8. backbone_trainable => 'bool'
    - Whether to train the backbone network

    9. sparse_categorical_cross_entropy => 'keras.losses.
                                            SparseCategoricalCrossentropy'
    - Classification loss function

    10. huber => 'keras.losses.Huber'
    - Regression loss function

    11. rpn_loss_lambda => 'int'
    - Balancing parameter when calculate multi task loss between classification
      loss and regression loss

    12. stochastic_gradient_descent => 'keras.optimizer_v2.gradient_descent.SGD'
    - Optimize function

    13. number_of_sampled_region => 'int'
    - Total sampling number of region for classification training

    14. progress_bar_length => 'int'
    - Length when Visualize progress bar

    15. save_cycle => 'int'
    - Cycle that save model

    16. *ckpt_managers
    - Checkpoint managers of models
    """

    def rpn_data_generator(images_list,
                           anchors_list,
                           ground_truth_bboxes_list,
                           number_of_anchors_per_feature,
                           number_of_features,
                           number_of_sampled_region):
        """
        Data generator for region proposal training

        Arguments)
        1. images_list => 'list'
        - Images list parsed from dataset

        2. anchors_list => 'list'
        - Anchors list generated from images list

        3. ground_truth_bboxes_list => 'list'
        - Ground truth bounding box list for all images list

        4. number_of_anchors_per_feature => 'int'
        - Number of anchors corresponding to a feature

        5. number_of_features => 'int'
        - Number of features in a feature map

        6. number_of_sampled_region => 'int'
        - Total sampling number of region for classification training

        Yields)
        x.1. image => 'tensorflow.python.framework.ops.EagerTensor'
        - Image to use for train step

        y.1. sample_indices => 'tensorflow.python.framework.ops.EagerTensor'
        - Indices for sampling data

        y.2. objectness => 'tensorflow.python.framework.ops.EagerTensor'
        - Presence or not of an object

        y.3. reg_label => 'tensorflow.python.framework.ops.EagerTensor'
        - Ground truth bounding boxes regression values for calculating
          regression loss
        """

        (images_list, anchors_list, ground_truth_bboxes_list) = \
            shuffle_list(images_list, anchors_list, ground_truth_bboxes_list)

        for data in zip(images_list, anchors_list, ground_truth_bboxes_list):
            image = tf.expand_dims(data[0], 0)
            anchors = tf.expand_dims(data[1], 0)
            ground_truth_bboxes = tf.expand_dims(data[2], 0)

            iou_map = get_iou_map(anchors, ground_truth_bboxes)
            max_iou_map = tf.reduce_max(iou_map, axis=-1)
            max_iou_indices = tf.argmax(iou_map, axis=-1)

            ground_truth_bboxes_close_to_anchors = tf.gather(tf.squeeze(
                ground_truth_bboxes, axis=0), max_iou_indices)

            max_of_max_iou_map = tf.reshape(tf.tile(tf.expand_dims(tf.reduce_max(
                tf.reshape(max_iou_map, [number_of_features,
                number_of_anchors_per_feature]), -1), -1), [1,
                number_of_anchors_per_feature]), [1, (number_of_features
                * number_of_anchors_per_feature)])

            pos_mask = tf.logical_or(tf.greater(max_iou_map, 0.7), tf.logical_and(
                tf.equal(max_iou_map, max_of_max_iou_map), tf.greater_equal(
                max_iou_map, 0.3)))
            pos_num = tf.minimum(tf.math.count_nonzero(pos_mask), int(
                number_of_sampled_region * 0.5))
            pos_indices = tf.reshape(tf.random.shuffle(tf.where(tf.squeeze(
                pos_mask)))[:pos_num], [1, pos_num])

            neg_mask = tf.less(max_iou_map, 0.3)
            neg_num = number_of_sampled_region - pos_num
            neg_indices = tf.reshape(tf.random.shuffle(tf.where(tf.squeeze(
                neg_mask)))[:neg_num], [1, neg_num])

            sample_indices = tf.expand_dims(tf.random.shuffle(tf.squeeze(tf.concat(
                [pos_indices, neg_indices], axis=1))), 0)

            objectness = tf.cast(pos_mask, tf.float32)

            reg_label = centered_to_regression_labels(
                ground_truth_bboxes_close_to_anchors, anchors)

            yield (image), (sample_indices, objectness, reg_label)


    def rpn_multi_task_loss(backbone_model,
                            proposal_model,
                            image,
                            sample_indices,
                            objectness,
                            reg_label,
                            sparse_categorical_cross_entropy,
                            huber,
                            rpn_loss_lambda,
                            training):
        """
        Operate multi task loss for region proposal network

        Arguments)
        1. backbone_model => 'custom.models.BackBone'
        - Feature extract model for image

        2. proposal_model => 'custom.models.Proposal'
        - Region proposal using feature map

        3. image => 'tensorflow.python.framework.ops.EagerTensor'
        - Image for extracting feature

        4. sample_indices => 'tensorflow.python.framework.ops.EagerTensor'
        - Indices for sampling data

        5. objectness => 'tensorflow.python.framework.ops.EagerTensor'
        - Presence or not of an object

        6. reg_label => 'tensorflow.python.framework.ops.EagerTensor'
        - Ground truth bounding boxes regression values for calculating
          regression loss

        7. sparse_categorical_cross_entropy => 'keras.losses.
                                                SparseCategoricalCrossentropy'
        - Classification loss function

        8. huber => 'keras.losses.Huber'
        - Regression loss function

        9. rpn_loss_lambda => 'int'
        - Balancing parameter when calculate multi task loss between classification
          loss and regression loss

        10. training => 'bool'
        - Train model or not

        Returns)
        1. multi_task_loss => 'tensorflow.python.framework.ops.EagerTensor'
        - Multi task loss define as (cls-loss + lambda * reg-loss)
        """

        feature_map = backbone_model(image, training=training)
        cls_pred, reg_pred = proposal_model(feature_map, training=training)
        sampled_cls_pred = tf.gather(tf.squeeze(cls_pred, axis=0),
            sample_indices)
        sampled_cls_label = tf.gather(tf.squeeze(objectness, axis=0),
            sample_indices)

        cls_loss = sparse_categorical_cross_entropy(sampled_cls_label,
            sampled_cls_pred)
        reg_loss = objectness * huber(reg_label, reg_pred)

        multi_task_loss = (tf.reduce_mean(cls_loss)
            + rpn_loss_lambda * tf.reduce_mean(reg_loss))

        return multi_task_loss


    def train_step_rpn(backbone_model,
                       proposal_model,
                       image,
                       sample_indices,
                       objectness,
                       reg_label,
                       sparse_categorical_cross_entropy,
                       huber,
                       rpn_loss_lambda,
                       stochastic_gradient_descent):
        """
        Calculate loss and update weights of region proposal network

        Arguments)
        1. backbone_model => 'custom.models.BackBone'
        - Feature extract model for image

        2. proposal_model => 'custom.models.Proposal'
        - Region proposal using feature map

        3. image => 'tensorflow.python.framework.ops.EagerTensor'
        - Image for extracting feature

        4. sample_indices => 'tensorflow.python.framework.ops.EagerTensor'
        - Indices for sampling data

        5. objectness => 'tensorflow.python.framework.ops.EagerTensor'
        - Presence or not of an object

        6. reg_label => 'tensorflow.python.framework.ops.EagerTensor'
        - Ground truth bounding boxes regression values for calculating
          regression loss

        7. sparse_categorical_cross_entropy => 'keras.losses.
                                                SparseCategoricalCrossentropy'
        - Classification loss function

        8. huber => 'keras.losses.Huber'
        - Regression loss function

        9. rpn_loss_lambda => 'int'
        - Balancing parameter when calculate multi task loss between
          classification loss and regression loss

        10. stochastic_gradient_descent => 'keras.optimizer_v2.gradient_descent.
                                            SGD'
        - Optimize function

        Returns)
        1. multi_task_loss => 'tensorflow.python.framework.ops.EagerTensor'
        - Multi task loss define as (cls-loss + lambda * reg-loss)
        """

        with tf.GradientTape() as tape:
            tape.watch(image)
            multi_task_loss = rpn_multi_task_loss(backbone_model,
                proposal_model, image, sample_indices, objectness, reg_label,
                sparse_categorical_cross_entropy, huber, rpn_loss_lambda, True)

        rpn_trainable_weights = (backbone_model.trainable_weights
            + proposal_model.trainable_weights)
        gradient_of_multi_task_loss = tape.gradient(multi_task_loss,
            rpn_trainable_weights)
        stochastic_gradient_descent.apply_gradients(
            zip(gradient_of_multi_task_loss, rpn_trainable_weights))

        return multi_task_loss


    backbone_model.trainable = backbone_trainable
    proposal_model.trainable = True

    number_of_anchors_per_feature = products_config.getint('variables',
        'number_of_anchors_per_feature')
    number_of_features = products_config.getint('variables',
        'number_of_features')
    number_of_data = products_config.getint('variables', 'number_of_data')

    train_steps = int(number_of_data * train_valid_split_rate)
    valid_steps = valid_steps_max
    remain_steps = number_of_data - train_steps

    images_list = train_dataset['images_list']
    anchors_list = train_dataset['anchors_list']
    ground_truth_bboxes_list = train_dataset['ground_truth_bboxes_list']
    ground_truth_labels_list = train_dataset['ground_truth_labels_list']

    if (remain_steps < valid_steps):
        valid_steps = remain_steps

    for epoch in range(epochs):
        print("Epoch: %3d/%3d" % (epoch + 1, epochs))
        epoch_start = time()
        train_loss_list = []
        valid_loss_list = []
        rpn_dataset = rpn_data_generator(images_list, anchors_list,
            ground_truth_bboxes_list, number_of_anchors_per_feature,
            number_of_features, number_of_sampled_region)

        for step in range(train_steps):
            step_start = time()
            (image), (sample_indices, objectness, reg_label) = next(rpn_dataset)

            multi_task_loss = train_step_rpn(backbone_model, proposal_model,
                image, sample_indices, objectness, reg_label,
                sparse_categorical_cross_entropy, huber, rpn_loss_lambda,
                stochastic_gradient_descent)

            train_loss_list.append(multi_task_loss)

            print_step_message(step, train_steps, step_start, multi_task_loss,
                progress_bar_length)

        for step in range(valid_steps):
            step_start = time()
            (image), (sample_indices, objectness, reg_label) = next(rpn_dataset)

            multi_task_loss = rpn_multi_task_loss(backbone_model,
                proposal_model, image, sample_indices, objectness, reg_label,
                sparse_categorical_cross_entropy, huber, rpn_loss_lambda, False)

            valid_loss_list.append(multi_task_loss)

            print_step_message(step, valid_steps, step_start, multi_task_loss,
                progress_bar_length)

        print_epoch_message(epoch_start, train_loss_list, valid_loss_list)
        save_weights(epoch, save_cycle, *ckpt_managers)


def train_fast_rcnn(train_dataset,
                    products_config,
                    epochs,
                    train_valid_split_rate,
                    valid_steps_max,
                    backbone_model,
                    proposal_model,
                    detection_model,
                    backbone_trainable,
                    sparse_categorical_cross_entropy,
                    huber,
                    fast_rcnn_loss_lambda,
                    stochastic_gradient_descent,
                    image_input_size,
                    number_of_proposals,
                    progress_bar_length,
                    save_cycle,
                    *ckpt_managers):
    """
    Train fast rcnn

    Arguments)
    1. train_dataset => 'dict'
    - Dataset for models training

    2. products_config => 'configparser.ConfigParser'
    - Parser that save products generated during processing

    3. epochs => 'int'
    - Total epochs when training

    4. train_valid_split_rate => 'float'
    - Split rate between train and validation of dataset when training

    5. valid_steps_max => 'int'
    - Max of validation dataset when validate during training

    6. backbone_model => 'custom.models.BackBone'
    - Feature extract model for image

    7. proposal_model => 'custom.models.Proposal'
    - Region proposal using feature map

    8. detection_model => 'custom.models.Detection'
    - Detection model

    9. backbone_trainable => 'bool'
    - Whether to train the backbone network

    10. sparse_categorical_cross_entropy => 'keras.losses.
                                             SparseCategoricalCrossentropy'
    - Classification loss function

    11. huber => 'keras.losses.Huber'
    - Regression loss function

    12. fast_rcnn_loss_lambda => 'int'
    - Balancing parameter when calculate multi task loss between classification
      loss and regression loss

    13. stochastic_gradient_descent => 'keras.optimizer_v2.gradient_descent.SGD'
    - Optimize function

    14. image_input_size => 'tuple'
    - Fixed shape when input image to model

    15. number_of_proposals => 'int'
    - Number of output in fast rcnn

    16. progress_bar_length => 'int'
    - Length when Visualize progress bar

    17. save_cycle => 'int'
    - Cycle that save model

    18. *ckpt_managers
    - Checkpoint managers of models
    """

    def fast_rcnn_data_generator(images_list,
                                 anchors_list,
                                 ground_truth_bboxes_list,
                                 ground_truth_labels_list):
        """
        Data generator for fast rcnn training

        Arguments)
        1. images_list => 'list'
        - Images list parsed from dataset

        2. anchors_list => 'list'
        - Anchors list generated from images list

        3. ground_truth_bboxes_list => 'list'
        - Ground truth bounding box list for all images list

        4. ground_truth_labels_list => 'list'
        - Ground truth object labels list for all images list

        Yields)
        x.1. image => 'tensorflow.python.framework.ops.EagerTensor'
        - Image to use for train step

        x.2. anchors => 'tensorflow.python.framework.ops.EagerTensor'
        - Generated anchors in input image

        y.1. ground_truth_bboxes => 'tensorflow.python.framework.ops.EagerTensor'
        - Ground truth bounding boxes in an image

        y.2. ground_truth_labels => 'tensorflow.python.framework.ops.EagerTensor'
        - Ground truth labels in an image
        """

        (images_list, anchors_list, ground_truth_bboxes_list,
            ground_truth_labels_list) = shuffle_list(images_list,
                anchors_list, ground_truth_bboxes_list, ground_truth_labels_list)

        for data in zip(images_list, anchors_list, ground_truth_bboxes_list,
                        ground_truth_labels_list):
            image = tf.expand_dims(data[0], 0)
            anchors = tf.expand_dims(data[1], 0)
            ground_truth_bboxes = tf.expand_dims(data[2], 0)
            ground_truth_labels = tf.expand_dims(data[3], 0)

            yield (image, anchors), (ground_truth_bboxes, ground_truth_labels)


    def train_fast_rcnn_data_process(reg_pred,
                                     anchors,
                                     ground_truth_bboxes,
                                     ground_truth_labels,
                                     image_input_size,
                                     feature_map_size,
                                     number_of_proposals,
                                     number_of_objects):
        """
        Get variables for calculating loss
        Apply stop gradient for save memory and prevent useless calculations

        Arguments)
        1. reg_pred => 'tensorflow.python.framework.ops.EagerTensor'
        - Regression bounding boxes from model

        2. anchors => 'tensorflow.python.framework.ops.EagerTensor'
        - Generated anchors in input image

        3. ground_truth_bboxes => 'tensorflow.python.framework.ops.EagerTensor'
        - Ground truth bounding boxes in an image

        4. ground_truth_labels => 'tensorflow.python.framework.ops.EagerTensor'
        - Ground truth labels in an image

        5. image_input_size => 'tuple'
        - Fixed shape when input image to model

        6. feature_map_size => 'tuple'
        - Extracted feature map shape from image by backbone

        7. number_of_proposals => 'int'
        - Number of output in fast rcnn

        8. number_of_objects => 'int'
        - Number of objects to predict

        Returns)
        1. rois => 'tensorflow.python.framework.ops.EagerTensor'
        - Region of intersection on feature map for roi pooling

        2. fast_rcnn_objectness => 'tensorflow.python.framework.ops.EagerTensor'
        - Presence or not of an object

        3. fast_rcnn_cls_labels => 'tensorflow.python.framework.ops.EagerTensor'
        - Ground truth bounding boxes classification values for calculating
          classification loss

        4. fast_rcnn_reg_labels => 'tensorflow.python.framework.ops.EagerTensor'
        - Ground truth bounding boxes regression values for calculating
          regression loss
        """

        height, width, _ = image_input_size
        f_height, f_width, _ = feature_map_size
        projection_constant = tf.constant([[[f_width / width, f_height / height,
            f_width / width, f_height / height]]])

        centered_bboxes_pred = regression_to_centered_predict(reg_pred, anchors)
        square_bboxes_pred = tf.concat(convert_centered_to_square_and_split(
            centered_bboxes_pred), -1)

        mask = tf.reduce_all(tf.concat([tf.greater_equal(square_bboxes_pred, 0.),
            tf.expand_dims(tf.less(square_bboxes_pred[:, :, 2],
                float(width)), -1),
            tf.expand_dims(tf.less(square_bboxes_pred[:, :, 3],
                float(height)), -1),
            tf.expand_dims(tf.less(square_bboxes_pred[:, :, 0],
                square_bboxes_pred[:, :, 2]), -1),
            tf.expand_dims(tf.less(square_bboxes_pred[:, :, 1],
                square_bboxes_pred[:, :, 3]), -1)], -1), -1)

        filtered_centered_bboxes_pred = tf.expand_dims(tf.boolean_mask(
            centered_bboxes_pred, mask), 0)
        filtered_squares_bboxes_pred = tf.expand_dims(tf.boolean_mask(
            square_bboxes_pred, mask), 0)

        iou_map = get_iou_map(filtered_centered_bboxes_pred,
            ground_truth_bboxes)
        max_iou_indices = tf.argmax(iou_map, axis=-1)
        max_iou_map = tf.reduce_max(iou_map, axis=-1)

        ground_truth_bboxes_close_to_filtered_bboxes_pred = tf.gather(tf.squeeze(
            ground_truth_bboxes, axis=0), max_iou_indices)
        ground_truth_labels_close_to_filtered_bboxes_pred = tf.gather(tf.squeeze(
            ground_truth_labels, axis=0), max_iou_indices)

        pos_mask = tf.greater_equal(max_iou_map, 0.5)
        pos_num = tf.minimum(tf.math.count_nonzero(pos_mask),
            int(number_of_proposals * 0.25))
        pos_indices = tf.reshape(tf.random.shuffle(tf.where(tf.squeeze(
            pos_mask)))[:pos_num], [1, pos_num])

        neg_mask = tf.less(max_iou_map, 0.5)
        neg_num = number_of_proposals - pos_num
        neg_indices = tf.reshape(tf.random.shuffle(tf.where(tf.squeeze(
            neg_mask)))[:neg_num], [1, neg_num])

        sample_indices = tf.expand_dims(tf.random.shuffle(tf.squeeze(tf.concat(
            [pos_indices, neg_indices], axis=-1))), 0)

        rois_temp = tf.multiply(filtered_squares_bboxes_pred, projection_constant)
        rois = tf.gather(tf.squeeze(rois_temp), sample_indices)

        fast_rcnn_objectness_temp = tf.cast(pos_mask, tf.float32)
        fast_rcnn_objectness = tf.gather(tf.squeeze(fast_rcnn_objectness_temp),
            sample_indices)

        fast_rcnn_cls_labels_temp = tf.where(pos_mask,
            ground_truth_labels_close_to_filtered_bboxes_pred,
            tf.zeros_like(ground_truth_labels_close_to_filtered_bboxes_pred))
        fast_rcnn_cls_labels = tf.gather(tf.squeeze(fast_rcnn_cls_labels_temp),
            sample_indices)

        fast_rcnn_reg_labels_tile = tf.gather(tf.squeeze(
            centered_to_regression_labels(
                ground_truth_bboxes_close_to_filtered_bboxes_pred,
                filtered_centered_bboxes_pred)), sample_indices)
        fast_rcnn_reg_labels = tf.tile(fast_rcnn_reg_labels_tile,
            [1, 1, number_of_objects])

        return (tf.stop_gradient(rois),
                tf.stop_gradient(fast_rcnn_objectness),
                tf.stop_gradient(fast_rcnn_cls_labels),
                tf.stop_gradient(fast_rcnn_reg_labels))


    def fast_rcnn_multi_task_loss(backbone_model,
                                  proposal_model,
                                  detection_model,
                                  image,
                                  anchors,
                                  ground_truth_bboxes,
                                  ground_truth_labels,
                                  image_input_size,
                                  feature_map_size,
                                  number_of_proposals,
                                  number_of_objects,
                                  sparse_categorical_cross_entropy,
                                  huber,
                                  fast_rcnn_loss_lambda,
                                  training):
        """
        Operate multi task loss for fast rcnn

        Arguments)
        1. backbone_model => 'custom.models.BackBone'
        - Feature extract model for image

        2. proposal_model => 'custom.models.Proposal'
        - Region proposal using feature map

        3. detection_model => 'custom.models.Detection'
        - Detection model

        4. image => 'tensorflow.python.framework.ops.EagerTensor'
        - Image to use for train step

        5. anchors => 'tensorflow.python.framework.ops.EagerTensor'
        - Generated anchors in input image

        6. ground_truth_bboxes => 'tensorflow.python.framework.ops.EagerTensor'
        - Ground truth bounding boxes in an image

        7. ground_truth_labels => 'tensorflow.python.framework.ops.EagerTensor'
        - Ground truth labels in an image

        8. image_input_size => 'tuple'
        - Fixed shape when input image to model

        9. feature_map_size => 'tuple'
        - Extracted feature map shape from image by backbone

        10. number_of_proposals => 'int'
        - Number of output in fast rcnn

        11. number_of_objects => 'int'
        - Number of objects to predict

        12. sparse_categorical_cross_entropy => 'keras.losses.
                                                 SparseCategoricalCrossentropy'
        - Classification loss function

        13. huber => 'keras.losses.Huber'
        - Regression loss function

        14. fast_rcnn_loss_lambda => 'int'
        - Balancing parameter when calculate multi task loss between classification
          loss and regression loss

        15. training => 'bool'
        - Train model or not

        Returns)
        1. multi_task_loss => 'tensorflow.python.framework.ops.EagerTensor'
        - Multi task loss define as (cls-loss + lambda * reg-loss)
        """

        feature_map = backbone_model(image, training=training)
        _, reg_pred = proposal_model(feature_map, training=False)

        (rois, fast_rcnn_objectness, fast_rcnn_cls_labels,
            fast_rcnn_reg_labels) = train_fast_rcnn_data_process(reg_pred,
                anchors, ground_truth_bboxes, ground_truth_labels,
                image_input_size, feature_map_size, number_of_proposals,
                number_of_objects)

        fast_rcnn_cls_logits_pred, fast_rcnn_reg_pred = detection_model(
            [feature_map, rois], training=training)

        fast_rcnn_cls_loss = sparse_categorical_cross_entropy(
            fast_rcnn_cls_labels, fast_rcnn_cls_logits_pred)
        fast_rcnn_reg_loss = fast_rcnn_objectness * huber(
            fast_rcnn_reg_labels, fast_rcnn_reg_pred)
        multi_task_loss = (tf.reduce_mean(fast_rcnn_cls_loss)
            + fast_rcnn_loss_lambda
            * tf.reduce_mean(fast_rcnn_reg_loss))

        return multi_task_loss


    def train_step_fast_rcnn(backbone_model,
                             proposal_model,
                             detection_model,
                             image,
                             anchors,
                             ground_truth_bboxes,
                             ground_truth_labels,
                             image_input_size,
                             feature_map_size,
                             number_of_proposals,
                             number_of_objects,
                             sparse_categorical_cross_entropy,
                             huber,
                             fast_rcnn_loss_lambda,
                             stochastic_gradient_descent):
        """
        Calculate loss and update weights of fast rcnn

        Arguments)
        1. backbone_model => 'custom.models.BackBone'
        - Feature extract model for image

        2. proposal_model => 'custom.models.Proposal'
        - Region proposal using feature map

        3. detection_model => 'custom.models.Detection'
        - Detection model

        4. image => 'tensorflow.python.framework.ops.EagerTensor'
        - Image to use for train step

        5. anchors => 'tensorflow.python.framework.ops.EagerTensor'
        - Generated anchors in input image

        6. ground_truth_bboxes => 'tensorflow.python.framework.ops.EagerTensor'
        - Ground truth bounding boxes in an image

        7. ground_truth_labels => 'tensorflow.python.framework.ops.EagerTensor'
        - Ground truth labels in an image

        8. image_input_size => 'tuple'
        - Fixed shape when input image to model

        9. feature_map_size => 'tuple'
        - Extracted feature map shape from image by backbone

        10. number_of_proposals => 'int'
        - Number of output in fast rcnn

        11. number_of_objects => 'int'
        - Number of objects to predict

        12. sparse_categorical_cross_entropy => 'keras.losses.
                                                 SparseCategoricalCrossentropy'
        - Classification loss function

        13. huber => 'keras.losses.Huber'
        - Regression loss function

        14. fast_rcnn_loss_lambda => 'int'
        - Balancing parameter when calculate multi task loss between
          classification loss and regression loss

        15. stochastic_gradient_descent => 'keras.optimizer_v2.gradient_descent.
                                            SGD'
        - Optimize function

        Returns)
        1. multi_task_loss => 'tensorflow.python.framework.ops.EagerTensor'
        - Multi task loss define as (cls-loss + lambda * reg-loss)
        """

        with tf.GradientTape() as tape:
            tape.watch(image)
            multi_task_loss = fast_rcnn_multi_task_loss(backbone_model,
                proposal_model, detection_model, image, anchors,
                ground_truth_bboxes, ground_truth_labels, image_input_size,
                feature_map_size, number_of_proposals, number_of_objects,
                sparse_categorical_cross_entropy, huber, fast_rcnn_loss_lambda,
                True)

        fast_rcnn_trainable_weights = (backbone_model.trainable_weights
            + detection_model.trainable_weights)
        gradient_of_multi_task_loss = tape.gradient(multi_task_loss,
            fast_rcnn_trainable_weights)
        stochastic_gradient_descent.apply_gradients(
            zip(gradient_of_multi_task_loss, fast_rcnn_trainable_weights))

        return multi_task_loss


    backbone_model.trainable = backbone_trainable
    proposal_model.trainable = False
    detection_model.trainable = True

    number_of_data = products_config.getint('variables', 'number_of_data')
    number_of_objects = products_config.getint('variables', 'number_of_objects')
    feature_map_size = literal_eval(products_config.get('variables',
        'feature_map_size'))
    train_steps = int(number_of_data * train_valid_split_rate)
    valid_steps = valid_steps_max
    remain_steps = number_of_data - train_steps

    images_list = train_dataset['images_list']
    anchors_list = train_dataset['anchors_list']
    ground_truth_bboxes_list = train_dataset['ground_truth_bboxes_list']
    ground_truth_labels_list = train_dataset['ground_truth_labels_list']

    if (remain_steps < valid_steps):
        valid_steps = remain_steps

    for epoch in range(epochs):
        print("Epoch: %3d/%3d" % (epoch + 1, epochs))
        epoch_start = time()
        train_loss_list = []
        valid_loss_list = []
        fast_rcnn_dataset = fast_rcnn_data_generator(images_list, anchors_list,
            ground_truth_bboxes_list, ground_truth_labels_list)

        for step in range(train_steps):
            step_start = time()
            (image, anchors), (ground_truth_bboxes, ground_truth_labels) = \
                next(fast_rcnn_dataset)

            multi_task_loss = train_step_fast_rcnn(backbone_model,
                proposal_model, detection_model, image, anchors,
                ground_truth_bboxes, ground_truth_labels, image_input_size,
                feature_map_size, number_of_proposals, number_of_objects,
                sparse_categorical_cross_entropy, huber, fast_rcnn_loss_lambda,
                stochastic_gradient_descent)

            train_loss_list.append(multi_task_loss)

            print_step_message(step, train_steps, step_start, multi_task_loss,
                progress_bar_length)

        for step in range(valid_steps):
            step_start = time()
            (image, anchors), (ground_truth_bboxes, ground_truth_labels) = \
                next(fast_rcnn_dataset)

            multi_task_loss = fast_rcnn_multi_task_loss(backbone_model,
                proposal_model, detection_model, image, anchors,
                ground_truth_bboxes, ground_truth_labels, image_input_size,
                feature_map_size, number_of_proposals, number_of_objects,
                sparse_categorical_cross_entropy, huber, fast_rcnn_loss_lambda,
                False)

            valid_loss_list.append(multi_task_loss)

            print_step_message(step, valid_steps, step_start, multi_task_loss,
                progress_bar_length)

        print_epoch_message(epoch_start, train_loss_list, valid_loss_list)
        save_weights(epoch, save_cycle, *ckpt_managers)

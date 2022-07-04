# Implement Faster-RCNN Project
# Editor: Lee, Jongmo
# Filename: /custom/utils.py

# Reference URL
# https://www.tensorflow.org/guide/checkpoint

import os
import math
from glob import glob
from xml.etree.ElementTree import parse
import cv2
from tqdm import tqdm
import tensorflow as tf

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
		latest = tf.train.lastest_checkpoint(ckpt_path)
		model.load_weights(latest)
		print("%s Restored from %s" % (model_class.__name__, latest))

	else:
		print("%s Initialized" % model_class.__name__)

	ckpt = tf.train.Checkpoint(model)
	ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep)

	return model, ckpt, ckpt_manager


def apply_weight_decay(model, l2):
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
					 shorter_side_scale):
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

	Returns)
	1. anchors => 'tensorflow.python.framework.ops.EagerTensor'
	- Generated anchors in input image
	"""

	number_of_coords = feature_map_size[0] * feature_map_size[1]

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
		(number_of_coords * number_of_anchors_per_feature, 2))

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
	anchors_shape = tf.tile(tile_anchors_shape, [number_of_coords, 1])
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
	products_config['variables']['number_of_data'] = str(len(metadataset))

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
			shorter_side_scale)

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


def save_products(products_config,
				  products_file):

	if (os.path.isfile(products_file)):

		with open(products_file, 'a') as f:
			products_config.write(f)

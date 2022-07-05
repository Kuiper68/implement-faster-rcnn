# Implement Faster-RCNN Project
# Editor: Lee, Jongmo
# Filename: /train.py

# Reference URL
# https://www.tensorflow.org/guide/autodiff

import os
import pickle
from ast import literal_eval
from configparser import ConfigParser
import tensorflow as tf
from custom import utils
from custom.models import BackBone
from custom.models import Proposal
from custom.models import Detection

config_file = 'config.ini'
products_file = 'products.ini'
train_dataset_file = 'train_dataset.pickle'

def parser_config(config_file):
	"""
	Set initial as global variables

	Arguments)
	1. config_file => 'str'
	 - Filename with extension '*.ini'
	"""

	def square(anchor_side):
		"""
		Get anchor size from anchor side

		Arguments)
		1. anchor_side => 'int'
		 - Length of anchor side line

		Returns)
		1. anchor_size => 'int'
		 - Area of anchor
		"""

		anchor_size = anchor_side * anchor_side

		return anchor_size


	config = ConfigParser()
	config.read(config_file)

	image_input_size = literal_eval(config.get('variables', 'image_input_size'))
	anchor_side_per_feature = literal_eval(config.get('variables',
		'anchor_side_per_feature'))
	anchor_size_per_feature = list(map(square, anchor_side_per_feature))
	anchor_aspect_ratio_per_feature = literal_eval(config.get('variables',
		'anchor_aspect_ratio_per_feature'))
	shorter_side_scale = config.getint('variables', 'shorter_side_scale')
	number_of_sampled_region = config.getint('variables',
		'number_of_sampled_region')
	number_of_proposals = config.getint('variables', 'number_of_proposals')
	rpn_loss_lambda = config.getint('variables', 'rpn_loss_lambda')
	fast_rcnn_loss_lambda = config.getint('variables', 'fast_rcnn_loss_lambda')
	l2 = config.getfloat('variables', 'l2')
	learning_rate = config.getfloat('variables', 'learning_rate')
	momentum = config.getfloat('variables', 'momentum')
	train_valid_split_rate = config.getfloat('variables',
		'train_valid_split_rate')
	valid_steps_max = config.getint('variables', 'valid_steps_max')
	progress_bar_length = config.getint('variables', 'progress_bar_length')
	save_cycle = config.getint('variables', 'save_cycle')

	images_path = config.get('path', 'images_path')
	metadata_path = config.get('path', 'metadata_path')
	checkpoint_parser = config.get('path', 'checkpoint')

	ckpt_config = ConfigParser()
	ckpt_config.read(checkpoint_parser)

	max_to_keep = ckpt_config.getint('variables', 'max_to_keep')
	backbone_dir = ckpt_config.get('network', 'backbone')
	proposal_dir = ckpt_config.get('network', 'proposal')
	detection_dir = ckpt_config.get('network', 'detection')

	globals().update(locals())


def main():
	"""
	Execute program
	"""

	# Parse Variables for Executing program
	parser_config(config_file)
	number_of_anchors_per_feature = (len(anchor_size_per_feature)
		* len(anchor_aspect_ratio_per_feature))

	# Load Backbone Model
	backbone_utils = utils.load_network_manager(BackBone, backbone_dir,
		max_to_keep, input_shape=image_input_size)
	backbone_model = utils.apply_weight_decay(backbone_utils[0], l2)
	backbone_ckpt_manager = backbone_utils[2]

	# Products-2
	_, *feature_map_size = backbone_model.compute_output_shape(((None,)
		+ image_input_size)).as_list()
	feature_map_size = tuple(feature_map_size)
	pool_size = (feature_map_size[0] // 2, feature_map_size[1] // 2)

	# Load Proposal Model
	proposal_utils = utils.load_network_manager(Proposal, proposal_dir,
		max_to_keep, input_shape=feature_map_size,
		number_of_anchors_per_feature=number_of_anchors_per_feature)
	proposal_model = utils.apply_weight_decay(proposal_utils[0], l2)
	proposal_ckpt_manager = proposal_utils[2]

	# Load Train-Dataset
	if (os.path.isfile(products_file) & os.path.isfile(train_dataset_file)):
		train_dataset, number_of_objects, products_config = \
			utils.load_train_dataset(products_file, train_dataset_file)

	else:
		train_dataset, number_of_objects, products_config = utils.make_cache(
			products_file, train_dataset_file, pool_size, images_path,
			metadata_path, anchor_size_per_feature,
			anchor_aspect_ratio_per_feature, number_of_anchors_per_feature,
			shorter_side_scale, image_input_size, feature_map_size)

	# Load detection Model
	detection_utils = utils.load_network_manager(Detection, detection_dir,
		max_to_keep, input_shape=feature_map_size, pool_size=pool_size,
		number_of_proposals=number_of_proposals,
		number_of_objects=number_of_objects)
	detection_model = utils.apply_weight_decay(detection_utils[0], l2)
	detection_ckpt_manager = detection_utils[2]

	# Loss Functions
	sparse_categorical_cross_entropy = \
		tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
		reduction=tf.keras.losses.Reduction.NONE)
	huber = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)

	stochastic_gradient_descent = tf.keras.optimizers.SGD(
		learning_rate=learning_rate, momentum=momentum)

	# 4-Step Alternating Training
	utils.train_rpn(train_dataset, products_config, 200, train_valid_split_rate,
		valid_steps_max, backbone_model, proposal_model, True,
		sparse_categorical_cross_entropy, huber, rpn_loss_lambda,
		stochastic_gradient_descent, number_of_sampled_region,
		progress_bar_length, save_cycle, backbone_ckpt_manager,
		proposal_ckpt_manager)

	utils.train_fast_rcnn(train_dataset, products_config, 200,
		train_valid_split_rate, valid_steps_max, backbone_model, proposal_model,
		detection_model, True, sparse_categorical_cross_entropy, huber,
		fast_rcnn_loss_lambda, stochastic_gradient_descent, image_input_size,
		number_of_proposals, progress_bar_length, save_cycle,
		backbone_ckpt_manager, detection_ckpt_manager)

	utils.train_rpn(train_dataset, products_config, 200, train_valid_split_rate,
		valid_steps_max, backbone_model, proposal_model, False,
		sparse_categorical_cross_entropy, huber, rpn_loss_lambda,
		stochastic_gradient_descent, number_of_sampled_region,
		progress_bar_length, save_cycle, backbone_ckpt_manager,
		proposal_ckpt_manager)

	utils.train_fast_rcnn(train_dataset, products_config, 200,
		train_valid_split_rate, valid_steps_max, backbone_model, proposal_model,
		detection_model, False, sparse_categorical_cross_entropy, huber,
		fast_rcnn_loss_lambda, stochastic_gradient_descent, image_input_size,
		number_of_proposals, progress_bar_length, save_cycle,
		backbone_ckpt_manager, detection_ckpt_manager)


if __name__ == "__main__":
	main()

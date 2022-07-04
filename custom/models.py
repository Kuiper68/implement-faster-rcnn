# Implement Faster-RCNN Project
# Editor: Lee, Jongmo
# Filename: /custom/models.py

# Reference URL
# https://www.tensorflow.org/guide/keras/custom_layers_and_models

import tensorflow as tf
from custom.layers import ROIPooling

number_of_cls, number_of_reg = 2, 4

class BackBone(tf.keras.Model):
	"""
	Extract feature map from image by using model vgg16

	Init Data)
	1. input_shape => 'tuple'
	- Fixed shape when input image

	2. weights => 'str'
	- Initialization option for model weights when using model vgg16

	Arguments)
	1. image => 'tensorflow.python.framework.ops.EagerTensor'
	- Image from which to extract feature maps

	Returns)
	1. pooled_feature_maps => 'tensorflow.python.framework.ops.EagerTensor'
	- Region of feature map with pooling operation applied
	"""

	def __init__(self, input_shape,
					   weights="imagenet",
					   **kwargs):

		super(BackBone, self).__init__(**kwargs)

		vgg16 = tf.keras.applications.vgg16.VGG16(include_top=False,
			input_shape=input_shape, weights=weights)
		self.input_layer = vgg16.input
		*layers, _ = vgg16.layers

		for layer in layers:
			exec(f"self.{layer.name} = layer")

		self.out = self.call(self.input_layer)

		super(BackBone, self).__init__(inputs=self.input_layer,
			outputs=self.out, **kwargs)


	def build(self):

		self._init_graph_network(inputs=self.input_layer, outputs=self.out)


	def call(self, x_in,
				   training=False):

		for layer in self.layers[1:]:
			x_out = layer(x_in)
			x_in = x_out

		out = x_out

		return out




class Proposal(tf.keras.Model):
	"""
	Propose the region of some objects by performing operation on feature map

	Init Data)
	1. input_shape => 'tuple'
	- Feature map shape extracted from image

	2. number_of_anchors_per_feature => 'int'
	- Number of anchors related with a feature

	Arguments)
	1. feature_map => 'tensorflow.python.framework.ops.EagerTensor'
	- Extracted from image by backbone

	Returns)
	1. cls_prediction => 'tensorflow.python.framework.ops.EagerTensor'
	- Vector indicating the presence or not of object

	2. reg_prediction => 'tensorflow.python.framework.ops.EagerTensor'
	- Vector representing the region of object
	"""

	def __init__(self, input_shape,
					   number_of_anchors_per_feature,
					   **kwargs):

		super(Proposal, self).__init__(**kwargs)

		input_height, input_width, _ = input_shape
		cls_shape = [input_height * input_width * number_of_anchors_per_feature,
			number_of_cls]
		reg_shape = [input_height * input_width * number_of_anchors_per_feature,
			number_of_reg]

		self.input_layer = tf.keras.Input(input_shape)

		self.cls_conv = tf.keras.layers.Conv2D(
			number_of_anchors_per_feature * number_of_cls, (1, 1),
			activation='linear', name='cls_conv')
		self.cls_reshape = tf.keras.layers.Reshape(cls_shape,
			name='cls_reshape')

		self.reg_conv = tf.keras.layers.Conv2D(
			number_of_anchors_per_feature * number_of_reg, (1, 1),
			activation='linear', name='reg_conv')
		self.reg_reshape = tf.keras.layers.Reshape(reg_shape,
			name='reg_reshape')

		self.out = self.call(self.input_layer)

		super(Proposal, self).__init__(inputs=self.input_layer,
			outputs=self.out, **kwargs)


	def build(self):

		self._init_graph_network(inputs=self.input_layer, outputs=self.out)


	def call(self, x_in,
				   training=False):

		cls_feature_map = self.cls_conv(x_in)
		cls_prediction = self.cls_reshape(cls_feature_map)

		reg_feature_map = self.reg_conv(x_in)
		reg_prediction = self.reg_reshape(reg_feature_map)

		out = [cls_prediction, reg_prediction]

		return out




class Detection(tf.keras.Model):
	"""
	Detect the region and logit of objects

	Init Data)
	1. input_shape => 'tuple'
	- Feature map shape extracted from image

	2. pool_size => 'tuple'
	- Fixed size of feature map to return

	3. number_of_proposals => 'int'
	- Number of regions to be extracted

	4. number_of_objects => 'int'
	- Number of objects to predict

	Arguments)
	1. feature_map => 'tensorflow.python.framework.ops.EagerTensor'
	- Target of roi pooling operation

	2. proposals => 'tensorflow.python.framework.ops.EagerTensor'
	- Proposed regions in feature_map

	Returns)
	1. cls_logits_prediction => 'tensorflow.python.framework.ops.EagerTensor'
	- Logits vetctor that outputs objectness probability by using softmax

	2. reg_prediction => 'tensorflow.python.framework.ops.EagerTensor'
	- Vector representing the region of an object
	"""

	def __init__(self, input_shape,
					   pool_size,
					   number_of_proposals,
					   number_of_objects,
					   **kwargs):

		super(Detection, self).__init__(**kwargs)

		self.feature_map_input_layer = tf.keras.Input(input_shape)
		self.proposals_input_layer = tf.keras.Input((None, number_of_reg))
		self.roi_pooling = ROIPooling(pool_size, number_of_proposals)

		self.flatten = tf.keras.layers.TimeDistributed(
			tf.keras.layers.Flatten(name='flatten'))
		self.dense = tf.keras.layers.TimeDistributed(
			tf.keras.layers.Dense(4069, activation='relu', name='dense'))

		self.object_cls_logits_prediction = tf.keras.layers.TimeDistributed(
			tf.keras.layers.Dense(number_of_objects + 1, activation='linear',
			name='object_cls_logits_prediction'))
		self.object_reg_prediction = tf.keras.layers.TimeDistributed(
			tf.keras.layers.Dense(number_of_objects * 4, activation='linear',
			name='object_reg_prediction'))

		self.out = self.call([self.feature_map_input_layer,
			self.proposals_input_layer])

		super(Detection, self).__init__(inputs=[self.feature_map_input_layer,
			self.proposals_input_layer], outputs=self.out, **kwargs)


	def build(self):

		self._init_graph_network(input=[self.feature_map_input_layer,
			self.proposals_input_layer], outputs=self.out)


	def call(self, x_in,
				   training=False):

		pooled_feature_maps = self.roi_pooling(x_in)

		feature_vector = self.flatten(pooled_feature_maps)
		feature_vector = self.dense(feature_vector)

		cls_logits_prediction = self.object_cls_logits_prediction(
			feature_vector)
		reg_prediction = self.object_reg_prediction(feature_vector)

		out = [cls_logits_prediction, reg_prediction]

		return out

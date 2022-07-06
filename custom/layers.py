# Implement Faster-RCNN Project
# Editor: Lee, Jongmo
# Filename: /custom/layers.py

# Reference URL
# https://www.tensorflow.org/tutorials/customization/custom_layers
# https://medium.com/xplore-ai/
# 	implementing-attention-in-tensorflow-keras-using-roi-pooling-992508b6592b

import tensorflow as tf

class ROIPooling(tf.keras.layers.Layer):
    """
    Function that wrap and pool a region of feature map

    !!Caution!! => Batch process operation not applied yet...

    Init Data)
    1. pool_size
    - Fixed size of feature map to return

    2. number_of_proposals
    - Number of regions to be extracted

    Arguments)
    1. feature_map
    - Target of roi pooling operation

    2. proposals
    - Proposed regions in feature_map

    Returns)
    1. pooled_feature_maps
    - Region of feature map with pooling operation applied
    """

    def __init__(self, pool_size,
                       number_of_proposals,
                       **kwargs):

        self.pool_size = pool_size
        self.number_of_proposals = number_of_proposals

        super(ROIPooling, self).__init__(**kwargs)


    def get_config(self):

        config = super().get_config()
        config.update({
            "pool_size": self.pool_size,
            "number_of_proposals": self.number_of_proposals,
        })

        return config


    def compute_output_shape(self, input_shape):

        return ((None, self.number_of_proposals) + self.pool_size
                + (input_shape[0][3]))


    def call(self, x_in):

        assert (len(x_in) == 2)

        feature_map = x_in[0]
        proposals = x_in[1]
        input_shape = feature_map.shape

        pooled_feature_maps = []

        for proposal_index in range(self.number_of_proposals):
            xmin = tf.cast(tf.math.floor(proposals[0, proposal_index, 0]),
                tf.int32)
            ymin = tf.cast(tf.math.floor(proposals[0, proposal_index, 1]),
                tf.int32)
            xmax = tf.cast(tf.math.ceil(proposals[0, proposal_index, 2]),
                tf.int32)
            ymax = tf.cast(tf.math.ceil(proposals[0, proposal_index, 3]),
                tf.int32)

            pooled_feature_maps.append(
                tf.image.resize(feature_map[:, ymin:ymax, xmin:xmax, :],
                self.pool_size, method='nearest'))

        outputs = tf.expand_dims(tf.concat(pooled_feature_maps, 0), 0)

        return outputs

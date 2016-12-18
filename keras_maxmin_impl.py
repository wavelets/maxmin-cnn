from keras import backend as K
from keras.layers import Convolution2D


class MaxMinConvolution2D(Convolution2D):
    def __init__(self, nb_filter, nb_row, nb_col, **kwargs):
        super(MaxMinConvolution2D, self).__init__(nb_filter, nb_row, nb_col, **kwargs)

    def build(self, input_shape):
        super(MaxMinConvolution2D, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        output = super(MaxMinConvolution2D, self).call(x)
        return output

    def get_output_shape_for(self, input_shape):
        output_shape = super(MaxMinConvolution2D, self).get_output_shape_for(input_shape)
        return output_shape

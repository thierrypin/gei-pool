

import sys

import numpy as np

import mxnet as mx
from mxnet import nd
# from mxnet.gluon.model_zoo import vision
from mxnet.gluon import HybridBlock, nn
from gluoncv.nn.feature import FeatureExtractor



class LRN(HybridBlock):
    def __init__(self, **kwargs):
        super(LRN, self).__init__(**kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        return F.LRN(x, nsize=5)

class ExpandDims(HybridBlock):
    def __init__(self, axis, **kwargs):
        super(ExpandDims, self).__init__(**kwargs)
        self.axis = axis

    def hybrid_forward(self, F, x, *args, **kwargs):
        return F.expand_dims(x, axis=self.axis)

# Based on code from https://github.com/harvitronix/five-video-classification-methods
class ResearchModels():
    def __init__(self, nb_classes, model, saved_model=None, input_shape=(1, 64, 44), train_all=False):
        """
        `model` =
        `nb_classes` = the number of classes to predict
        `saved_model` = the path to a saved Keras model to load
        """

        # Set defaults.
        self.input_shape = input_shape
        self.nb_classes = nb_classes

        # The function, among the ones below, that builds the model
        # We get it by its name
        network_builder = getattr(self, model, None)
        if network_builder is None:
            print("Unknown network.")
            sys.exit(1)

        # Load pre-trained weights
        if saved_model is not None:
            print("Loading weights %s" % saved_model)
            symbols = saved_model[:-11] + "symbol.json"
            self.model = nn.SymbolBlock.imports(symbols, ['data'], saved_model, ctx=mx.gpu())
        else:
        # Get the appropriate model.
            self.model = network_builder()

        self.model.initialize(ctx=mx.gpu())

        shape = (1,) + input_shape
        x = nd.zeros(shape, dtype=np.float32, ctx=mx.gpu())
        print(x.shape)
        self.model.summary(x)

        self.model(x)
        self.model.hybridize()



    def geinet(self):

        net = nn.HybridSequential()
        with net.name_scope():
            net.add(nn.Conv2D(18, (7, 7)))
            net.add(nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
            net.add(LRN())

            net.add(nn.Conv2D(45, (5 ,5)))
            net.add(nn.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
            net.add(LRN())

            net.add(nn.Dense(1024))
            net.add(nn.Dropout(0.3))

            net.add(nn.Dense(self.nb_classes))

        return net
    
    def no_pool_geinet(self):

        net = nn.HybridSequential()
        with net.name_scope():
            net.add(nn.Conv2D(18, (7, 7), strides=(2, 2)))
            net.add(LRN())

            net.add(nn.Conv2D(45, (5, 5), strides=(2, 2)))
            net.add(LRN())

            net.add(nn.Dense(1024))
            net.add(nn.Dropout(0.3))

            net.add(nn.Dense(self.nb_classes))

        return net

    def no_pool(self):

        net = nn.HybridSequential()
        with net.name_scope():
            net.add(nn.Conv2D(18, (7, 7), input_size=self.input_shape))
            net.add(nn.Conv2D(18, (7, 7), strides=(2, 2)))
            net.add(LRN())

            net.add(nn.Conv2D(45, (5 ,5)))
            net.add(nn.Conv2D(45, (5, 5), strides=(2, 2)))
            net.add(LRN())

            net.add(nn.Dense(1024))
            net.add(nn.Dropout(0.3))

            net.add(nn.Dense(self.nb_classes))

        return net



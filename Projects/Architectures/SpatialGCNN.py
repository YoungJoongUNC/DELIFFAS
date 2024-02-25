
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import Architectures.ResNet50 as Layers
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
SPATIAL_GCNN_NAME_COUNTER = 0
from os import path

########################################################################################################################
# Class
########################################################################################################################

class SpatialGCNN:

    #######################################

    def __init__(self,denseInitializerScale, dataFormat, featureSize1, featureSize2, useBatchNorm, fullyConnected, adjacency, ringValue, normalize,denseInnerBlock,numResidualBlocks, numGraphNodes,inputSize, outputSize,basePath,sequence):

        print('++ Init Spatial GCNN')

        if numResidualBlocks % 2 == 0:
            self.numResidualBlocks = numResidualBlocks
        else:
            print('Number of residual blocks has to be even!')

        self.featureSize1               = featureSize1
        self.featureSize2               = featureSize2
        self.useBatchNorm               = useBatchNorm
        self.denseInitializerScale      = denseInitializerScale
        self.numGraphNodes              = numGraphNodes

        ##################################

        adjPath = basePath + sequence + 'Adj' + str(self.numGraphNodes) + '.npy'
        print('Check if adjacency exists')
        print(adjPath)

        if path.exists(adjPath):
            print('Path exists --> load file')
            self.A = np.load(adjPath)
        else:
            print('Path DOES NOT exists --> save file')
            self.A = self.getA(adjacency, ringValue, normalize)
            np.save(adjPath, self.A)

        self.A = tf.reshape(self.A, [-1, self.numGraphNodes, self.numGraphNodes])

       ###################################


        self.dataFormat                 = dataFormat
        self.layerCounter               = 0
        self.ringValue                  = ringValue
        self.normalize                  = normalize
        self.fullyConnected             = fullyConnected
        self.denseInnerBlock            = denseInnerBlock
        self.inputSize                  = inputSize
        self.outputSize                 = outputSize

        print(' ++ F1 size: ' + str(self.featureSize1))
        print(' ++ F2 size: ' + str(self.featureSize2))
        print(' ++ Use BatchNorm: ' + str(self.useBatchNorm))
        print(' ++ Init scale: ' + str(self.denseInitializerScale))
        print(' ++ # of Graph nodes: ' + str(self.numGraphNodes))
        print(' ++ Data format: ' + str(self.dataFormat))
        print(' ++ Ring value: ' + str(self.ringValue))
        print(' ++ Normalize: ' + str(self.normalize))
        print(' ++ Fully connected: ' + str(self.fullyConnected))
        print(' ++ # of residual blocks: ' + str(self.numResidualBlocks))
        print(' ++ Input size: ' + str(self.inputSize))
        print(' ++ Output size: ' + str(self.outputSize))

        print('++ End Init Spatial GCNN')

    #######################################

    def getA(self,adjacency, ringValue, normalize):

        print('     ++ Start getting adjacency')

        num_vertices = adjacency.shape[0]
        a = np.zeros(shape=[num_vertices, num_vertices], dtype=np.float32)
        rangeVertices = range(0,len(adjacency))
        # old adjacency
        if ringValue == -1:
            for vertex, neighbors in enumerate(adjacency):  # vertex is 0-indexed
                a[vertex, vertex] = 1

            for vertex in range(0, len(adjacency)):  # rows
                for neighbours in adjacency[vertex]: #neighbours
                    if neighbours != 0:
                        a[vertex, neighbours - 1] = 1
        # new adjacency
        else:
            for vertex, neighbors in enumerate(adjacency):  # vertex is 0-indexed
                a[vertex, vertex] = ringValue

            while ringValue > 0:

                for vertex in rangeVertices:  # rows
                    for neighborCandidate  in rangeVertices:  # columns
                        if a[vertex,neighborCandidate] == ringValue:  # if it is a neighbour
                            for neighbours in adjacency[neighborCandidate]:
                                if a[vertex, neighbours - 1] == 0 and neighbours!=0:
                                    a[vertex, neighbours - 1] = ringValue -1

                ringValue = ringValue -1

            if normalize:
                row_sums = a.sum(axis=1)
                a = a / row_sums[:, np.newaxis]

        print('     ++ End getting adjacency')

        return a

    #######################################

    def build(self, nameScope, training):

        print('++ Build Spatial GCNN')

        def graphBatchNorm(x, filterSize):
            x = tf.reshape(x, [-1, self.numGraphNodes * filterSize])
            x = Layers.batch_norm(inputs=x, data_format=self.dataFormat, nameScope=nameScope, training=training)
            x = tf.reshape(x, [-1, self.numGraphNodes, filterSize])
            return x

        def gcnn(x, filterSize,initializerScale, dense=False):
            print('  GCNNOperator ' +str(filterSize))
            layer = GCNNOperator(filterSize, initializerScale, self.A, nameScope=nameScope, denseConnect=dense,numGraphNodes=self.numGraphNodes)#, sharedWeights='sharing')
            layer.build(x.shape)
            x = layer(x)
            return x

        def denseBlock(x, xSkip):
            if self.useBatchNorm:
                x = graphBatchNorm(x, self.featureSize2)
            x = tf.nn.elu(x)
            x = gcnn(x, self.featureSize2, self.denseInitializerScale* 0.01, dense=True)
            x = xSkip + x
            xSkip = x

            return x, xSkip

        def residualBlock(x, xSkip):
            if self.useBatchNorm:
                x = graphBatchNorm(x, self.featureSize2)
            x = tf.nn.elu(x)
            x = gcnn(x, self.featureSize1, self.denseInitializerScale)

            if self.useBatchNorm:
                x = graphBatchNorm(x, self.featureSize1)
            x = tf.nn.elu(x)
            x = gcnn(x, self.featureSize2, self.denseInitializerScale)

            x = xSkip + x
            xSkip = x

            return x, xSkip

        temporalPose = Input(shape=[self.numGraphNodes, self.inputSize])

        x = gcnn(temporalPose, self.featureSize2, self.denseInitializerScale)
        xSkip = x

        ######################

        for block in range(0, int(self.numResidualBlocks/2)):
            x, xSkip = residualBlock(x, xSkip)

        ######################

        if self.denseInnerBlock:
            x, xSkip = denseBlock(x, xSkip)

        ######################

        for block in range(0, int(self.numResidualBlocks/2)):
            x, xSkip = residualBlock(x, xSkip)

        ######################
        #map to output

        #try other fully connected backbone
        if self.fullyConnected == 'variant1':
            x = tf.nn.elu(x)
            x = gcnn(x, 1, self.denseInitializerScale)
            x = tf.nn.elu(x)
            x = tf.reshape(x, [-1, self.numGraphNodes])
            x = Layers.dense(x=x, outputSize=self.numGraphNodes * self.outputSize, denseInitializerScale=self.denseInitializerScale, nameScope=nameScope)
            x = tf.reshape(x, [-1, self.numGraphNodes, self.outputSize])

        elif self.fullyConnected == 'variant2':
            x = tf.nn.elu(x)
            x = gcnn(x, self.outputSize, self.denseInitializerScale)
            x = tf.nn.elu(x)
            x = Layers.dense(x=x, outputSize=self.outputSize, denseInitializerScale=self.denseInitializerScale, nameScope=nameScope)

        elif self.fullyConnected == 'variant3':
            x = tf.nn.elu(x)
            x = gcnn(x, self.outputSize, self.denseInitializerScale,dense=True)

        else:
            x = tf.nn.elu(x)
            x = gcnn(x, self.outputSize, self.denseInitializerScale)

        self.model = Model(temporalPose, x)

        Layers.BATCH_NORM_NAME_COUNTER = 0
        Layers.CONV2D_NAME_COUNTER = 0
        Layers.DENSE_NAME_COUNTER = 0

        print(self.model.summary())

        print('++ End Build Spatial GCNN')

########################################################################################################################
# Class
########################################################################################################################

class GCNNOperator(tf.keras.layers.Layer):

    def __init__(self, fNew, denseInitializerScale, A, nameScope, denseConnect = False, numGraphNodes = 0, sharedWeights='no_sharing'):
        super(GCNNOperator, self).__init__()
        self.fNew                       = fNew
        self.denseInitializerScale      = denseInitializerScale
        self.A                          = A
        self.nameScope                  = nameScope
        self.denseConnect               = denseConnect
        self.numGraphNodes              = numGraphNodes
        self.sharedWeights              = sharedWeights # no_sharing , sharing , neighbour_split

    #######################################

    def build(self, input_shape):

        global SPATIAL_GCNN_NAME_COUNTER

        def initializerDenseKernel(*args, **kwargs):
            rand = tf.random.normal([self.numGraphNodes , self.numGraphNodes ], -1, 1)
            rand3 = rand * self.denseInitializerScale
            return rand3

        def initializerKernel(*args, **kwargs):
            if self.sharedWeights == 'sharing':
                rand = tf.random.normal([1, input_shape[2], self.fNew], -1, 1)
            else:
                rand = tf.random.normal([self.numGraphNodes , input_shape[2], self.fNew], -1, 1)
            rand = rand * self.denseInitializerScale
            return rand

        def initializerBias(*args, **kwargs):
            if self.sharedWeights == 'sharing':
                rand = tf.random.normal([1, self.fNew], -1, 1)
            else:
                rand = tf.random.normal([self.numGraphNodes , self.fNew], -1, 1)
            rand = rand * self.denseInitializerScale
            return rand

        if self.denseConnect:
            self.kernel = self.add_weight(name=self.nameScope + '/dense_' +str(SPATIAL_GCNN_NAME_COUNTER) +'/kernel_' + str(SPATIAL_GCNN_NAME_COUNTER),
                                          shape=([self.numGraphNodes , self.numGraphNodes ]),
                                          initializer=initializerDenseKernel,
                                          trainable=True)
        else:
            if self.sharedWeights == 'sharing':
                self.kernel = self.add_weight(
                    name=self.nameScope + '/gcnn_' + str(SPATIAL_GCNN_NAME_COUNTER) + '/kernel_' + str(SPATIAL_GCNN_NAME_COUNTER),
                    shape=([1, input_shape[2], self.fNew]),
                    initializer=initializerKernel,
                    trainable=True)
            else:
                self.kernel = self.add_weight(
                    name=self.nameScope + '/gcnn_' + str(SPATIAL_GCNN_NAME_COUNTER) + '/kernel_' + str(SPATIAL_GCNN_NAME_COUNTER),
                    shape=([self.numGraphNodes , input_shape[2], self.fNew]),
                    initializer=initializerKernel,
                    trainable=True)


            if self.sharedWeights == 'sharing':
                self.kernel = tf.tile(self.kernel,[self.numGraphNodes,1,1])

        if self.sharedWeights == 'sharing':
            self.bias = self.add_weight(name=self.nameScope + '/gcnn_' +str(SPATIAL_GCNN_NAME_COUNTER) +'/bias_' + str(SPATIAL_GCNN_NAME_COUNTER),
                                          shape=([1, self.fNew]),
                                          initializer=initializerBias,
                                          trainable=True)
        else:
            self.bias = self.add_weight(
                name=self.nameScope + '/gcnn_' + str(SPATIAL_GCNN_NAME_COUNTER) + '/bias_' + str(SPATIAL_GCNN_NAME_COUNTER),
                shape=([self.numGraphNodes, self.fNew]),
                initializer=initializerBias,
                trainable=True)

        if self.sharedWeights == 'sharing':
            self.bias = tf.tile(self.bias, [self.numGraphNodes, 1])

        self.F =  input_shape[2]

        SPATIAL_GCNN_NAME_COUNTER = SPATIAL_GCNN_NAME_COUNTER + 1

        super(GCNNOperator, self).build(input_shape)

    #######################################

    def call(self, input_):

        # dense
        if self.denseConnect:
            hNewFinal = tf.matmul(self.kernel, input_)
        else:
            #tile
            hOld = tf.reshape(input_, [-1, self.numGraphNodes , self.F, 1])
            hOld = tf.tile(hOld, [1, 1, 1, self.fNew])

            #kernel
            hNewFinal = hOld * self.kernel
            hNewFinal = tf.reduce_sum(hNewFinal, 2)

            hNewFinal = tf.matmul(self.A, hNewFinal)

        #bias
        hNewFinal = hNewFinal + self.bias

        return hNewFinal



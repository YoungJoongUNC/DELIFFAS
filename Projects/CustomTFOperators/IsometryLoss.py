import sys

sys.path.append("../")

import tensorflow as tf
import AdditionalUtils.OBJReader as OBJReader

########################################################################################################################
# Isometry Loss
########################################################################################################################

class IsometryLoss:

    ########################################################################################################################

    def __init__(self, meshFilePath, restTensor = None):

        # obj Reader
        self.objReader = OBJReader.OBJReader(meshFilePath)

        # adjacency
        self.adjacency = tf.constant(self.objReader.adjacency, dtype=tf.float32)
        self.adjacency = tf.reshape(self.adjacency, [self.objReader.numberOfVertices, self.objReader.numberOfVertices, 1])
        self.adjacency = tf.tile(self.adjacency, [1, 1, 3])

        # rest vertex pos
        if restTensor is not None:
            self.restVertexPos = restTensor
        else:
            self.restVertexPos =  self.objReader.vertexCoordinates
            self.restVertexPos = tf.reshape(self.restVertexPos, [1, tf.shape(self.restVertexPos)[0],tf.shape(self.restVertexPos)[1]])

        #rest edge length
        self.restEdgeLength = self.get_edge_length(self.restVertexPos)

    ########################################################################################################################

    def get_edge_length(self, vertexPos):

        vertexPosI = tf.reshape(vertexPos,[-1, self.objReader.numberOfVertices, 1, 3])
        vertexPosI = tf.tile(vertexPosI, [1, 1, self.objReader.numberOfVertices, 1])
        vertexPosI = self.adjacency * vertexPosI

        vertexPosJ = tf.reshape(vertexPos, [-1, 1, self.objReader.numberOfVertices, 3])
        vertexPosJ = tf.tile(vertexPosJ, [1, self.objReader.numberOfVertices,1, 1])
        vertexPosJ = self.adjacency * vertexPosJ

        edgeLength = vertexPosI - vertexPosJ
        edgeLength = edgeLength  * edgeLength
        edgeLength = tf.reduce_sum(edgeLength, 3)
        edgeLength = tf.sqrt(edgeLength)

        return edgeLength

    ########################################################################################################################

    def getLoss(self, inputMeshTensor):
        batchSize = tf.shape(inputMeshTensor)[0]

        restEdgeLength = tf.tile(self.restEdgeLength, [batchSize, 1,1])

        edgeLength = self.get_edge_length(inputMeshTensor)

        diff = restEdgeLength - edgeLength

        reduc1 = tf.reduce_sum((diff * diff), 0)

        loss = tf.reduce_sum(reduc1) #* self.objReader.adjacencyWeights)

        loss = loss / float(batchSize * self.objReader.numberOfEdges)

        return loss

    ########################################################################################################################

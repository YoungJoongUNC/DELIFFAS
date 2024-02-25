
########################################################################################################################
# Imports
########################################################################################################################

import tensorflow as tf

########################################################################################################################
# Chamfer loss
########################################################################################################################

def chamferLoss(point_set_a, point_set_b, threshold = -1):

    # Create N x M matrix where the entry i,j corresponds to ai - bj (vector of
    # dimension D).
    difference = (tf.expand_dims(point_set_a, axis=-2) - tf.expand_dims(point_set_b, axis=-3))

    # Calculate the square distances between each two points: |ai - bj|^2.
    # square_distances = tf.einsum("...i,...i->...", difference, difference)
    square_distances = tf.reduce_sum(tf.square(difference), axis= -1)

    minimum_square_distance_a_to_b = tf.reduce_min( input_tensor=square_distances, axis=-1)
    minimum_square_distance_b_to_a = tf.reduce_min( input_tensor=square_distances, axis=-2)

    # mask it
    if threshold != -1:
        maskAB = tf.less_equal(minimum_square_distance_a_to_b, threshold * threshold)
        maskAB = tf.cast(maskAB, dtype=tf.float32)
        minimum_square_distance_a_to_b = minimum_square_distance_a_to_b * maskAB

        maskBA = tf.less_equal(minimum_square_distance_b_to_a, threshold * threshold)
        maskBA = tf.cast(maskBA, dtype=tf.float32)
        minimum_square_distance_b_to_a = minimum_square_distance_b_to_a * maskBA

    return tf.reduce_mean(input_tensor=minimum_square_distance_a_to_b, axis=-1) + tf.reduce_mean(input_tensor=minimum_square_distance_b_to_a, axis=-1)


########################################################################################################################
# Chamfer Hausdorff metric
########################################################################################################################

def chamferHausdorffMetric(point_set_a, point_set_b):

    # Create N x M matrix where the entry i,j corresponds to ai - bj (vector of
    # dimension D).
    difference = (tf.expand_dims(point_set_a, axis=-2) - tf.expand_dims(point_set_b, axis=-3))

    # Calculate the square distances between each two points: |ai - bj|^2.
    # square_distances = tf.einsum("...i,...i->...", difference, difference)
    square_distances = tf.sqrt(tf.reduce_sum(tf.square(difference), axis= -1))

    minimum_square_distance_a_to_b = tf.reduce_min( input_tensor=square_distances, axis=-1)
    minimum_square_distance_b_to_a = tf.reduce_min( input_tensor=square_distances, axis=-2)

    a2b = tf.reduce_mean(input_tensor=minimum_square_distance_a_to_b, axis=-1)
    b2a = tf.reduce_mean(input_tensor=minimum_square_distance_b_to_a, axis=-1)

    return (a2b + b2a) / 2.0, a2b

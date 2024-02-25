
#############################################################################################
# import
#############################################################################################

import tensorflow as tf

#############################################################################################
# Model architecture
#############################################################################################

def init_nerf_model(D=3,
                    W=128,
                    uv_feat_ch=32,
                    normal_feat_ch=32,
                    output_ch=3,
                    skips=[],
                    ):


    relu = tf.keras.layers.ReLU()
    def dense(W, act=relu):
        return tf.keras.layers.Dense(W, activation=act)

    uv_feat_ch = int(uv_feat_ch)
    normal_feat_ch = int(normal_feat_ch)

    inputs = tf.keras.Input(shape=(None,uv_feat_ch+normal_feat_ch))
    uv_feat, normal_feat = tf.split(inputs, [uv_feat_ch, normal_feat_ch], -1)

    outputs = uv_feat

    for i in range(D):
        outputs = dense(W)(outputs)
        if (D > 3) and (i == 4):
            outputs = tf.concat([normal_feat, outputs], -1)

    outputs = dense(output_ch, act=None)(outputs)



    model = tf.keras.Model(inputs=inputs, outputs=outputs)


    print(model.summary())
    return model
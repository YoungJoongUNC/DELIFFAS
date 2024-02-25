
########################################################################################################################
# Imports
########################################################################################################################

import tensorflow as tf

########################################################################################################################
# UNet
########################################################################################################################

class UNet:

    ########################################################################################################################
    # downsample
    ########################################################################################################################

    def downsample(self,filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result

    ########################################################################################################################
    # upsample
    ########################################################################################################################

    def upsample(self,filters, size, apply_dropout=False, strides=2):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=strides, padding='same', kernel_initializer=initializer, use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

    ########################################################################################################################
    # __init__
    ########################################################################################################################


    def __init__(self, inputChannels, outputChannels, inputResolutionU, inputResolutionV, lastActivation='tanh', netMode='bigNet', deep_skip_unet=False):

        self.outputChannels = outputChannels

        inputs = tf.keras.layers.Input(shape=[inputResolutionV,inputResolutionU,inputChannels])


        down_stack = [
            self.downsample(16, 4, apply_batchnorm=False),

            self.downsample(32, 4),
            self.downsample(64, 4),

            self.downsample(128, 4),
            self.downsample(256, 4),
            self.downsample(512, 4),

            self.downsample(512, 4),
            self.downsample(512, 4),
            self.downsample(512, 4),

            self.downsample(512, 4),
        ]

        up_stack = [
            self.upsample(512, 4, apply_dropout=True),
            self.upsample(512, 4, apply_dropout=True),
            self.upsample(512, 4, apply_dropout=True),

            self.upsample(512, 4),
            self.upsample(256, 4),
            self.upsample(128, 4),

            self.upsample(64, 4),
            self.upsample(32, 4),
            self.upsample(16, 4),
        ]

        initializer = tf.random_normal_initializer(0., 0.02)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        up_idx=0
        output_idx = []
        outputs = []
        for up, skip in zip(up_stack, skips):
            x = up(x)
            if up_idx in output_idx:
                outputs.append(x)
            x = tf.keras.layers.Concatenate()([x, skip])
            up_idx += 1

        end = tf.keras.layers.Conv2DTranspose(self.outputChannels,
                                             4,
                                             strides=2,
                                             padding='same',
                                             kernel_initializer=initializer,
                                             activation= lastActivation)

        x = end(x)

        ##################################

        if lastActivation == 'tanh':
            x = (x + 1.0) / 2.0
        else:
            x = tf.keras.layers.ReLU()(x)
        outputs.append(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)




        print(self.model.summary())

    ########################################################################################################################
    # Additional backbone
    ########################################################################################################################

    def initTinyBackbone(self, inputResV, inputResU, inputChannels, outputChannels):

        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[inputResV, inputResU, inputChannels], name='input_image')

        x= tf.keras.layers.Conv2D(outputChannels, (1,1), strides=1, padding='same', kernel_initializer=initializer, use_bias=False, activation='tanh')(inp)
        x = (x + 1.0) / 2.0

        self.tinyBackbone = tf.keras.Model(inputs=inp, outputs=x)

        print(self.tinyBackbone.summary())

    ########################################################################################################################
    # discriminator
    ########################################################################################################################

    def initDiscriminator(self,inputResV, inputResU, inputChannels, outputChannels):

        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[inputResV, inputResU, inputChannels], name='input_image')
        tar = tf.keras.layers.Input(shape=[inputResV, inputResU, outputChannels], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])

        down1 = self.downsample(16, 8, False)(x)
        down2 = self.downsample(32, 4)(down1)
        down3 = self.downsample(64, 4)(down2)
        down4 = self.downsample(128, 4)(down3)
        down5 = self.downsample(256, 4)(down4)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down5)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,  kernel_initializer=initializer, use_bias=False)(zero_pad1)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

        last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)

        self.discriminator =  tf.keras.Model(inputs=[inp, tar], outputs=last)

        print(self.discriminator.summary())

    ########################################################################################################################
    # Discriminator loss
    ########################################################################################################################

    def discriminator_loss (self, disc_real_output, disc_generated_output):
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

        real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = tf.reduce_mean(real_loss) + tf.reduce_mean(generated_loss) * 0.0
        return total_disc_loss

    ########################################################################################################################
    # Generator loss
    ########################################################################################################################

    def generator_loss(self, disc_generated_output, gen_output, target):
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)

        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        gan_loss = tf.reduce_mean(gan_loss) * 0.0

        # Mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output)) * 1.0
        total_gen_loss = gan_loss + l1_loss

        return total_gen_loss, gan_loss, l1_loss
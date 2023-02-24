from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.constraints import max_norm


def ResidualBlock(input, nb_channels, sampling_rate, nb_filters_1, nb_filters_2):
    # Shortcut connection
    shortcut = input
    # Convolutional layers
    block1 = Conv2D(nb_filters_1, (1, 1), padding='same',
                    input_shape=(nb_channels, sampling_rate, 1),
                    use_bias=False)(input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    block1 = Conv2D(nb_filters_1, (3, 3), padding='same',
                    input_shape=(nb_channels, sampling_rate, 1),
                    use_bias=False)(input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    block1 = Conv2D(nb_filters_2, (1, 1), padding='same',
                    input_shape=(nb_channels, sampling_rate, 1),
                    use_bias=False)(input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    # Add the shortcut connection
    block2 = Concatenate()([block1, shortcut])
    block2 = Activation('relu')(block2)

    return block2


def ResNet(input, nb_classes, norm_rate, nb_channels, sampling_rate, nb_filters_1, nb_filters_2):
    # Initial convolutional layer
    x = Conv2D(nb_filters_1, (1, 1), padding='same',
               input_shape=(nb_channels, sampling_rate, 1),
               use_bias=False)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3))(x)

    # Residual blocks
    x = ResidualBlock(x,  nb_channels,
                      sampling_rate, nb_filters_1, nb_filters_2)
    x = ResidualBlock(x, nb_channels,
                      sampling_rate, nb_filters_1, nb_filters_2)
    x = ResidualBlock(x, nb_channels,
                      sampling_rate, nb_filters_1, nb_filters_2)

    # Final layers
    flatten = Flatten(name='flatten')(x)

    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input, outputs=softmax)


nb_classes = 4
norm_rate = 0.25
nb_channels = 64
sampling_rate = 256
nb_filters_1 = 128
nb_filters_2 = 512

input = Input(shape=(nb_channels, sampling_rate, 1), name='input')
residual_block = Model(inputs=input, outputs=ResidualBlock(
    input, nb_channels, sampling_rate, nb_filters_1, nb_filters_2))
resnet = ResNet(input, nb_classes, norm_rate, nb_channels,
                sampling_rate, nb_filters_1, nb_filters_2)

dot_img_file_1 = 'D:\Phd\ResNet\Residual_block.png'
dot_img_file_2 = 'D:\Phd\ResNet\Resnet.png'


plot_model(residual_block, to_file=dot_img_file_1, show_shapes=True)
plot_model(resnet, to_file=dot_img_file_2, show_shapes=True)

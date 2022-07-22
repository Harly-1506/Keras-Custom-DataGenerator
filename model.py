from keras import layers
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Flatten, Dense, MaxPooling2D, BatchNormalization


class MyModel(keras.Model):
    """
  Author: Trần Minh Hải
  Customized model with increased number of filters:
    + There are 4 stages, each satge has 2 conv2d layers and a maxpooling layers
    + The number of filters through each stage will be multiplied by 2

  """

    def __init__(self, filter=64, kernel=(3, 3),  node=64, outputs=1):
        super(MyModel, self).__init__()
        self.filter = filter
        self.kernel = kernel
        self.node = node
        self.outputs = outputs

        self.stage1 = keras.Sequential([
            layers.Conv2D(self.filter, kernel_size=self.kernel,
                          activation='relu'),
            layers.Conv2D(self.filter, kernel_size=self.kernel,
                          activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2))
        ])
        self.stage2 = keras.Sequential([
            layers.Conv2D(self.filter*2, kernel_size=self.kernel,
                          activation='relu'),
            layers.Conv2D(self.filter*2, kernel_size=self.kernel,
                          activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2))
        ])
        self.stage3 = keras.Sequential([
            layers.Conv2D(self.filter*4, kernel_size=self.kernel,
                          activation='relu', padding='same'),
            layers.Conv2D(self.filter*4, kernel_size=self.kernel,
                          activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2))
        ])
        self.stage4 = keras.Sequential([
            layers.Conv2D(self.filter*8, kernel_size=self.kernel,
                          activation='relu', padding='same'),
            layers.Conv2D(self.filter*8, kernel_size=self.kernel,
                          activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2), padding='same')
        ])
        self.outputs = keras.Sequential([
                                        layers.Flatten(),
                                        layers.Dense(
                                            self.node*64, activation="relu"),
                                        layers.Dense(
                                            self.outputs, activation="softmax")
                                        ])

    def call(self, inputs):
        x = self.stage1(inputs)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.outputs(x)
        return x

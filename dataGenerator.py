from tensorflow.keras.utils import Sequence, to_categorical
import numpy as np
import cv2


class DataGenerator(Sequence):
    def __init__(self,
                 img_paths,
                 labels, 
                 batch_size=32,
                 dim=(224, 224),
                 n_channels=3,
                 n_classes=4,
                 shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.img_paths = img_paths
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.img_indexes = np.arange(len(self.img_paths))
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.floor(len(self.img_indexes) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temps = [self.img_indexes[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temps)
        return X, y
    def on_epoch_end(self):
        #Updates indexes after each epoch
        self.indexes = np.arange(len(self.img_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, list_IDs_temps):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = []
        for i, ID in enumerate(list_IDs_temps):
              img = cv2.imread(self.img_paths[ID])
              img = cv2.resize(img, (128, 128))
              img = img.reshape((1, 128, 128, 3))
              
              X[i,] = img
              y.append(self.labels[ID])

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

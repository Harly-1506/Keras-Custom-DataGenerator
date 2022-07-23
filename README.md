# Keras-Custom-ImageDataGenerator

## What is data generator?

- When you have a big dataset and you can not load your data in common way. Then the phenomenon of running **out of RAM** will occur. So **data generator** will help you fix this problem easily.
- Data generator will help to split the data by batch_size to upload during training. In addition, we can also customize the training data easily 
- So in this repo I will share how to custom dataGenerator in Keras :wink:

## Data Generator :bulb:
## 1. Standard Keras Data Generator

- Keras provides a data generator with image data. And it help us to make more data call augmentation, you can read [ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator) and [flow_from_directory](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_directory) for more information

```python
from keras.preprocessing.image import ImageDataGenerator

aug = ImageDataGenerator(rescale=1/255.0)

train_ds = aug.flow_from_directory(train_paths, target_size=(224,224),  
                                   class_mode='categorical' ,batch_size=128,shuffle = True)
val_ds =  aug.flow_from_directory(val_paths, target_size=(224,224), 
                                  class_mode='categorical', batch_size=128 )
test = aug.flow_from_directory(test_path,class_mode="categorical", target_size=(224,224), batch_size=64 )

model = ...
...

model.compile(...)

H = model.fit(train_ds, validation_data = val_ds, epochs= 5)

```
- I have full example [here](https://github.com/Harly-1506/American-Sign-languages-datasets-Classification/blob/main/ASL_ResNet50.ipynb) :smile:

## 2. Keras Custom dataGenerator :dart:

- To custom Data Generator. Keras provides us with the Sequence class and allows us to create classes that can inherit from it.
  
    - Initialization function: **__init__()**: includes the function's input as images and labels and image properties such as dimension, number of layers
    -  **on_epoch_end()** function: update indexes through each epoch and suffle data if suffle = True
    - **__len__()** function: Returns the number of batches per epoch
    - **__data_generation()** function: Load image in batch_size
    - **__getitem__()** function: get indexes and return batch of images according to __data_generation() function
 
```python
from tensorflow.keras.utils import Sequence, to_categorical
import numpy as np
import cv2
import keras


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
        
configs = {'batch_size': 64,
          'dim': (128,128),
          'n_channels' : 3,
          'n_classes': 29,
          'shuffle': True}

# Load datasets
train_generator = DataGenerator(train_images_path, train_labels, **configs)
val_generator = DataGenerator(val_images_path, val_labels, **configs)

model = ...
model.fit(train_generator, validation_data = val_generator, epochs= 50)

```

#### Data Generator with albumentations

- You can use custom your data when you load batch_size to your liking, you can use **albumentations** like this code below

```python
from albumentations import ( 
Transpose, Compose, Rotate, RandomBrightness, RandomContrast, RandomRotate90
) 

transforms_train = Compose([
            Rotate(limit=40),
            RandomBrightness(limit=0.1),
            RandomContrast(limit=0.2, p=0.5),
            RandomRotate90(),
            Transpose(),         
        ])
        
def train_fn(image):
  data = {"image":image}
  aug_data = transforms_train(**data)
  aug_img = aug_data["image"]
  return aug_img
  
class DataGenerator(Sequence):
   .....

  def __data_generation(self, list_IDs_temps):
      X = np.empty((self.batch_size, *self.dim, self.n_channels))
      y = []
      for i, ID in enumerate(list_IDs_temps):
            # processing for your image or your data
            .....
            if self.augmentaiton == True:
              img = train_fn(img)

            X[i,] = img
            y.append(self.labels[ID])

      return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
```
## 3.Run code in this repo

```
git clone https://github.com/Harly-1506/Keras-Custom-DataGenerator.git
```
- After that, run file **main.ipynb** in your jupyter or Colab

___
**If you like  this repo, just star to support me** :star:

**Enjoy with Code** :wink:

**Harly**

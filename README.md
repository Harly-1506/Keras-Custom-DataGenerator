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

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

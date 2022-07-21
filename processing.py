import os
import numpy as np
import shutil
import shutil


def Folder_split(root_folder, list_labels ):
  # split flow labels
  for label in list_labels:
    os.makedirs(root_folder + "/train/" + label )  
    os.makedirs(root_folder + "/val/" + label ) 
    os.makedirs(root_folder + "/test/" + label ) 

    src = root_folder + "/" + label

    allPathImages = os.listdir(src)
    np.random.shuffle(allPathImages)

    train_paths, val_paths, test_paths = np.split(np.array(allPathImages), [int(len(allPathImages)*0.7) , int(len(allPathImages)*0.85)]) # split data

    train_paths = [src + "/" + name for name in train_paths.tolist()]
    val_paths = [src + "/" + name for name in val_paths.tolist()]
    test_paths = [src + "/" + name for name in test_paths.tolist()]

# save data
    for name in train_paths:
      shutil.copy(name, root_folder + "/train/" +  label )
      
    for name in val_paths:
      shutil.copy(name, root_folder + "/val/" +  label )

    for name in test_paths:
      shutil.copy(name, root_folder + "/test/" +  label )

  return root_folder + "/train" , root_folder + "/val" ,root_folder + "/test"
  
def get_images_labels(path):
  images = []
  labels = []
  for label in os.listdir(path):
    for image in os.listdir(path + "/" + label):
      images.append(path + "/" + label + "/" + image)
      labels.append(label)
  return images, labels
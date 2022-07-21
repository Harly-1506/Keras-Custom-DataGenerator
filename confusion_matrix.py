import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np

classes = ['A','B','C', 'D', 'E','F', 'G','H', 'I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','not','space']
           
def Confusion_Matrix(model, x_test, y_test,  labels): 

  y_pred = model.predict(x_test)
  y_pred = np.argmax(y_pred ,axis = 1)
  y_true=np.argmax(y_test,axis = 1)
  a = precision_recall_fscore_support(y_true, y_pred, average='macro')
  cf = confusion_matrix(y_true,y_pred)

  print(f'+ precision = {a[0]:.3f}')
  print(f'+ recall = {a[1]:.3f}')
  print(f'+ f1_score = {a[2]:.3f}')
  plt.figure(figsize = (15,13))

  ax = sns.heatmap(cf,fmt="d",annot = True, 
             cmap='Blues')
  ax.xaxis.set_ticklabels(labels)
  ax.yaxis.set_ticklabels(labels)
  plt.xlabel("True_labels")
  plt.ylabel("Predicted labels")
  plt.title("Confusion Matrix")
  plt.show(ax)
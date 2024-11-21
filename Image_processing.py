import cv2
from google.colab.patches import cv2_imshow
import numpy as np
def histogram_e(impath):
  im=cv2.imread(impath,cv2.IMREAD_GRAYSCALE)
  equal=cv2.equalizeHist(im)
  cv2_imshow(im)
  cv2_imshow(equal)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
histogram_e("/content/R.png")

def threshold (impath,threshold_value=128):
  im=cv2.imread(impath,cv2.IMREAD_GRAYSCALE)
  _,thresholded=cv2.threshold(im,threshold_value,255,cv2.THRESH_BINARY)
  cv2_imshow(im)
  cv2_imshow(thresholded)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
threshold("/content/R.png",threshold_value=120)

def edge(impath):
  im=cv2.imread(impath,cv2.IMREAD_GRAYSCALE)
  edges=cv2.Canny(im,100,200)
  cv2_imshow(im)
  cv2_imshow(edges)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
edge("/content/R.png")

def data_augment(impath):
  im=cv2.imread(impath)
  filpped=cv2.flip(im,1)
  cv2_imshow(im)
  cv2_imshow(filpped)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
data_augment("/content/R.png")

def morph(impath):
  im=cv2.imread(impath,cv2.IMREAD_GRAYSCALE)
  kernel=np.ones((5,5),np.uint8)
  erosion=cv2.erode(im,kernel,iterations=1)
  cv2_imshow(im)
  cv2_imshow(erosion)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
morph("/content/R.png")
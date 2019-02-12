import os

class Train:
  BASE_PATH = os.getcwd()
  TRAIN_PATH =  './Dataset/Open Images Dataset v4 (Bounding Boxes)/person_car_phone_train_annotation.txt' # Training data (annotation file)

  NUM_ROIS = 4 # Number of RoIs to process at once.
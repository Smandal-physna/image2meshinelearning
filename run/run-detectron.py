import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import sys
# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


im = cv2.imread(sys.argv[1])
#print(im.shape)
#cc = numpy.zeros(im.shape[0
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

#print('masks:',outputs["instances"].pred_masks.shape)
#print('scores:',outputs["instances"].scores)
#print(im.shape)
#print(outputs["instances"].pred_boxes.area())
#print(pred_boxes)
#for boxes in pred_boxes
#sess = tf.InteractiveSession()
maskw = outputs["instances"].pred_masks[0]*1
#import tensorflow as tf
#nn = tf.keras.backend.eval(maskw)
#nn = maskw.eval()
#imm = np.asarray(im)*nn
bb = maskw.cpu().data.numpy()
imm = np.zeros((im.shape[0],im.shape[1],3))
imm[:,:,0] = bb*im[:,:,0]
imm[:,:,1] = bb*im[:,:,1]
imm[:,:,2] = bb*im[:,:,2]
imm[imm == 0] = 255
bp =  np.asarray(outputs["instances"].pred_boxes[0].tensor.cpu(),dtype='int')[0]
x1,y1,x2,y2  = bp[0],bp[1],bp[2],bp[3]
imm = imm[y1:y2,x1:x2,:]
cv2.imwrite(sys.argv[2],imm)

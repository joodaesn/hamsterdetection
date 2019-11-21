import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import math
import time

input_movie = "hamster.mp4"
add_similar_animals = True
upper_class_of_hamster = ['Animal', 'Squirrel', 'Bat', 'Mouse', 'Mammal']
if add_similar_animals:
    upper_class_of_hamster += ['Fish', 'Bird', 'Sparrow', 'Cat', 'Marine mammal', 'Shark', 'Dog', 'Harbor seal', 'Sea lion', 'Jellyfish', 'Duck']
    

sys.path.append("..")
import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

import label_map_util

import visualization_utils as vis_util

MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'oid_v4_label_map.pbtxt')

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image):
  ops = tf.get_default_graph().get_operations()
  all_tensor_names = {output.name for op in ops for output in op.outputs}
  tensor_dict = {}
  for key in [
      'num_detections', 'detection_boxes', 'detection_scores',
      'detection_classes', 'detection_masks'
  ]:
    tensor_name = key + ':0'
    if tensor_name in all_tensor_names:
      tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
          tensor_name)
  if 'detection_masks' in tensor_dict:
    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
        detection_masks, detection_boxes, image.shape[1], image.shape[2])
    detection_masks_reframed = tf.cast(
        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
    tensor_dict['detection_masks'] = tf.expand_dims(
        detection_masks_reframed, 0)
  image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

  # Run inference
  output_dict = sess.run(tensor_dict,
                         feed_dict={image_tensor: image})

  output_dict['num_detections'] = int(output_dict['num_detections'][0])
  output_dict['detection_classes'] = output_dict[
      'detection_classes'][0].astype(np.int64)
  output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
  output_dict['detection_scores'] = output_dict['detection_scores'][0]
  if 'detection_masks' in output_dict:
    output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict





import cv2

capture = cv2.VideoCapture(0)
capture.open(0)
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=100)

#lst = []
for i in range(0):
    capture.read()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.85

prevTime = 0 #이전 시간을 저장할 변수
loc = []#햄스터경로의 좌표를저장할 리스트입니다

with detection_graph.as_default():
  with tf.Session(config=config) as sess:
    while True:  
          ret, frame = capture.read()
          image_np = frame
          image_np_expanded = np.expand_dims(image_np, axis=0)
          
          fgmask = fgbg.apply(frame)#움직임감지알고리즘(BackgroundSubtractorMOG2) 을 적용한다.
          nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)#BackgroundSubtractorMOG2 알고리즘에따라추론한다.
          
          
          output_dict = run_inference_for_single_image(image_np_expanded)## 딥러닝모델로  추론한다.
          img, rtn_names, rtn_axis = vis_util.visualize_boxes_and_labels_on_image_array(## 인식된위치에 상자를그려준다,
              image_np,
              output_dict['detection_boxes'],
              output_dict['detection_classes'],
              output_dict['detection_scores'],
              category_index,
              instance_masks=output_dict.get('detection_masks'),
              use_normalized_coordinates=True,
              line_thickness=8,
              output_lst = upper_class_of_hamster)
          
          x_axises = []
          y_axises = []
          
          for rtn in rtn_axis:
              x_axises.append(int(frame.shape[1] * (rtn[1] + rtn[3])/2.0)) #딥러닝 모델에서인식된 햄스터좌표를 절대좌표에서 상대좌표로바꾼다
              y_axises.append(int(frame.shape[0] * (rtn[0] + rtn[2])/2.0))
          
          
          ###BackgroundSubtractorMOG2 알고리즘결과출력
          areas = []
          circles = []
          rectangles = []
          for index, centroid in enumerate(centroids):
                if stats[index][0] == 0 and stats[index][1] == 0:
                    continue
                if np.any(np.isnan(centroid)):
                    continue
            
            
                x, y, width, height, area = stats[index]
                centerX, centerY = int(centroid[0]), int(centroid[1])
            
                if area > 300:
                    areas.append(area)
                    circles.append((centerX, centerY))
                    rectangles.append([(x, y), (x + width, y + height)])
                    
                    
          x_axis = -1
          y_axis = -1
          for i in range(len(areas)):
                if areas[i] == max(areas):#움직임이가장넓은 범위에서 인식된 지점만 출력한다.
                    x_axis = circles[i][0]
                    y_axis = circles[i][1]
                    cv2.circle(image_np, circles[i], 1, (0, 255, 0), 2)
                    cv2.rectangle(image_np, rectangles[i][0], rectangles[i][1], (0, 0, 255))
          if x_axis != -1 and y_axis != -1:
              x_axises.append(x_axis)
              y_axises.append(y_axis)
            
          if len(loc) == 0 and len(x_axises) == 0:
                _=0
          elif len(loc) == 0 and len(x_axises) != 0:
                loc.append([x_axises[0], y_axises[0]])
          elif len(loc) > 0 and len(x_axises) == 0:
                loc.append([loc[-1][0], loc[-1][1]])
          elif len(loc) > 0 and len(x_axises) != 0:
                tmp = 999999999
                x_tmp = -1
                y_tmp = -1
                for i in range(len(x_axises)):
                    dist = math.sqrt((loc[-1][0] - x_axises[i]) ** 2 + (loc[-1][1] - y_axises[i]) ** 2)
                    if dist < tmp:
                        tmp = dist
                        x_tmp = x_axises[i]
                        y_tmp = y_axises[i]
                
                    
                    
                if math.sqrt((loc[-1][0] - x_tmp) ** 2 + (loc[-1][1] - y_tmp) ** 2) < math.sqrt(frame.shape[0] ** 2 + frame.shape[1] ** 2) // 4:
                    loc.append([x_tmp, y_tmp])
                else:
                    loc.append([loc[-1][0], loc[-1][1]])
            
          if len(loc) > 10:
                avg_vel = math.sqrt((loc[-1][0] - loc[-11][0]) ** 2 + (loc[-1][1] - loc[-11][1]) ** 2)
                
          else:
                avg_vel = -1
                
          cv2.putText(frame, 'velocity : ' + str(int(avg_vel)) + "pixels / frame", (10,50), 0 , 1, (255,255,255), 2)#속도를계산합니다.

        
            
          if len(loc) > 2 :
                for i in range(1, len(loc)):
                    cv2.line(frame, (loc[i-1][0], loc[i-1][1]), (loc[i][0], loc[i][1]), (255, 0, 0), 3)

          # 현재 시간 가져오기 (초단위로 가져옴)
          #curTime = time.time()

          # 현재 시간에서 이전 시간을 빼면?
          # 한번 돌아온 시간!!
          #sec = curTime - prevTime
          # 이전 시간을 현재시간으로 다시 저장시킴
          #prevTime = curTime

          # 프레임 계산 한바퀴 돌아온 시간을 1초로 나누면 된다.
          # 1 / time per frame
          #fps = 1 / (sec)

          # 디버그 메시지로 확인해보기
          #print ("Time {0} ", format(sec))
          #print ("Estimated fps {0} ", format(fps))

          # 프레임 수를 문자열에 저장
          #string = "FPS : %0.1f" % fps

          # 표시
          #cv2.putText(frame, string, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))


          cv2.imshow('mask',fgmask)
          cv2.imshow("VideoFrame", image_np)
          
          
          if cv2.waitKey(33) > 0: 
              break


## 위치값을저장합니다.좌표는 좌측 상단이원점, 가로축은x축, 세로축은y축
f = open("day6.csv", 'w')
f.write("x,y\n")
for line in loc:
    f.write(str(line[0]) + "," + str(line[1]) + "\n")

f.close()
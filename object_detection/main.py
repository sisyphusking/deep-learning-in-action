import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

PATH_TO_CKPT = './data/frozen_inference_graph.pb'
PATH_TO_LABELS = './data/mscoco_label_map.pbtxt'
NUM_CLASSES = 90

detection_graph = tf.Graph()
# 加载模型
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        od_graph_def.ParseFromString(fid.read())
        tf.import_graph_def(od_graph_def, name='')
# 载入coco数据集标签文件
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# 将图片转换为ndarray数组的形式
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


TEST_IMAGE_PATHS = ['images/image1.jpg', 'images/image2.jpg']

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # 获取模型中的tensor
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # boxes用来显示识别结果
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # score代表识别出的物体与标签匹配的相似程度
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        for image_path in TEST_IMAGE_PATHS:
            image = Image.open(image_path)
            # 将图片转化成numpy格式
            image_np = load_image_into_numpy_array(image)
            # 将图片扩展一维，最后进入神经网络的图片格式应该是[1,?,?,3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # 开始检测
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # 可视化结果
            vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes),
                                                               np.squeeze(classes).astype(np.int32), np.squeeze(scores),
                                                               category_index, use_normalized_coordinates=True,
                                                               line_thickness=8)
            plt.figure(figsize=[12, 8])
            plt.imshow(image_np)
            plt.show()


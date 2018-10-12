import matplotlib.pyplot as plt
import tensorflow as tf

# Image annotation
image_raw_data_jpg = tf.gfile.FastGFile('./images/catdog.jpg', "rb").read()

with tf.Session() as sess:
    img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg)
    img_data_jpg = tf.expand_dims(tf.image.convert_image_dtype(img_data_jpg, tf.float32), 0)

    boxes = tf.constant([[[0.08, 0.08, 0.91, 0.5]]])
    result = tf.image.draw_bounding_boxes(img_data_jpg, boxes)
    """
    For example, if an image is 100 x 200 pixels (height x width) and the bounding box is [0.1, 0.2, 0.5, 0.9],
    the upper-left and bottom-right coordinates of the bounding box will be (40, 10) to (180, 50) (in (x,y) coordinates)
    """
    plt.figure(1)
    plt.imshow(result[0].eval())
    plt.savefig('./images/label_dog.jpg')
    plt.show()

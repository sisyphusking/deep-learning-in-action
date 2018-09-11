CONTENT_IMG = './images/content.png'
STYLE_IMG = './images/style.jpg'

IMAGE_W = 800
IMAGE_H = 600
COLOR_C = 3

NOISE_RATIO = 0.7
BETA = 5
ALPHA = 100
# 预训练模型下载地址：http://www.vlfeat.org/matconvnet/pretrained/
VGG_MODEL = './data/imagenet-vgg-verydeep-19.mat'

STYLE_LAYERS = [('conv1_1', 0.5), ('conv2_1', 1.0), ('conv3_1', 1.5), ('conv4_1', 3.0), ('conv5_1', 4.0)]

OUTPUT_DIR = "./output/"

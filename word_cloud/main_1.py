from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import jieba


'''
WordCloud()可选的参数：

font_path：可用于指定字体路径，包括otf和ttf
width：词云的宽度，默认为400
height：词云的高度，默认为200
mask：蒙版，可用于定制词云的形状
min_font_size：最小字号，默认为4
max_font_size：最大字号，默认为词云的高度
max_words：词的最大数量，默认为200
stopwords：将被忽略的停用词，如果不指定则使用默认的停用词词库
background_color：背景颜色，默认为black
mode：默认为RGB模式，如果为RGBA模式且background_color设为None，则背景将透明

'''


# 打开文本
text = open('./data/xyj.txt', encoding='utf-8').read()

# 中文分词
text = " ".join(jieba.cut(text))

# 生成对象
# mask = np.array(Image.open('./images/demo.png'))

wc = WordCloud(font_path='./data/Hiragino.ttf', mode="RGBA", background_color=None).generate(text)

# 从图片中生成颜色
# image_colors = ImageColorGenerator(mask)
# wc.recolor(color_func=image_colors)

plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

wc.to_file("./images/wc_1.png")

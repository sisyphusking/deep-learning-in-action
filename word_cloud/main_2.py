from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import jieba
import random

'''
蒙版，生成指定图片样式的词云
'''

# 打开文本
text = open('./data/xyj.txt', encoding='utf-8').read()

# 中文分词
text = " ".join(jieba.cut(text))

# 生成对象
mask = np.array(Image.open('./images/color_mask.png'))

wc = WordCloud(mask=mask, font_path='./data/Hiragino.ttf', mode="RGBA", background_color=None).generate(text)

# 从图片中生成颜色
image_colors = ImageColorGenerator(mask)
wc.recolor(color_func=image_colors)  # 颜色函数可以自定义，设置color_func=random_color即可


# 颜色函数
def random_color(word, font_size, position, orientation, font_path, random_state):
    s = 'hsl(0, %d%%, %d%%)' % (random.randint(60, 80), random.randint(60, 80))
    print(s)
    return s


# 显示词云
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

# 保存文件
wc.to_file("./images/wc_2.png")

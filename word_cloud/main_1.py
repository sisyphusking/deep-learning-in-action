from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import jieba


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

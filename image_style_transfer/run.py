from image_style_transfer.config import *
from image_style_transfer.model import *

MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))


def the_current_time():
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time()))))


def load_image(path):
    image = scipy.misc.imread(path)
    image = scipy.misc.imresize(image, (IMAGE_H, IMAGE_W))
    image = np.reshape(image, ((1, ) + image.shape))
    image = image - MEAN_VALUES
    return image


def save_image(path, image):
    image = image + MEAN_VALUES
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)


the_current_time()
if __name__ == '__main__':

    with tf.Session() as sess:
        content_image = load_image(CONTENT_IMG)
        style_image = load_image(STYLE_IMG)
        model = load_vgg_model(VGG_MODEL)

        # 随机生成的图片
        input_image = generate_noise_image(content_image)
        sess.run(tf.global_variables_initializer())

        # 计算内容图片和随机图片的误差
        sess.run(model['input'].assign(content_image))
        content_loss = content_loss_func(sess, model)

        # 计算风格图片和随机图片的误差
        sess.run(model['input'].assign(style_image))
        style_loss = style_loss_func(sess, model)

        # 损失函数
        total_loss = BETA * content_loss + ALPHA * style_loss
        optimizer = tf.train.AdamOptimizer(2.0)
        train = optimizer.minimize(total_loss)

        # 随机图片作为输入
        sess.run(tf.global_variables_initializer())
        sess.run(model['input'].assign(input_image))

        ITERATIONS = 2000
        for i in range(ITERATIONS):
            sess.run(train)
            if i % 100 == 0:
                output_image = sess.run(model['input'])
                the_current_time()
                print('Iteration %d' % i)
                print('Cost: ', sess.run(total_loss))

                save_image(os.path.join(OUTPUT_DIR, 'output_%d.jpg' % i), output_image)



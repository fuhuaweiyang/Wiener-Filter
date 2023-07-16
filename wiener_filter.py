import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal.windows import gaussian
from skimage.color import rgb2gray

#
image_snr = -1


def add_white_noise(image, noise_amplitude):
    # numpy.random.normal(loc=0.0, scale=1.0, size=None)
    # loc：表示分布的均值（期望值）。
    # scale：表示分布的标准差。
    # size：表示要生成的随机数的形状，可以是一个整数，或表示形状的元组。
    noise = np.random.normal(0, noise_amplitude, image.shape)
    # 将噪声叠加在输入图像上
    noisy_image = image + noise
    # 将超出灰度范围（0-255）的像素值截断到边界范围
    noisy_image = np.clip(noisy_image, 0, 255)
    # 转换图像数据类型为无符号8位整数
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image


def estimate_noise_amplitude(image):    # 将图像转换为灰度图像
    if len(image.shape) == 3:
        # 将输入图像转换为灰度图像
        gray_image = rgb2gray(image)
        # 将浮点型灰度图像转换为8位无符号整数类型的灰度图像
        gray_image = (gray_image * 255).astype(np.uint8)
    else:
        gray_image = image
    # 计算梯度幅度
    # 使用了sobel函数对灰度图像 gray_image 进行水平方向上的边缘检测。
    # gray_image：输入的灰度图像。
    # cv2.CV_64F：指定输出图像的深度为64位浮点数。
    # 1、0：表示要计算的导数的阶数。在这种情况下，使用一阶导数，其中1表示水平方向上的导数，0表示垂直方向上的导数。
    # ksize = 3：指定Sobel算子的内核大小。这里使用大小为3的内核来计算导数
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    # 通过计算水平和垂直梯度的平方和的平方根，得到图像的梯度幅度。
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    # 使用OTSU阈值分割法得到二值图像,也就是黑白图像，方便确定平坦区域的位置
    # gradient_magnitude.astype(np.uint8)：将梯度幅度图像转换为8位无符号整数类型。这是由于 cv2.threshold() 函数要求输入图像为8位。
    np_uint8 = gradient_magnitude.astype(np.uint8)
    # 0：指定用于二值化的阈值。在这种情况下，阈值设置为0，因为 cv2.THRESH_OTSU 将自动选择最佳阈值。
    # 255：将二值图像中大于阈值的像素设置为255（白色）。
    # cv2.THRESH_BINARY + cv2.THRESH_OTSU：指定二值化的方法。cv2.THRESH_BINARY 表示使用固定阈值二值化方法，而 cv2.THRESH_OTSU 表示使用Otsu自适应阈值方法。
    _, binary_image = cv2.threshold(np_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 使用连通组件分析找到最大联通区域（平坦区域），可以识别出图像中的不同连通区域，并为每个连通区域分配唯一的标签。
    # binary_image：输入的二值图像，其中只包含0和255两个像素值。
    # connectivity=8：指定连接性，即在进行连通组件分析时，考虑相邻像素的方式。这里设置为8表示考虑每个像素的8个相邻像素。
    # num_labels：整数类型，表示连通组件的数量（包括背景）。
    # labels：与 binary_image 相同大小的数组，每个像素被标记为其所属的连通组件编号。
    # stats：一个形状为 (num_labels, 5) 的二维数组，每一行包含了对应连通组件的统计信息，如左上角坐标、宽度、高度和该组件的像素总数等
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    # 寻找具有最大面积的连通组件的标签
    # np.argmax()：返回数组中最大值的索引。stats[1:, cv2.CC_STAT_AREA]：从 stats 数组中提取所有连通组件（除了背景）的面积信息。
    # 注意，这里加一不能省略，因为 stats 数组的索引是从0开始的，而实际的连通组件标签从1开始
    max_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    # 创建一个掩码图像，用于标记灰度图像中最大连通区域（平坦区域）的位置。
    # (labels == max_label)：创建一个布尔数组，其形状与 labels 相同，其中与最大标签相等的像素位置为True，其他位置为False。
    # .astype(np.uint8)：将布尔数组转换为无符号8位整数类型，即将True转换为1，False转换为0。
    # * 255：将数组中的非零值乘以255，将值为1的像素设置为255（白色），其他像素保持为0（黑色）。
    flat_region_mask = (labels == max_label).astype(np.uint8) * 255
    # 使用cv2.bitwise_and() 函数来对灰度图像 gray_image 和掩码图像 flat_region_mask 进行按位与操作。
    # 保留两个输入图像中对应位置像素值都为非零的像素，并将其他位置的像素值设为零。
    # 得到的结果 flat_region 将只包含灰度图像中最大连通区域的像素值，其它地方都是黑色。
    flat_region = cv2.bitwise_and(gray_image, flat_region_mask)
    # 计算灰度直方图
    hist = cv2.calcHist([flat_region], [0], None, [256], [0, 256])
    # 估计白噪声的幅度
    noise_amplitude = np.std(flat_region) - 20
    return noise_amplitude


def wiener(image):
    # 此处的image为np.ndarray
    # 定义一个全局变量方便之后输出的时候显示不需要在求一遍信噪比
    global image_snr
    # 拷贝
    copy_image = np.copy(image)
    # 对copy_image进行二维快速傅里叶变换
    dummy = fft2(copy_image)
    # noise_amplitude为估计的噪音幅度
    noise_amplitude = estimate_noise_amplitude(image)
    # 求高斯核
    size = 2*math.sqrt(noise_amplitude)
    size = math.ceil(size)
    # 生成一个大小为size的一维高斯滤波器向量，其中高斯函数的标准差size/2
    gua = gaussian(size, size/2)
    # 对生成的一维向量进行形状重塑操作，将其变成一个列向量，形状为 (16, 1)。
    vector = gua.reshape(size, 1)
    # 计算向量 vector 与其转置的矩阵乘积。
    two_dim = np.dot(vector, vector.transpose())
    # 将 two_dim 的每个元素除以其所有元素的和，从而对矩阵进行归一化处理。其中每个元素表示其在整个矩阵中所占的比例。
    ratio = two_dim / np.sum(two_dim)
    # 使用二维快速傅里叶变换（FFT）将核函数 ratio 转换为与输入图像 image 具有相同大小的频域表示。
    kernel = fft2(ratio, image.shape)
    # 计算信号幅度,pixel_sum = 0为初始值
    pixel_sum = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_sum += image[i, j]
    # num_pixels为总的像素个数
    num_pixels = image.shape[0] * image.shape[1]
    # 假定图像的信号幅度等于总的灰度值除以总的像素数量
    signal_amplitude = pixel_sum / num_pixels
    # 计算信噪比
    snr = signal_amplitude / noise_amplitude
    image_snr = snr
    # 使用conj()函数计算了kernel数组的共轭复数conjugate_plural
    conjugate_plural = np.conj(kernel)
    # np.abs(kernel)计算kernel的模，
    wiener1 = conjugate_plural / (np.abs(kernel) ** 2 + 1 / snr)
    # 矩阵相乘
    inverse_result = dummy * wiener1
    # 矩阵相乘之后的结果不一定为正，但是图像要求的像素大小为0~255
    non_positive_result = ifft2(inverse_result)
    # 所以要用np.abs取绝对值
    result = np.abs(non_positive_result)
    return result


def main(img):
    # 如果输入的img为彩色图像，要把其转化为灰度图像
    if len(img.shape) == 3:
        # 将输入图像转换为灰度图像
        img = rgb2gray(img)
        # 将浮点型灰度图像转换为8位无符号整数类型的灰度图像
        img = (img * 255).astype(np.uint8)
    # 对输入的图像进行加白噪音处理
    noise_image = add_white_noise(img, 60)
    # 估算白噪音幅度，这个值越大白噪音越大
    noise_size = estimate_noise_amplitude(noise_image)
    print("noise_size is "+ str(noise_size))
    # 对噪音图像进行维纳滤波
    wiener_image = wiener(noise_image)
    # 将 wiener_image 数组中的元素限制在指定的范围内。具体来说，它将所有小于 0 的元素置为 0，将所有大于 255 的元素置为 255。
    np.clip(wiener_image, 0, 255, out=wiener_image)
    # 直接调用库函数进行双边滤波，与维纳滤波形成对比
    bilateral_image = cv2.bilateralFilter(noise_image, 9, 75, 75)
    # 定义一个包含2行2列子图的Matplotlib图像
    _, ax = plt.subplots(2, 2, figsize=(10, 10))
    # 输出原图像
    # 将 BGR 图像转换为 RGB 图像
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax[0][0].imshow(rgb_image)
    ax[0][0].title.set_text("origin")
    # 输出噪音图像
    # 将 BGR 图像转换为 RGB 图像
    rgb_noise_image = cv2.cvtColor(noise_image, cv2.COLOR_BGR2RGB)
    ax[0][1].imshow(rgb_noise_image)
    ax[0][1].title.set_text("noise_image noise_size:"+str(round(noise_size,2)))
    # 输出维纳滤波后的图像
    # cv2.convertScaleAbs() 用于将图像数组按比例缩放并转换为绝对值，将图像从一种颜色空间转换为另一种颜色空间
    # 不调用这个函数会使得由于输入图像的深度不受支持而引起的报错
    rgb_wiener_image = cv2.convertScaleAbs(wiener_image)
    rgb_wiener_image = cv2.cvtColor(rgb_wiener_image, cv2.COLOR_GRAY2BGR)
    ax[1][0].imshow(rgb_wiener_image)
    ax[1][0].title.set_text("wiener_image")
    # 输出双边滤波图像作为对比
    rgb_bilateral_image = cv2.convertScaleAbs(bilateral_image)
    rgb_bilateral_image = cv2.cvtColor(rgb_bilateral_image, cv2.COLOR_GRAY2BGR)
    ax[1][1].imshow(rgb_bilateral_image)
    ax[1][1].title.set_text("bilateral_image")
    plt.show()
    # 绘制灰度直方图
    plt.hist(noise_image.flatten(), bins=256, range=[0, 256])
    plt.title("Grayscale cube chart")
    plt.xlabel("Gray")
    plt.ylabel("Pixel")
    plt.show()

    return wiener_image, noise_image


image = cv2.imread('C:/Users/18133/Desktop/wiener/origin.jpg')
img, noise_img = main(image)
cv2.imwrite("C:/Users/18133/Desktop/wiener/wiener1.jpg", img)
cv2.imwrite("C:/Users/18133/Desktop/wiener/noise1.jpg", noise_img)
print(image_snr)
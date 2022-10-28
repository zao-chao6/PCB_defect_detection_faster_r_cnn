# -*- coding:utf-8 -*-
"""数据增强
   1. 翻转变换 flip
   2. 图片裁剪 crop
   3. 色彩抖动 color jittering
   4. 平移变换 shift
   5. 尺度变换 scale
   6. 对比度变换 contrast
   7. 噪声扰动 noise
   8. 旋转变换/反射变换 Rotation/reflection
   9.直方图增强
   10.拉普拉斯算子
   11.对数变换
   12.伽马变换
   13.限制对比度自适应直方图均衡化CLAHE
   14.retinex SSR
   15.retinex MMR
   16.

"""

import logging
import os
import random
import threading
import time
from dataclasses import dataclass
from distutils.log import error

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFile


# 图片裁剪
def read_path(file_pathname):

    for filename in os.listdir(file_pathname):
        # print(filename)
        img_filename = os.path.join(file_pathname, filename)  #将图片路径与图片名进行拼接
        
        img = cv2.imread(img_filename)       #img_path为图片所在路径
        crop_img = img[0:3585,0:3629]    #x0,y0为裁剪区域左上坐标；x1,y1为裁剪区域右下坐标（y0：y1，x0:x1）

        #####save figure
        # cv2.imwrite(r'date_set\data_source1'+"/"+filename,crop_img)
        cv2.imwrite(r'jixing\polarity'+"/"+filename,crop_img)


logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataAugmentation:
    """
    包含数据增强的八种方式
    """

    def __init__(self):
        pass
 
    @staticmethod
    def openImage(image):
        img=cv2.imread(image)
        return img

    @staticmethod
    def randomRotation(image, center=None, scale=1.0):    #mode=Image.BICUBIC
        """
         对图像进行随机任意角度(0~360度)旋转
        :return: 旋转转之后的图像
        """
        random_angle = np.random.randint(-180, 180)
        (h, w) = image.shape[:2]
        # If no rotation center is specified, the center of the image is set as the rotation center
        if center is None:
            center = (w / 2, h / 2)
        m = cv2.getRotationMatrix2D(center, random_angle, scale)  #center：旋转中心坐标.angle：旋转角度，负号为逆时针，正号为顺时针.scale：缩放比例，1为等比例缩放
        rotated = cv2.warpAffine(image, m, (w, h))
        return rotated
    
    @staticmethod
    def transpose(image):
        """
        水平垂直翻转
        :return: 旋转转之后的图像
        """
        random_angle = np.random.randint(-2, 2)  #取[-1,1]的随机整数
        img_filp=cv2.flip(image,random_angle)
        return img_filp
    
    '''噪声抖动'''

    @staticmethod
    def randomColor(image):
        """
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
        """
        saturation=random.randint(0,1)
        brightness=random.randint(0,1)
        contrast=random.randint(0,1)
        sharpness=random.randint(0,1)
        image=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))  #转化为PIL.Image对象,才能使用ImageEnhance.Brightness(image)
        if random.random() < saturation:
            random_factor = np.random.randint(0, 31) / 10.  # 随机因子
            image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        if random.random() < brightness:
            random_factor = np.random.randint(10, 21) / 10.  # 随机因子
            image = ImageEnhance.Brightness(image).enhance(random_factor)  # 调整图像的亮度
        if random.random() < contrast:
            random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
            image = ImageEnhance.Contrast(image).enhance(random_factor)  # 调整图像对比度
        if random.random() < sharpness:
            random_factor = np.random.randint(0, 31) / 10.  # 随机因子
            image= ImageEnhance.Sharpness(image).enhance(random_factor)  # 调整图像锐度
        image=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)    #转换为cv格式
        return image
    
    @staticmethod
    def randomGaussian(image, mean=0.2, sigma=0.04):  
        """
         对图像进行高斯噪声处理
        mean:设置高斯分布的均值和方差
        sigma:设置高斯分布的标准差,sigma值越大，噪声越多
        
        返回:
        gaussian_out : 噪声处理后的图片
        """
         # 将图片灰度标准化
        img = image / 255
        # 产生高斯 noise
        noise = np.random.normal(mean, sigma, img.shape)
        # 将噪声和图片叠加
        gaussian_out = img + noise
        # 将超过 1 的置 1，低于 0 的置 0
        gaussian_out = np.clip(gaussian_out, 0, 1)
        # 将图片灰度范围的恢复为 0-255
        gaussian_out = np.uint8(gaussian_out*255)
        # 将噪声范围搞为 0-255
        # noise = np.uint8(noise*255)
        return gaussian_out

    @staticmethod
    def Pepper_noise(image):
        '''
        椒盐噪声
        '''
        #设置添加椒盐噪声的数目比例
        s_vs_p = 0.04
        #设置添加噪声图像像素的数目
        amount =0.03
        noisy_img = np.copy(image)
        #添加salt噪声
        num_salt = np.ceil(amount * image.size * s_vs_p)
        #设置添加噪声的坐标位置
        coords = [np.random.randint(0,i - 1, int(num_salt)) for i in image.shape]
        noisy_img[tuple(coords)] = 255
        #添加pepper噪声
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        #设置添加噪声的坐标位置
        coords = [np.random.randint(0,i - 1, int(num_pepper)) for i in image.shape]
        noisy_img[tuple (coords)] = 0
        return noisy_img
 
    @staticmethod
    def Poisson_noise(image):
        '''泊松噪声'''

        #计算图像像素的分布范围
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        #给图片添加泊松噪声
        noisy_img = np.random.poisson(image * vals) / float(vals)
        return noisy_img

    '''图像增强算法'''

    @staticmethod
    def hist(image):
        '''直方图均衡增强'''
        r, g, b = cv2.split(image)
        r1 = cv2.equalizeHist(r)
        g1 = cv2.equalizeHist(g)
        b1 = cv2.equalizeHist(b)
        image_equal_clo = cv2.merge([r1, g1, b1])
        return image_equal_clo

    @staticmethod
    def laplacian(image):
        '''拉普拉斯算子'''
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image_lap = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        return image_lap

    @staticmethod
    def log(image):
        '''对数变换'''
        image_log = np.uint8(np.log(np.array(image) + 1))
        cv2.normalize(image_log, image_log, 0, 255, cv2.NORM_MINMAX)
        # 转换成8bit图像显示
        cv2.convertScaleAbs(image_log, image_log)
        return image_log

    @staticmethod
    def gamma(image):
        '''伽马变换'''
        fgamma = 0.5    #数值越大，生成的图片越黑
        image_gamma = np.uint8(np.power((np.array(image) / 255.0), fgamma) * 255.0)
        cv2.normalize(image_gamma, image_gamma, 0, 255, cv2.NORM_MINMAX)
        cv2.convertScaleAbs(image_gamma, image_gamma)
        return image_gamma

    @staticmethod
    def clahe(image):
        '''# 限制对比度自适应直方图均衡化CLAHE'''
        b, g, r = cv2.split(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        b = clahe.apply(b)
        g = clahe.apply(g)
        r = clahe.apply(r)
        image_clahe = cv2.merge([b, g, r])
        return image_clahe
    
    def __replaceZeroes(data):
        min_nonzero = min(data[np.nonzero(data)])
        data[data == 0] = min_nonzero
        return data

    def __SSR(src_img, size):

        L_blur = cv2.GaussianBlur(src_img, (size, size), 0)
        img =DataAugmentation.__replaceZeroes(src_img)

        L_blur =DataAugmentation. __replaceZeroes(L_blur)

        dst_Img = cv2.log(img/255.0)
        dst_Lblur = cv2.log(L_blur/255.0)
        dst_IxL = cv2.multiply(dst_Img, dst_Lblur)
        log_R = cv2.subtract(dst_Img, dst_IxL)

        dst_R = cv2.normalize(log_R,None, 0, 255, cv2.NORM_MINMAX)
        log_uint8 = cv2.convertScaleAbs(dst_R)
        return log_uint8

    @staticmethod
    def SSR_image(image):
        '''SSR_image'''
        size = 3
        b_gray, g_gray, r_gray = cv2.split(image)
        b_gray =DataAugmentation.__SSR(b_gray, size)
        g_gray =DataAugmentation.__SSR(g_gray, size)
        r_gray =DataAugmentation.__SSR(r_gray, size)
        result = cv2.merge([b_gray, g_gray, r_gray])
        return result

    # retinex MSR
    def __MSR(img, scales):
        weight = 2 / 3.0
        scales_size = len(scales)
        h, w = img.shape[:2]
        log_R = np.zeros((h, w), dtype=np.float32)

        for i in range(scales_size):
            img =DataAugmentation. __replaceZeroes(img)
            L_blur = cv2.GaussianBlur(img, (scales[i], scales[i]), 0)
            L_blur =DataAugmentation. __replaceZeroes(L_blur)
            dst_Img = cv2.log(img/255.0)
            dst_Lblur = cv2.log(L_blur/255.0)
            dst_Ixl = cv2.multiply(dst_Img, dst_Lblur)
            log_R += weight * cv2.subtract(dst_Img, dst_Ixl)

        dst_R = cv2.normalize(log_R,None, 0, 255, cv2.NORM_MINMAX)
        log_uint8 = cv2.convertScaleAbs(dst_R)
        return log_uint8

    @staticmethod   
    def MSR_image(image):
        '''MSR_image'''
        scales = [15, 101, 301]  # [3,5,9]
        b_gray, g_gray, r_gray = cv2.split(image)
        b_gray =DataAugmentation.__MSR(b_gray, scales)
        g_gray =DataAugmentation. __MSR(g_gray, scales)
        r_gray =DataAugmentation. __MSR(r_gray, scales)
        result = cv2.merge([b_gray, g_gray, r_gray])
        return result


def imageOps(func_name, image1,  img_des_path, img_file_name, times=1):   #times=1每种方式，每张图片运行一次
    funcMap = {#"randomRotation": DataAugmentation.randomRotation, 
                "randomcolor": DataAugmentation.randomColor,"transpose": DataAugmentation.transpose, 
               "randomGaussian": DataAugmentation.randomGaussian, "pepper_noise": DataAugmentation.Pepper_noise,
               "Poisson_noise": DataAugmentation.Poisson_noise, "hist": DataAugmentation.hist, 
               "laplacian": DataAugmentation.laplacian,"log": DataAugmentation.log,
                "gamma": DataAugmentation.gamma, "clahe": DataAugmentation.clahe, 
                "SSR_image": DataAugmentation.SSR_image, "MSR_image": DataAugmentation.MSR_image
               }
    if funcMap.get(func_name) is None:
        logger.error("%s is not exist", func_name)
        return -1
 
    for _i in range(0, times, 1):
        new_image = funcMap[func_name](image1)   #经过变化后的图片
        # print('new_image：',new_image)
        # path=os.path.join(img_des_path, func_name + str(_i) + img_file_name)  #存图的新名字
        path=os.path.join(img_des_path, img_file_name)
        # print('new_filename：',path)
        cv2.imwrite (path,new_image) 


# opsList = {"transpose",'randomcolor',"gamma","MSR_image","pepper_noise","hist","log","clahe",'randomGaussian',
#             'Poisson_noise','laplacian','SSR_image'}
opsList = {"clahe"}   #clahe图像增强效果较好
 
def threadOPS(img_path, new_img_path):
    """
    多线程处理事务
    :param src_path: 源文件
    :param des_path: 存放文件
    :return:
    """
    #img path 
    if os.path.isdir(img_path):
        img_names = os.listdir(img_path)
        # print('img_names值为：',img_names)
    else:
        img_names = [img_path]
        # print('img_names1值为：',img_names)
 
    img_num = 0
 
    #img num
    for img_name in img_names:
        tmp_img_name = os.path.join(img_path, img_name)
        if os.path.isdir(tmp_img_name):
            print('contain file folder')
            exit()
        else:
            img_num = img_num + 1
            num = img_num
            # print("num数值为：",num )
 
 
    for i in range(num):
        img_name = img_names[i]
        # print("img_name:",img_name)
        tmp_img_name = os.path.join(img_path, img_name)
        # 读取文件并进行操作
        image1 = DataAugmentation.openImage(tmp_img_name)
        # print("读取文件image：",image1)
        
        # threadImage =[0] * 12   #定义一个元组，其长度为12.
        threadImage ={}           #定义为空字典类型。用来装线程结果信息
        _index = 0
        for ops_name in opsList:
            # print("ops_name:",ops_name)
            #创建一个新线程
            threadImage[_index] = threading.Thread(target=imageOps,
                                                    args=(ops_name, image1, new_img_path,img_name))
            print('threadImage[{}]:{}'.format(_index,threadImage))
            threadImage[_index].start()   #启动线程
            _index += 1      #显示每个线程的起停位置
            time.sleep(0.2)  #线程执行的时间
 
 
if __name__ == '__main__':
    threadOPS(#r"F:\Desktop\PCB_code\date_set\1shujuchuli",
              #r"F:\Desktop\PCB_code\date_set\2shujucunfang"
              r'F:\Desktop\PCB_code\data_set1\data_shiyan',
              r'F:\Desktop\PCB_code\data_set1\data_shiyan_kuochong')

    # read_path(r'F:\Desktop\PCB_code\data_set1\data_shiyan')   #图片裁剪


    '''
    路径问题：
    关于上述路径中，\table\name\rain中的\t,\n,\r都易被识别为转义字符。
    解决的办法主要由以下三种：
    #1
    path=r"C:\data\table\name\rain"
    #前面加r表示不转义

    #2
    path="C:\\data\\table\\name\\rain"
    #用\\代替\

    #3
    path="C:/data/table/name/rain"
    #用\代替/

    '''


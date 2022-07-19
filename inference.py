'''
本文件是整个模型的入口。请从后往前浏览，找到main，那里是程序入口。
'''
# 下面的是本地编写的代码
from seg import U2NETP
from GeoTr import GeoTr
from IllTr import IllTr
from inference_ill import rec_ill

# 下面的是导入的包
import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.io as io
import numpy as np      # NunPy
import cv2              # OpenCV
# 下面这个glob模块用来查找文件，似乎没用过，我把它注释起来了
# import glob
import os
from PIL import Image   # 用于进行图片的读取、保存等操作。对图片本身的运算则不使用这个库
import argparse         # 用于解析命令行参数
# 关掉程序的警告信息
import warnings
warnings.filterwarnings('ignore')

# 几何校正的封装类，注意这里的几何校正也包括识别文档边界的部分
class GeoTr_Seg(nn.Module):
    def __init__(self):
        super(GeoTr_Seg, self).__init__()
        self.msk = U2NETP(3, 1)                 # 识别文档边界。这里实例化了U2NETP类。该类的定义见seg.py
        self.GeoTr = GeoTr(num_attn_layers=6)   # 几何矫正。GeoTr类的定义见GeoTr.py
        
    def forward(self, x):
        msk, _1,_2,_3,_4,_5,_6 = self.msk(x)    # _1到_6都是无用值，直接丢弃。只需要msk
        msk = (msk > 0.5).float()
        x = msk * x

        bm = self.GeoTr(x)
        bm = (2 * (bm / 286.8) - 1) * 0.99
        
        return bm
        
# 加载预训练模型，至于加载的是几何还是光照，由参数model决定
def reload_model(model, path=""):   # path默认值为空，表示没有预训练
    if not bool(path):      # path为空时原封不动地返回model
        return model
    else:                   # 否则，加载模型
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cuda:0')
        print(len(pretrained_dict.keys()))
        # 下面大括号中的是一个字典推导式，生成一个字典
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model
        
# 加载分割文档边界的预训练模型，参考reload_model函数。
def reload_segmodel(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cuda:0')
        print(len(pretrained_dict.keys()))
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict}
        print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model
        
# 控制整个修复过程的函数
def rec(opt):
    # 下面这行是作者注释起来的，可以看到他使用的torch版本是1.5.1
    # print(torch.__version__) # 1.5.1
    img_list = os.listdir(opt.distorted_path)  # 扭曲文档图片的列表(list)

    # 下面两个if是判断输出文件夹是否存在，若不存在则创建
    if not os.path.exists(opt.gsave_path):  # create save path
        os.mkdir(opt.gsave_path)
    if not os.path.exists(opt.isave_path):  # create save path
        os.mkdir(opt.isave_path)
    
    # 加载文档边界分割模型和几何矫正模型。二者是捆绑在一起的。
    GeoTr_Seg_model = GeoTr_Seg().cuda()    # cuda()表示把数据调入GPU运算，下同。
    # 加载文档边界分割模型的参数
    reload_segmodel(GeoTr_Seg_model.msk, opt.Seg_path)
    # 加载几何矫正预训练模型的参数
    reload_model(GeoTr_Seg_model.GeoTr, opt.GeoTr_path)
    
    # 加载光照修复模型
    IllTr_model = IllTr().cuda()
    # 加载光照修复预训练模型的参数
    reload_model(IllTr_model, opt.IllTr_path)
    
    # To eval mode
    # 不启用 Batch Normalization 和 Dropout，注意和python内置的eval不是同一个东西
    GeoTr_Seg_model.eval()
    IllTr_model.eval()
  
    # 程序主循环，逐个处理扭曲图像
    for img_path in img_list:
        name = img_path.split('.')[-2]  # 去掉图片文件的后缀名。[-2]是反向索引。

        img_path = opt.distorted_path + img_path  # 在图片文件前面加上路径，便于读取
        '''
        Image模块来自PIL库。PIL: Python Image Library
        Image.open()返回一个Image对象 该对象可直接转换为numpy.ndarray
        得到的ndarray是一个三维的张量 H * W * 3
        其中H是图片纵向的像素宽度 W是横向的宽度
        Image对象根据其是彩图或灰度图等会有不同的mode属性
        这里的mode = 'RGB'。每个像素有3个通道值 因此张量的第三维有3个元素
        有关:3的含义 我猜测是考虑到RGBA模式的4通道情况，这里直接砍掉(可能的)第4维
        im: image的缩写。
        '''
        im_ori = np.array(Image.open(img_path))[:, :, :3] / 255.    # 255.的.是必要的，用于进行浮点除法
        h, w, _ = im_ori.shape                  # h w 就是上面提到的H W，而_应该是直接扔掉了
        '''
        下面一步将输入图像无条件压缩至288 * 288。这里的im是cv2.Mat类型。
        见原论文3.1节：
        given an image I_D, we first downsample it and get the image I_d, where H_0 = W_0 = 288
        and C_0 = 3 is the number of RGB channels.
        '''
        im = cv2.resize(im_ori, (288, 288))     # im 现在是 288 * 288 * 3
        # 有关numpy.transpose()，见 https://www.cnblogs.com/caizhou520/p/11227986.html
        # 下式将 x, y, z 的顺序调整为 z, x, y (目的是？)
        im = im.transpose(2, 0, 1)              # im 现在是 3 * 288 * 288
        '''
        下面这步将numpy数组格式的im转成torch内置的Tensor格式。cv2.Mat似乎可以直接转np.ndarray ?
        unsqueeze()用于升维。在这里，它在整个张量外层添加一层括号。
        float()指示torch将目标张量的类型设为torch.float32 (32位浮点数)
        '''
        im = torch.from_numpy(im).float().unsqueeze(0)  # im 现在是 1 * 3 * 288 * 288
        
        '''
        下面的with torch.no_grad()是十分常见的写法。
        该代码段中Tensor的计算都不会进行自动求导，节省了显存。
        '''
        with torch.no_grad():
            # 边界分割和几何矫正
            bm = GeoTr_Seg_model(im.cuda())

            bm = bm.cpu()   # 剩下的后处理在 CPU 进行
            bm0 = cv2.resize(bm[0, 0].numpy(), (w, h))  # x flow
            bm1 = cv2.resize(bm[0, 1].numpy(), (w, h))  # y flow
            bm0 = cv2.blur(bm0, (3, 3))
            bm1 = cv2.blur(bm1, (3, 3))
            lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0)  # h * w * 2
            
            '''
            下面这步就是论文3.1节提到的双线性插值
            F: import torch.nn.functional as F
            im_ori即original image之意，表示原始的图像，但实际上也是除以了255，并且砍掉了(可能的)第4维。
            permute: 类似于transpose，交换Tensor的几个维度的顺序。
            align_corners: 不太懂。有篇文章 https://zhuanlan.zhihu.com/p/87572724
            '''
            out = F.grid_sample(torch.from_numpy(im_ori).permute(2,0,1).unsqueeze(0).float(), lbl, align_corners=True)
            '''
            out[0]表示取out的第一个维度，因为图片在预处理中已经变成了 1 * 3 * 288 * 288 的形式。
            原先预处理时除以的255现在要乘回来。
            permute: 同上。此处将out[0]的第1个维度放到第3个维度。
            ((out[0]*255).permute(1, 2, 0).numpy())的结果是一个 H * W * 3 的np.ndarray。注意上面的双线性插值已经改变了H和W，它们不再是288了。
            [:,:,::-1]是针对一个三维张量的索引，前两个维度保持不变，最后一个维度按倒序取值。
            '''
            img_geo = ((out[0]*255).permute(1, 2, 0).numpy())[:,:,::-1].astype(np.uint8)
            cv2.imwrite(opt.gsave_path + name + '_geo' + '.png', img_geo)  # 保存图片
            
            # 光照修复
            if opt.ill_rec:     # 只有在参数中指定进行光照修复时，才执行下面的代码
                ill_savep = opt.isave_path + name + '_ill' + '.png'
                rec_ill(IllTr_model, img_geo, saveRecPath=ill_savep)    # rec_ill函数，见inference_ill.py
        
        print('Done: ', img_path)

# 程序总入口
def main():
    # 下面parser开头的这几行是命令行参数解析，所有参数都给出了默认值。
    parser = argparse.ArgumentParser() 
    parser.add_argument('--distorted_path',  default='./distorted/')    # 存放扭曲图片的源文件夹
    parser.add_argument('--gsave_path',  default='./geo_rec/')          # 存放几何矫正输出图片的文件夹
    parser.add_argument('--isave_path',  default='./ill_rec/')          # 存放光照修复输出图片的文件夹
    parser.add_argument('--Seg_path',  default='./model_pretrained/seg.pth')        # 存放边界分割训练模型的位置
    parser.add_argument('--GeoTr_path',  default='./model_pretrained/geotr.pth')    # 几何训练模型的位置
    parser.add_argument('--IllTr_path',  default='./model_pretrained/illtr.pth')    # 光照训练模型的位置
    parser.add_argument('--ill_rec',  default=False)    # 是否进行光照修复，默认为否
    
    opt = parser.parse_args()

    rec(opt)


if __name__ == '__main__':
    main()

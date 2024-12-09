from model.net import (
    Restormer_Encoder,
    Restormer_Decoder,
    BaseFeatureExtractor,
    DetailFeatureExtractor,
)
import os
import numpy as np
from utils.evaluator import Evaluator
import torch
import torch.nn as nn
from utils.imageUtils import img_save, image_read_cv2
import warnings
import logging
import matplotlib.pyplot as plt 
from PIL import Image 
def normalize_image(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255.0
    return image.astype(np.uint8)
def display_images(data_IR, data_VIS, fi, title_IR="Infrared (IR)", title_VIS="Visible (VI)", title_Fuse="Fused Image"):
    """
    显示红外图像、可见光图像和融合图像。
    
    参数:
    data_IR (Tensor): 红外图像数据
    data_VIS (Tensor): 可见光图像数据
    fi (ndarray): 融合图像
    title_IR (str): 红外图像标题
    title_VIS (str): 可见光图像标题
    title_Fuse (str): 融合图像标题
    """
    # 创建图形和子图
    plt.figure(figsize=(15, 5))

    # 显示红外图像
    plt.subplot(1, 3, 1)
    plt.title(title_IR)
    plt.imshow(data_IR.cpu().numpy().squeeze(), cmap="gray")
    plt.axis("off")

    # 显示可见光图像
    plt.subplot(1, 3, 2)
    plt.title(title_VIS)
    plt.imshow(data_VIS.cpu().numpy().squeeze(), cmap="gray")
    plt.axis("off")

    # 显示融合图像
    plt.subplot(1, 3, 3)
    plt.title(title_Fuse)
    plt.imshow(fi, cmap="gray")
    plt.axis("off")

    # 调整布局并显示图像
    plt.tight_layout()
    plt.show()
def display_fused_image(img_name, data_IR, data_VIS_color, data_VIS, fi, fi_img_color):
    """
    显示红外图像、可见光图像和彩色融合图像。
    
    参数:
    img_name (str): 图像文件名
    data_IR (Tensor): 红外图像数据
    data_VIS_color (PIL Image): 可见光彩色图像
    data_VIS (Tensor): 可见光灰度图像数据
    fi (ndarray): 灰度融合图像
    fi_img_color (ndarray): 彩色融合图像
    """
    
    # 检查融合图像是否正常
    if np.max(fi) == 0:
        print(f"Warning: Fused image {img_name} is completely black!")
        return

    # 显示图像
    _, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 显示红外图像
    axes[0].imshow(np.squeeze(data_IR.cpu().numpy()), cmap='gray')
    axes[0].set_title("Infrared (IR)")
    axes[0].axis('off')

    # 显示可见光图像
    axes[1].imshow(data_VIS_color)
    axes[1].set_title("Visible (VI)")
    axes[1].axis('off')

    # 显示彩色融合图像
    axes[2].imshow(fi_img_color)
    axes[2].set_title("Fused Image (Color)")
    axes[2].axis('off')

    plt.show()

#忽略所有警告信息，确保程序的输出干净。
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#读取深度学习模型
ckpt_path = os.path.join("checkPoints","daf_net.pth")
print(f"{ckpt_path}\n")
#初始化输入输出
print(f"The test result of input:")
test_folder = os.path.join("input")
test_out_folder = os.path.join("output")

device = "cuda" if torch.cuda.is_available() else "cpu"
#加载模型权重
Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
BaseFuseLayer = nn.DataParallel(BaseFeatureExtractor(dim=64, num_heads=8)).to(
    device
    )
DetailFuseLayer = nn.DataParallel(DetailFeatureExtractor(num_layers=1)).to(device)
Encoder.load_state_dict(torch.load(ckpt_path)["DIDF_Encoder"])
Decoder.load_state_dict(torch.load(ckpt_path)["DIDF_Decoder"])
BaseFuseLayer.load_state_dict(torch.load(ckpt_path)["BaseFuseLayer"])
DetailFuseLayer.load_state_dict(torch.load(ckpt_path)["DetailFuseLayer"])
#eval评估模式
Encoder.eval()
Decoder.eval()
BaseFuseLayer.eval()
DetailFuseLayer.eval()

with torch.no_grad():
    for img_name in os.listdir(os.path.join(test_folder, "ir")):

        data_IR = (
            image_read_cv2(os.path.join(test_folder, "ir", img_name), mode="GRAY")[
                np.newaxis, np.newaxis, ...
            ]
            / 255.0
            )
        data_VIS_color = Image.open(os.path.join(test_folder, "vi", img_name)).convert("RGB")
        data_VIS = (
                image_read_cv2(os.path.join(test_folder, "vi", img_name), mode="GRAY")[
                    np.newaxis, np.newaxis, ...
                ]
                / 255.0
            )

        data_IR, data_VIS = torch.FloatTensor(data_IR), torch.FloatTensor(data_VIS)

        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

        feature_V_B, feature_V_D, feature_V = Encoder(data_VIS)
        feature_I_B, feature_I_D, feature_I = Encoder(data_IR)
        feature_F_B = BaseFuseLayer(feature_V_B + feature_I_B)
        feature_F_D = DetailFuseLayer(feature_V_D + feature_I_D)
        data_Fuse, _ = Decoder(data_VIS, feature_F_B, feature_F_D)
        data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (
                torch.max(data_Fuse) - torch.min(data_Fuse)
            )
        fi = np.squeeze((data_Fuse * 255).cpu().numpy())
        fi = normalize_image(fi)  
        img_save(fi, img_name.split(sep=".")[0], test_out_folder)


        if np.max(fi) == 0:
            print(f"Warning: Fused image {img_name} is completely black!")
            continue
        
        data_VIS_color_np = np.array(data_VIS_color)  
        display_images(data_IR, data_VIS, fi)
        
        # fi_img_color = (fi[:, :, np.newaxis] / 255.0 * np.array(data_VIS_color)).astype(np.uint8)
        # display_fused_image(img_name, data_IR, data_VIS_color, data_VIS, fi, fi_img_color)


    eval_folder = test_out_folder
    ori_img_folder = test_folder

    metric_result = np.zeros((8))
    for img_name in os.listdir(os.path.join(ori_img_folder, "ir")):
        ir = image_read_cv2(os.path.join(ori_img_folder, "ir", img_name), "GRAY")
        vi = image_read_cv2(os.path.join(ori_img_folder, "vi", img_name), "GRAY")
        fi = image_read_cv2(
            os.path.join(eval_folder, img_name.split(".")[0] + ".png"), "GRAY"
        )
        metric_result += np.array(
            [
                Evaluator.EN(fi),
                Evaluator.SD(fi),
                Evaluator.SF(fi),
                Evaluator.MI(fi, ir, vi),
                Evaluator.SCD(fi, ir, vi),
                Evaluator.VIFF(fi, ir, vi),
                Evaluator.Qabf(fi, ir, vi),
                Evaluator.SSIM(fi, ir, vi),
            ]
        )

    metric_result /= len(os.listdir(eval_folder))
    print("\t\t EN\t\t SD\t\t SF\t\t MI\t\t SCD\t\t VIF\t\t Qabf\t\t SSIM")
    print(
        "\t\t "
        + str(np.round(metric_result[0], 4))
        + "\t\t"
        + str(np.round(metric_result[1], 4))
        + "\t\t"
        + str(np.round(metric_result[2], 4))
        + "\t\t"
        + str(np.round(metric_result[3], 4))
        + "\t\t"
        + str(np.round(metric_result[4], 4))
        + "\t\t"
        + str(np.round(metric_result[5], 4))
        + "\t\t"
        + str(np.round(metric_result[6], 4))
        + "\t\t"
        + str(np.round(metric_result[7], 4))
    )

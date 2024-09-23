import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np


# 读取中文路径名称的图片函数
def cv_imread(file_path, flags=cv2.IMREAD_COLOR):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flags)
    if cv_img is None:
        print("图像未能加载，请检查路径和文件是否存在。")
    else:
        print("图像加载成功。")
    return cv_img


# 写入中文路径名称的图片函数
def cv_imwrite(file_path, src):
    cv2.imencode(".jpg", src)[1].tofile(file_path)  # 保存图像


def draw_chinese_text(image, text, position, font_size, color):
    font = ImageFont.truetype("simhei.ttf", font_size, encoding="utf-8")
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    draw.text(position, text, font=font, fill=color)
    # return  pil_image
    return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

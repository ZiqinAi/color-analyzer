import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import json

# 加载 ImageNet 类别映射
with open("D:\桌面\智能信息网络\color-analyzer\imagenet_class_index.json", "r") as f:
    class_idx = json.load(f)

app = Flask(__name__)
CORS(app)  # 允许所有来源的请求

UPLOAD_FOLDER = "uploads" # 上传文件的目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # 创建目录
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"} # 允许的文件格式

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# 计算平均亮度的函数（使用OpenCV和PIL两种方式）
def calculate_brightness(image_tensor, normalize=False):
    # 获取图像张量的通道数，通常彩色图像有3个通道（R、G、B）
    channels = image_tensor.shape[2]
    # 初始化一个空列表，用于存储每个通道的平均亮度
    avg_brightness = []
    # 遍历每个颜色通道
    for i in range(channels):
        # 计算当前通道的平均亮度：当i为0、1、2 时，分别对应R、G、B三个通道
        channel_mean = np.mean(image_tensor[:, :, i])
        # 如果有需求，可以对亮度进行归一化处理
        if normalize:
            channel_mean = channel_mean / 255.0
        # 将当前通道的平均亮度添加到列表中
        avg_brightness.append(channel_mean)
    return avg_brightness

# 使用OpenCV加载图像并计算平均亮度
def load_image_with_cv2(image_path):
    image = cv2.imread(image_path)
    # OpenCV默认以BGR格式读取图像，所以将图像从BGR格式转换为RGB格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 将OpenCV图像转换为NumPy数组，以便后续进行数值计算和处理
    image_tensor = np.array(image_rgb)
    return image_tensor

# 使用PIL加载图像并计算平均亮度
def load_image_with_pil(image_path):
    image = Image.open(image_path)
    # 将PIL图像对象转换为NumPy数组，以便后续进行数值计算和处理
    image_tensor = np.array(image)
    return image_tensor

# 判断图像的颜色（基于平均亮度）
def classify_image_color(brightness, normalized=False):
    """ 根据 RGB 亮度值判断主导颜色 """
    # RGB亮度阈值设置
    threshold = 200 if not normalized else 0.78  # 用于判断白色
    black_threshold = threshold / 5  # 用于判断黑色

    # 通道间亮度差异的阈值
    color_diff_threshold = threshold / 4  # 当两个通道的亮度差异小于该阈值时，认为是颜色接近

    r, g, b = brightness  # 拆分 R, G, B 三个通道

    # 找到最亮的通道
    max_channel = max(brightness)
    min_channel = min(brightness)

    # 如果三通道亮度都高，判断是否接近白色
    if min_channel > threshold:
        return "白色"
    # 如果三通道亮度都低，判断是否接近黑色
    elif max_channel < black_threshold:
        return "黑色"
    
    # 如果最大通道亮度差异小于阈值，判断为多色图像
    if abs(r - g) < color_diff_threshold and abs(r - b) < color_diff_threshold and abs(g - b) < color_diff_threshold:
        return "多色图像"
    
    # 具体颜色分类逻辑
    elif r >= g and r >= b:  # 红色占主导
        return "红色" if g < threshold and b < threshold else "橙色/黄色"
    elif g >= r and g >= b:  # 绿色占主导
        return "绿色"
    elif b >= r and b >= g:  # 蓝色占主导
        return "蓝色"
    elif r >= b and g >= b:  # 黄绿色
        return "黄色/绿色"
    elif g >= r and b >= r:  # 青色
        return "青色"
    elif r >= g and b >= g:  # 品红
        return "品红"
    else:
        return "多色图像"


# 下面是我的新增功能，辅助判断：
# 使用 K-Means 聚类计算主导颜色
def get_dominant_color(image, k=3):
    """ 
    使用 K-Means 计算主导颜色:
    输入值：
        - pixels: 输入的像素点数据。
        - k: 指定聚类数（颜色数）。
        - None: 忽略预定义的标签。
        - criteria: 终止条件。
        - 10: 表示执行 10 次 K-Means 以求得最佳分类（避免局部最优）。
        - cv2.KMEANS_RANDOM_CENTERS: 初始质心随机选择。
    返回值：
        - _: K-Means 计算的压缩误差SSE。
        - labels: 每个像素点所属的聚类索引0~k-1。
        - centers: k 个聚类中心，即代表 k 种颜色。 
    """
    # 将图像展平成 (H*W, 3) 的 2D 数组，其中每一行是一个像素的颜色（R, G, B）
    pixels = image.reshape(-1, 3).astype(np.float32)

    # 终止条件：
        #   1. 迭代次数达到 10 次 (MAX_ITER=10)
        #   2. 质心变化小于 1.0 (EPS=1.0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 统计最常见的颜色
    counts = np.bincount(labels.flatten())
    dominant_color = centers[np.argmax(counts)]

    return dominant_color.astype(int)  # 返回 RGB 格式

# 使用 HSV 颜色空间分类颜色
def classify_color_hsv(image):
    """ 使用 HSV 颜色空间分类颜色 """
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # 转换为 HSV 颜色空间
    h, s, v = cv2.split(image_hsv) # 拆分 H, S, V 三个通道

    mean_hue = np.mean(h)  # 色调
    mean_saturation = np.mean(s)  # 饱和度
    mean_value = np.mean(v)  # 亮度

    # 低饱和度颜色（黑、白、灰）
    if mean_saturation < 40:
        if mean_value < 50:
            return "黑色"
        elif mean_value > 200:
            return "白色"
        else:
            return "灰色"

    # 彩色部分的分类
    if 0 <= mean_hue < 15 or 165 <= mean_hue <= 180:
        return "红色"
    elif 15 <= mean_hue < 30:
        return "橙色"
    elif 30 <= mean_hue < 45:
        return "黄色"
    elif 45 <= mean_hue < 90:
        return "绿色"
    elif 90 <= mean_hue < 150:
        return "蓝色"
    elif 150 <= mean_hue < 165:
        return "紫色"
    else:
        return "未知颜色"

# 计算颜色偏向
def get_color_bias(dominant_color):
    """ 根据 RGB 值计算颜色偏向 """
    r, g, b = dominant_color
     # 计算各通道的偏向值
    biases = {
        "红色偏向": r - (g + b) / 2,
        "绿色偏向": g - (r + b) / 2,
        "蓝色偏向": b - (r + g) / 2
    }
    
    # 判断是否为灰色（RGB 三个通道的最大最小差值小于 threshold）
    threshold = 5
    max_rgb, min_rgb = max(r, g, b), min(r, g, b)
    if max_rgb - min_rgb < threshold:
        return "无明显偏向（可能是灰色）"
    
    # 选择偏向最大的颜色
    max_bias_color, max_bias = max(biases.items(), key=lambda x: x[1])

    return max_bias_color if max_bias > 0 else "无明显偏向"

# 加载模型
def load_models():
    # 加载预训练的AlexNet和VGG16模型
    alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    
    # 设置为评估模式
    alexnet.eval()
    vgg16.eval()

    return alexnet, vgg16

# 图像分类函数
def classify_image(image_path, models, class_idx):
    transform = transforms.Compose([
        transforms.Resize(256),  # 调整图像大小
        transforms.CenterCrop(224), # 中心裁剪
        transforms.ToTensor(), # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    results = {}
    for model_name, model in models.items():
        with torch.no_grad():  # 禁用梯度计算
            output = model(image)
            _, predicted_idx = torch.max(output, 1)  # 获取预测的类别索引
            predicted_idx = predicted_idx.item()
            # 获取类别名称
            class_name = class_idx[str(predicted_idx)][1]
            results[model_name] = class_name

    return results

# 加载模型
alexnet, vgg16 = load_models()
models = {"alexnet": alexnet, "vgg16": vgg16}

@app.route("/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "未上传文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "文件格式不支持"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # 解析前端传递的参数
    normalize = request.form.get("normalize", "false") == "true"

    # 使用OpenCV和PIL两种方式加载图像并计算亮度
    image_cv2 = load_image_with_cv2(file_path)
    brightness_cv2 = calculate_brightness(image_cv2, normalize)

    image_pil = load_image_with_pil(file_path)
    brightness_pil = calculate_brightness(image_pil, normalize)

    # 根据平均亮度判断图像颜色
    color_cv2 = classify_image_color(brightness_cv2, normalize)
    color_pil = classify_image_color(brightness_pil, normalize)

    # 检查是否为多色彩图像
    multiple_colors = "是多色彩图像" if color_cv2 != color_pil else None

    # 计算主导颜色（K-Means）
    dominant_color = get_dominant_color(image_cv2)

    # 计算 HSV 颜色分类
    hsv_color = classify_color_hsv(image_cv2)

    # 计算颜色偏向
    color_bias = get_color_bias(dominant_color)

    # 加载模型并进行分类
    classification_results = classify_image(file_path, models, class_idx)
    
    # 返回结果
    return jsonify({
        "filename": file.filename, # 文件名
        "normalize": normalize, # 是否归一化
        "mean_brightness_opencv": brightness_cv2, # 平均亮度: OpenCV
        "mean_brightness_pil": brightness_pil, # 平均亮度: PIL
        "color_classification_opencv": color_cv2, # 颜色分类: OpenCV
        "color_classification_pil": color_pil, # 颜色分类: PIL   
        "multiple_colors": multiple_colors, # 是否多色图像
        "dominant_color_rgb": dominant_color.tolist(),  # 主要颜色（RGB）
        "dominant_color_hsv": hsv_color,  # 主要颜色分类（HSV）
        "color_bias": color_bias, # 颜色偏向
        "classification_results": {
        "alexnet": classification_results["alexnet"], # 分类结果: AlexNet
        "vgg16": classification_results["vgg16"] # 分类结果: VGG16
    }
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

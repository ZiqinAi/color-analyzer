import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import AlexNet_Weights, VGG16_Weights
from PIL import Image

# 预处理步骤（适用于ImageNet预训练模型）
transform = transforms.Compose([
        transforms.Resize(256),  # 调整图像大小
        transforms.CenterCrop(224), # 中心裁剪
        transforms.ToTensor(), # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
])

# 使用 OpenCV 加载图像
def load_image_opencv(image_path):
    """使用 OpenCV 加载图像并转换为 RGB"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.array(image_rgb)

# 使用 PIL 加载图像
def load_image_pil(image_path):
    """使用 PIL 加载图像"""
    image = Image.open(image_path).convert("RGB")
    return np.array(image)

# 计算平均亮度
def calculate_brightness(image, normalize=False):
    """计算平均亮度"""
    # 计算当前通道的平均亮度：当i为0、1、2 时，分别对应R、G、B三个通道
    # 计算原理：计算该通道所有像素的平均值（image（高，宽，通道i））/H*W）
    avg_brightness = [np.mean(image[:, :, i]) for i in range(3)]
    if normalize:
        avg_brightness = [b / 255.0 for b in avg_brightness]
    return avg_brightness


def classify_image_color(brightness):
    """基于亮度判断主导颜色"""
    threshold = 200
    black_threshold = threshold / 5
    color_diff_threshold = threshold / 4
    r, g, b = brightness

    if min(brightness) > threshold:
        return "白色"
    elif max(brightness) < black_threshold:
        return "黑色"
    if abs(r - g) < color_diff_threshold and abs(r - b) < color_diff_threshold:
        return "多色图像"
    if r >= g and r >= b:
        return "红色" if g < threshold and b < threshold else "橙色/黄色"
    if g >= r and g >= b:
        return "绿色"
    if b >= r and b >= g:
        return "蓝色"
    return "多色图像"


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
    # 将图像转换为二维数组，每行是一个像素的 RGB 值
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

def classify_color_hsv(image):
    """使用 HSV 颜色空间进行颜色分类"""
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mean_hue = np.mean(image_hsv[:, :, 0])
    mean_saturation = np.mean(image_hsv[:, :, 1])
    mean_value = np.mean(image_hsv[:, :, 2])

    if mean_saturation < 40:
        if mean_value < 50:
            return "黑色"
        elif mean_value > 200:
            return "白色"
        return "灰色"
    if 0 <= mean_hue < 15 or 165 <= mean_hue <= 180:
        return "红色"
    if 15 <= mean_hue < 30:
        return "橙色"
    if 30 <= mean_hue < 45:
        return "黄色"
    if 45 <= mean_hue < 90:
        return "绿色"
    if 90 <= mean_hue < 150:
        return "蓝色"
    if 150 <= mean_hue < 165:
        return "紫色"
    return "未知颜色"


# 计算颜色偏向
# 根据上面的RGB结果，计算颜色偏向
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


def classify_image(model, image_path):
    """使用预训练模型进行图像分类"""
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    # 加载 ImageNet 标签
    labels_path = os.path.join(os.path.dirname(__file__), "imagenet_classes.txt")
    if not os.path.exists(labels_path):
        import urllib.request
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        urllib.request.urlretrieve(url, labels_path)

    with open(labels_path) as f:
        labels = [line.strip() for line in f.readlines()]

    return labels[predicted.item()]


def main():
    """主函数"""

    # 让用户输入图片路径
    input_paths = input("请输入图像文件路径（多个路径用逗号分隔）：").strip()
    image_paths = [path.strip() for path in input_paths.split(",") if os.path.exists(path)]

    if not image_paths:
        print("未找到有效的图像文件，请检查输入路径！")
        return

    # 让用户选择是否归一化
    normalize_input = input("是否归一化亮度？(yes/no): ").strip().lower()
    normalize = normalize_input in ["yes", "y"]

    # 选择模型
    model_name = input("请选择模型（alexnet/vgg16）：").strip().lower()
    if model_name == "alexnet":
        model = models.alexnet(weights=AlexNet_Weights.DEFAULT)  
    elif model_name == "vgg16":
        model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    else:
        print("不支持的模型！默认使用 AlexNet")
        model = models.alexnet(pretrained=True)

    for image_path in image_paths:
        print("\n===== 计算结果 =====")
        print(f"图像文件: {image_path}")

       # 使用 OpenCV 加载图像
        image_opencv = load_image_opencv(image_path)  # RGB 格式

        # 使用 PIL 加载图像
        image_pil = load_image_pil(image_path) # RGB 格式

        # 计算亮度
        brightness_cv = calculate_brightness(image_opencv, normalize)
        brightness_pil = calculate_brightness(image_pil, normalize)

        print(f"是否归一化: {'是' if normalize else '否'}")
        print(f"OpenCV 计算的平均亮度: {brightness_cv}")
        print(f"PIL 计算的平均亮度: {brightness_pil}")
        
        # 将 OpenCV 加载的图像转换为张量并输出
        transform = transforms.ToTensor()
        image_opencv_tensor = transform(Image.fromarray(image_opencv)) # OpenCV -> PIL -> Tensor
        print(f"转换后的图像张量 (OpenCV):\n {image_opencv_tensor}")

        # 将 PIL 加载的图像转换为张量并输出
        image_pil_tensor = transform(Image.fromarray(image_pil))
        print(f"转换后的图像张量 (PIL):\n {image_pil_tensor}")

        # 颜色判断
        color_cv = classify_image_color(brightness_cv)
        color_pil = classify_image_color(brightness_pil)

        print(f"主导颜色 (OpenCV): {color_cv}")
        print(f"主导颜色 (PIL): {color_pil}")
        print(f"是否多色图像: {'是' if color_cv == '多色图像' else '否'}")

        # HSV 颜色分类
        hsv_color = classify_color_hsv(image_opencv)
        print(f"HSV 颜色分类: {hsv_color}")

        # 计算主导颜色
        dominant_color = get_dominant_color(image_opencv)
        print(f"主导颜色 (RGB): {dominant_color}")

        # 计算颜色偏向
        color_bias = get_color_bias(dominant_color)
        print(f"颜色偏向度: {color_bias}")

        # 进行图像分类
        predicted_label = classify_image(model, image_path)
        print(f"识别的物体分类: {predicted_label}")


if __name__ == "__main__":
    main()

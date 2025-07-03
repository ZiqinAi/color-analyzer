import React, { useState } from "react";
import axios from "axios";
import { Bar, Line } from "react-chartjs-2"; // 导入Bar组件，绘制条形图；导入Line组件，绘制折线图
import "chart.js/auto";

function App() {
    // 使用 useState 管理图片、预览、分析结果和图表数据的状态
    const [image, setImage] = useState(null); // 图片
    const [preview, setPreview] = useState(null); // 预览
    const [result, setResult] = useState(null); // 分析结果
    const [multipleColors, setMultipleColors] = useState(null); // 多色分析结果
    const [meanBrightnessPIL, setMeanBrightnessPIL] = useState(null); // PIL的平均亮度值
    const [meanBrightnessOpenCV, setMeanBrightnessOpenCV] = useState(null); // OpenCV的平均亮度值
    const [chartData, setChartData] = useState(null); // 图表数据
    const [brightnessData, setBrightnessData] = useState(null); // 亮度数据
    const [normalize, setNormalize] = useState(false);  // 默认不标准化
    const [dominantColor, setDominantColor] = useState(null);  // 主导颜色
    const [hsvColor, setHsvColor] = useState(null);  // HSV分类
    const [colorBias, setColorBias] = useState(null);  // 颜色偏向度
    const [showDominantColor, setShowDominantColor] = useState(false); // 控制显示主导颜色
    const [showHsvColor, setShowHsvColor] = useState(false); // 控制显示HSV分类
    const [showColorBias, setShowColorBias] = useState(false); // 控制显示颜色偏向度
    const [classifyImage, setClassifyImage] = useState(false); // 默认不启用图像分类
    const [classificationResult, setClassificationResult] = useState(null); // 图像分类结果

    const handleImageChange = (event) => {
        // 获取用户选择的文件
        const file = event.target.files[0];
        // 如果用户选择了文件
        if (file) {
            // 更新图片状态
            setImage(file);
            // 创建文件的 URL 并更新预览状态
            setPreview(URL.createObjectURL(file));
        }
    };

    const handleUpload = async () => {
        if (!image) {
            alert("请先选择一张图片！");
            return;
        }

        // 创建一个 FormData 对象来上传文件
        const formData = new FormData();
        formData.append("file", image);// 将图片添加到 FormData 中
        formData.append("normalize", normalize ? "true" : "false");  // 传递normalize标志
        formData.append("classify", classifyImage ? "true" : "false");  // 传递图像分类开关

        try {
            // 发送 POST 请求到服务器
            const response = await axios.post("http://127.0.0.1:5000/upload", formData);
            // 从响应中获取数据
            const data = response.data;

             // 设置分析结果
             setResult(
                <div style={{ textAlign: "center", marginTop: "10px" }}>
                    <div style={{ backgroundColor: "white", padding: "10px 20px", borderRadius: "5px", marginBottom: "10px", fontSize: "18px", fontWeight: "bold", color: "#333", boxShadow: "0 2px 4px rgba(0, 0, 0, 0.1)" }}>
                        颜色分类 (PIL): {data.color_classification_pil}
                    </div>
                    <div style={{ backgroundColor: "white", padding: "10px 20px", borderRadius: "5px", fontSize: "18px", fontWeight: "bold", color: "#333", boxShadow: "0 2px 4px rgba(0, 0, 0, 0.1)" }}>
                        颜色分类 (OpenCV): {data.color_classification_opencv}
                    </div>
                </div>
            );

            setMultipleColors(data.multiple_colors);

            // 设置PIL和OpenCV的平均亮度值
            setMeanBrightnessPIL(data.mean_brightness_pil);
            setMeanBrightnessOpenCV(data.mean_brightness_opencv);

            // 设置图表数据
            setChartData({
                labels: ["Red", "Green", "Blue"],
                datasets: [
                    {
                        label: "平均亮度(PIL)",
                        data: data.mean_brightness_pil,
                        backgroundColor: ["red", "green", "blue"],
                    },
                    {
                        label: "平均亮度(OpenCV)",
                        data: data.mean_brightness_opencv,
                        backgroundColor: ["red", "green", "blue"],
                    },
                ],
            });
            // 设置主导颜色（RGB）
            setDominantColor(data.dominant_color_rgb);

            // 设置HSV颜色分类
            setHsvColor(data.dominant_color_hsv);

            // 设置颜色偏向度
            setColorBias(data.color_bias);
            
            // 设置分类结果
            if (data.classification_results) {
                setClassificationResult(data.classification_results);
            }

            // 设置亮度图数据
            setBrightnessData({
                labels: ["Red", "Green", "Blue"],
                datasets: [
                    {
                        label: "OpenCV 计算的亮度",
                        data: data.mean_brightness_opencv,
                        backgroundColor: "rgba(255, 99, 132, 0.5)",
                        borderColor: "rgba(255, 99, 132, 1)",
                        borderWidth: 1,
                    },
                    {
                        label: "PIL 计算的亮度",
                        data: data.mean_brightness_pil,
                        backgroundColor: "rgba(54, 162, 235, 0.5)",
                        borderColor: "rgba(54, 162, 235, 1)",
                        borderWidth: 1,
                    },
                ],
            });

        } catch (error) {
            console.error("上传失败", error.response ? error.response.data : error);
            setResult("上传失败，请检查服务器是否运行！");
        }
    };

    // 切换normalize状态
    const toggleNormalize = () => {
        setNormalize(!normalize);
    };

    return (
        <div style={{ textAlign: "center", padding: "20px", backgroundColor: "#f0f8ff", borderRadius: "10px", boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)" }}>
        <h2 style={{ color: "#333", marginBottom: "20px" }}>颜色亮度分析</h2>
    
        {/* 上传图片 */}
        <input type="file" accept="image/*" onChange={handleImageChange} style={{ marginBottom: "10px" }} />
        <br />
        {preview && (
        <img
            src={preview}
            alt="预览"
            style={{
                maxWidth: "300px",
                marginTop: "10px",
                borderRadius: "5px",
                boxShadow: "0 2px 4px rgba(0, 0, 0, 0.1)",
                }}
            />
        )}
        <br />

        <div style={{ display: "flex", justifyContent: "center", gap: "10px", marginBottom: "20px" }}>
            <button onClick={handleUpload} style={{ padding: "10px 20px", cursor: "pointer", backgroundColor: "#4CAF50", color: "white", border: "none", borderRadius: "5px" }}>
                分析图片
            </button>
            <button onClick={toggleNormalize} style={{ padding: "10px 20px", cursor: "pointer", backgroundColor: "#4CAF50", color: "white", border: "none", borderRadius: "5px" }}>
                {normalize ? "取消标准化" : "使用标准化"}
            </button>
        </div>

        {/* 控制主导颜色、HSV分类、主导颜色和图像分类显示的按钮 */}
        <div style={{ display: "flex", justifyContent: "center", gap: "20px", marginBottom: "20px" }}>
            <label>
                <input
                    type="checkbox"
                    checked={showDominantColor}
                    onChange={() => setShowDominantColor(!showDominantColor)}
                    style={{ marginRight: "10px" }}
                />
                显示主导颜色 (RGB)
            </label>
            <label>
                <input
                    type="checkbox"
                    checked={showHsvColor}
                    onChange={() => setShowHsvColor(!showHsvColor)}
                    style={{ marginRight: "10px" }}
                />
                显示HSV颜色分类
            </label>
            <label>
                <input
                    type="checkbox"
                    checked={showColorBias}
                    onChange={() => setShowColorBias(!showColorBias)}
                    style={{ marginRight: "10px" }}
                />
                显示颜色偏向度
            </label>
            <label>
                <input
                    type="checkbox"
                    checked={classifyImage}
                    onChange={() => setClassifyImage(!classifyImage)}
                    style={{ marginRight: "10px" }}
                />
                启用图像分类(AlexNet & VGG16)
            </label>
        </div>

        {/* 平均亮度 */}
        <h3>{result}</h3>
        
        {multipleColors && <p><strong>{multipleColors}</strong></p>}

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px", marginBottom: "20px" }}>
                {meanBrightnessPIL && (
                    <div style={boxStyle}>
                        <h4>PIL计算的平均亮度：</h4>
                        <p>红色: {meanBrightnessPIL[0].toFixed(3)}</p>
                        <p>绿色: {meanBrightnessPIL[1].toFixed(3)}</p>
                        <p>蓝色: {meanBrightnessPIL[2].toFixed(3)}</p>
                    </div>
                )}
                {meanBrightnessOpenCV && (
                    <div style={boxStyle}>
                        <h4>OpenCV计算的平均亮度：</h4>
                        <p>红色: {meanBrightnessOpenCV[0].toFixed(3)}</p>
                        <p>绿色: {meanBrightnessOpenCV[1].toFixed(3)}</p>
                        <p>蓝色: {meanBrightnessOpenCV[2].toFixed(3)}</p>
                    </div>
                )}
        </div>

        {showDominantColor && dominantColor && (
            <div style={{ ...boxStyle, marginBottom: "20px", maxWidth: "300px" }}>
                <h4>主导颜色 (RGB)</h4>
                <div style={{ display: "flex", justifyContent: "center", gap: "10px", alignItems: "center" }}>
                    <div style={{ width: "30px", height: "30px", backgroundColor: `rgb(${dominantColor[0]}, ${dominantColor[1]}, ${dominantColor[2]})` }}></div>
                    <p>红色: {dominantColor[0]}</p>
                    <p>绿色: {dominantColor[1]}</p>
                    <p>蓝色: {dominantColor[2]}</p>
                </div>
            </div>
        )}

        {showHsvColor && hsvColor && (
            <div style={{ ...boxStyle, marginBottom: "20px", maxWidth: "300px" }}>
                <h4>HSV 颜色分类</h4>
                <div style={{ display: "flex", justifyContent: "center", gap: "10px", alignItems: "center" }}>
                    <p>{hsvColor}</p>
                </div>
            </div>
        )}

        {showColorBias && colorBias && (
            <div style={{ ...boxStyle, marginBottom: "20px", maxWidth: "300px" }}>
                <h4>颜色偏向度</h4>
                <div style={{ textAlign: "center" }}>
                    <p>{colorBias}</p>
                </div>
            </div>
        )}
                    
        {classifyImage && classificationResult && (
            <div style={{ ...boxStyle, marginBottom: "20px", maxWidth: "300px" }}>
                <h4>图像分类结果：</h4>
                <p>AlexNet 分类结果: {classificationResult.alexnet}</p>
                <p>VGG16 分类结果: {classificationResult.vgg16}</p>
            </div>
        )}

         {/* 亮度图 */}
         {brightnessData && (
                <div style={{ maxWidth: "800px", margin: "20px auto" }}>
                    <h3>亮度图</h3>
                    <Line data={brightnessData} />
                </div>
        )}
        
        {/* 绘制图表 */}
            {chartData && <Bar data={chartData} />}
        </div>
    );
}

const boxStyle = {
    border: "2px solid #4CAF50",
    borderRadius: "10px",
    padding: "10px",
    boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
    backgroundColor: "#f9f9f9",
    textAlign: "center",
    width: "500px", 
    margin: "0 auto",   // 水平居中
};

export default App;

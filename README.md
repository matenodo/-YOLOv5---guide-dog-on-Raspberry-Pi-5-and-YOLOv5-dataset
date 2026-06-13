# 基于 YOLOv5s 的智能导盲系统（树莓派5B 部署版）

本项目实现了一个运行在树莓派5B上的实时目标检测与语音导盲系统。系统采用轻量化YOLOv5s模型，通过CSI摄像头采集环境图像，识别行人、车辆、坑洞、红绿灯等11类障碍物，并根据目标方位和优先级进行语音提示，为视障人士提供辅助出行支持。

## 主要特性

- **实时检测**：经过ONNX格式优化，在416×416输入尺寸下可达6~8.6 FPS，端到端延迟约309ms。
- **语义理解**：可区分坑洞、障碍物、路障、行人、车辆、斑马线、树木、杆子、台阶、红绿灯等11类目标。
- **优先级播报**：支持高危目标（坑洞等）始终播报，中低危目标根据面积大小和密度触发。
- **方位提示**：判断目标位于左前方、前方或右前方，配合预录OGG语音文件播报。

## 系统硬件要求

| 部件 | 推荐型号 | 备注 |
|------|----------|------|
| 主控 | 树莓派5B（8GB 内存） | 性能优于4B，推荐使用 |
| 摄像头 | Raspberry Pi Camera Module 1 (CSI接口) | 或兼容的CSI摄像头 |
| 扬声器 | USB声卡 + 小音箱 或 3.5mm音频输出 | 建议使用USB声卡降低CPU负载 |
| 电源 | 5V/5A USB-C 电源 (如官方27W电源) | 确保稳定供电 |
| 存储 | microSD卡（≥32GB，建议A2等级） | 用于系统与数据 |

## 软件环境

- 操作系统：Raspberry Pi OS (64位，基于Debian Bookworm)
- Python 3.9+
- 主要依赖库：`onnxruntime`、`opencv-python`、`pygame`、`picamera2`、`numpy`

## 数据集

本系统使用的数据集为自行采集并标注的导盲场景数据集，共包含11个类别：
`坑洞`、`障碍物`、`路障`、`行人`、`车辆`、`斑马线`、`树木`、`杆子`、`台阶`、`红灯`、`绿灯`。

**数据集下载**（约930MB）：
[点击此处下载数据集] （后期附）

下载后请将数据集解压到项目根目录下的 `dataset/` 文件夹（若仅需模型权重，可忽略数据集）。

**演示视频**：
http://xhslink.com/o/8vYJLD1PA1Z
## 树莓派部署指南

### 1. 烧录系统并初始设置

- 使用 Raspberry Pi Imager 烧录 **Raspberry Pi OS (64-bit)** 到microSD卡。
- 启用SSH和VNC（可选），设置用户名密码。
- 连接网络，执行系统更新：
  ```bash
  sudo apt update && sudo apt upgrade -y
  
 # 安装系统工具
 ```bash
sudo apt install python3-pip python3-venv libopenblas-dev -y

# 创建虚拟环境（推荐）
cd ~
python3 -m venv yolov5_env
source yolov5_env/bin/activate

# 安装核心库
pip install numpy opencv-python pygame picamera2

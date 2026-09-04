# 测谎图形数据合成与 YOLO 训练

以下命令均在项目根目录执行。

## 1. 安装依赖

```bash
pip3 install -r requirements.txt
```

## 2. 合成训练数据

使用默认配置生成数据：

```bash
python3 ml/build_lie_dataset.py
```

默认配置：

- 25 种图案，统一标注为 `lie_shape`
- 2000 张训练图，每种图案 80 张
- 500 张验证图，每种图案 20 张
- 图像尺寸为 960×640
- 数据输出到 `ml/lie_dataset/`
- YOLO 数据配置写入 `ml/data_lie.yaml`

完整参数命令：

```bash
python3 ml/build_lie_dataset.py \
  --train 2000 \
  --val 500 \
  --background-samples 48 \
  --width 960 \
  --height 640 \
  --seed 20260831
```

参数说明：

- `--train`：训练集图片数量。
- `--val`：验证集图片数量。
- `--background-samples`：每段录屏用于恢复背景的抽样帧数量。
- `--width`、`--height`：生成图片尺寸。
- `--seed`：随机种子；相同参数和种子可以复现数据。

> 注意：每次运行都会删除并重新生成整个 `ml/lie_dataset/` 目录。

生成后可查看：

- `ml/lie_dataset/preview.jpg`：25 种图案及标注框预览。
- `ml/lie_dataset/summary.json`：数据量和图案类型统计。
- `ml/lie_dataset/manifest.json`：每张图片的来源与图案元数据。
- `ml/lie_dataset/assets/`：背景和轮廓资产。

## 3. 训练 YOLO

```bash
python3 ml/train.py \
  --data data_lie.yaml \
  --name lie_shape \
  --out models/lie_shape_yolo.pt

python ml/train.py --data data_lie.yaml --name lie_shape --out models/lie_shape_yolo.pt --epochs 100
```

训练脚本默认使用：

- 基础模型：`yolo11n.pt`
- 训练轮数：100
- 输入尺寸：640
- 批次大小：16
- 设备：自动选择 CUDA 或 CPU

自定义训练参数示例：

```bash
python3 ml/train.py \
  --data data_lie.yaml \
  --model yolo11n.pt \
  --epochs 200 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --name lie_shape \
  --out models/lie_shape_yolo.pt
```

如果没有 NVIDIA GPU，可以指定 CPU：

```bash
python3 ml/train.py \
  --data data_lie.yaml \
  --device cpu \
  --name lie_shape_cpu \
  --out models/lie_shape_yolo.pt
```

训练产物：

- 训练记录：`ml/runs/lie_shape/`
- 最佳权重：`models/lie_shape_yolo.pt`

## 4. 一次执行数据合成和训练

```bash
python3 ml/build_lie_dataset.py && \
python3 ml/train.py \
  --data data_lie.yaml \
  --name lie_shape \
  --out models/lie_shape_yolo.pt
```


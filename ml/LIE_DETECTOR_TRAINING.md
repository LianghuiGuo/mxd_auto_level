# 测谎图形数据合成与 YOLO 训练

推荐使用 V2 数据集。旧的 `build_lie_dataset.py` 保留用于基线对照；V2
使用折射式弱边界、短序列、局内固定尺寸、遮挡、负样本、压缩退化和
鼠标移除痕迹，并按录屏来源隔离训练集与验证集。

以下命令均在项目根目录执行。

## 1. 安装依赖

```bash
pip3 install -r requirements.txt
```

## 2. 合成 V2 训练数据（推荐）

```bash
python3 ml/build_lie_dataset_v2.py \
  --train 8000 \
  --val 1000 \
  --width 700 \
  --height 464 \
  --clip-length 4 \
  --recorded-ratio 0.68 \
  --negative-ratio 0.16 \
  --cursor-probability 0.50 \
  --background-samples 72 \
  --seed 20260904
```

输出：

- `ml/lie_dataset_v2/images/{train,val}`：YOLO 图片。
- `ml/lie_dataset_v2/labels/{train,val}`：单类 YOLO 标注。
- `ml/lie_dataset_v2/manifest.json`：`clip_id`、`track_id`、角度及目标身份。
- `ml/lie_dataset_v2/summary.json`：数据分布和 train/val 背景来源。
- `ml/lie_dataset_v2/preview.jpg`：包含目标框和普通候选框的预览。
- `ml/data_lie_v2.yaml`：训练配置。

生成器只读取仓库根目录下原有的四段素材，不读取 `ml/videos/`，因此后者
可继续作为真实测试集。默认将 `测谎视频4.mp4` 的背景只用于验证；可通过
`--validation-source` 更换，但不要让同一录屏背景同时进入 train 和 val。

如果已经在 CVAT/Label Studio 中人工复核了真实帧，可按下面的目录组织：

```text
ml/lie_real_reviewed/
  images/train/  labels/train/
  images/val/    labels/val/
```

然后追加到 V2 数据集中：

```bash
python3 ml/build_lie_dataset_v2.py \
  --reviewed-real ml/lie_real_reviewed
```

这里只接受标准的单类 YOLO 标签并校验坐标范围。该入口是显式可选的，避免
把 `ml/videos/` 中的最终测试视频意外混入训练集。建议真实复核帧最终占训练
数据的约 20%–25%，且真实验证帧必须按整段录屏与训练集隔离。

### 2.1 半自动生成真实预标注（推荐配合 `--reviewed-real`）

> **划分原则（重要）**：只能**按整段视频（整局）划分** train/val，绝不能在
> 单段视频内部按时间前 80% 训练 / 后 20% 验证。同一局测谎里相邻帧共享同一
> 背景、同一组图案、同一几何、同一运动轨迹，时间切分会把近乎相同的帧泄漏到
> 两个集合，验证分数虚高、测不出泛化。

当前训练集只使用 `ml/videos/` 下的 **11 段 `测谎录屏*.mp4`**（已移除旧的
`测谎视频*`）。各段面板 ROI、active 时间窗口、train/val 归属全部由
`ml/lie_videos_config.json` 描述；ROI 由 `ml/detect_lie_panels.py` **根据测谎仪
对话框边框自动定位**（底部提示栏 + 顶部标题栏夹出中间图案区）。默认整段划分：
9 段 train / 2 段 val（`测谎录屏4` + `测谎录屏11`）。生成后请对照
`preview_panels.jpg` / `preview_panels_config.jpg` 与
`ml/lie_real_dataset/preview.jpg` 人工复核。

```bash
# 1) 自动检测录屏系列面板 ROI（默认只处理文件名含「录屏」的视频）
python3 ml/detect_lie_panels.py
#    -> 更新 ml/lie_videos_config.json + ml/preview_panels.jpg
#    已有手填的「测谎视频*」条目会保留；录屏条目会刷新。

# 只按现有配置画红框复核（不重新检测）
python3 ml/detect_lie_panels.py --from-config
#    -> ml/preview_panels_config.jpg

# 2) 生成预标注（纯背景差分，最稳）
python3 ml/build_lie_real_dataset.py

# 叠加现有 YOLO 作为辅助候选（低置信，仅补漏，不作硬门控）
python3 ml/build_lie_real_dataset.py --model models/lie_shape_yolo.pt

# 覆盖配置里的 val 划分（可重复）
python3 ml/build_lie_real_dataset.py --val-video 测谎录屏4.mp4 --val-video 测谎视频3.mp4
```

> 面板是岩石纹理、与游戏场景颜色接近时，旧的「浅色画布」检测会失败；当前实现
> 改为锚定对话框上下两根浅色实心栏。若某段 active 窗口偏短或 ROI 略含标题栏，
> 直接改 `lie_videos_config.json` 对应字段即可。配置里 `active_end` 若超过面板
> 实际存在时间，运动阶段末尾会抽到普通游戏画面——复核时删帧或调小 `active_end`。

流程：分阶段抽帧（高亮 2 FPS、淡出 9 FPS、运动 4 FPS）→ 去绿色鼠标 → 时间
中值背景差分候选（YOLO 可选辅助）→ 固定为开局白色图案的标准尺寸并按可见
区域裁剪 → 相邻帧轨迹连续性补齐漏检、删除瞬时误检。

输出到 `ml/lie_real_dataset/`：

- `images/{train,val}`、`labels/{train,val}`：与 `--reviewed-real` 完全兼容。
- `metadata/{train,val}_tracks.json`：轨迹与 `is_target`（复核时填），供后续
  评测 tracker / DeepSORT 使用，**不参与 YOLO 训练**。
- `cvat_tasks.json`：导入 CVAT / Label Studio 的任务清单。
- `preview.jpg`、`manifest.json`、`summary.json`。

> 预标注目前质量很差（木纹背景差分常出近全图框），**不要当真值用**。启动阶段
> 请走下面的本地手动画框工具，先标约 80–150 张干净框，**直接用手动集训练**
> （见 §A.6 / §3.1）；模型可用后再回头做预标 + 审校。绿色圆圈是玩家鼠标，
> 不能作为真值。

### 2.2 人工标注（启动数据优先用本地画框工具）

绿色圆圈是玩家鼠标，**任何阶段都不标**。无图案的帧标成**空帧**（负样本），不要
硬画框。`is_target` 在 YOLO 检测阶段不需要管。

#### 方案 A：本地画框页（推荐启动用，零依赖）

预标注不可靠时，从空白画框比审校坏框更快。仅需 Python 标准库 + 浏览器，
单类 `lie_shape`，写出标准 YOLO txt，并用 `.meta.json` 区分「已标 / 空帧 / 未标」。

相关文件：

| 路径 | 作用 |
|------|------|
| `ml/prepare_lie_manual_subset.py` | 从 `lie_real_dataset` 抽样启动帧（不拷贝预标） |
| `ml/lie_annotator/server.py` | 本地 HTTP 标注服务 |
| `ml/lie_annotator/static/index.html` | 画布标注页 |
| `ml/lie_manual_dataset/` | 标注工作区（图片 + 你写出的标签） |

##### A.1 前置条件

先有已裁切的面板帧（`build_lie_real_dataset.py` 产出即可；**不依赖其预标质量**）：

```bash
# 若尚无 ml/lie_real_dataset/images/{train,val}
python3 ml/build_lie_real_dataset.py
```

确认 `ml/lie_videos_config.json` 里 `val` 是整视频划分（当前为录屏 4、11），
抽样脚本会按该配置把帧分到 train / val，避免时间泄漏。

##### A.2 抽样启动集

```bash
python3 ml/prepare_lie_manual_subset.py --per-video 10 --reset
```

常用参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--per-video N` | `10` | 每个录屏抽 N 张（建议 8–15） |
| `--reset` | off | 清空已有 `lie_manual_dataset` 再拷贝（**会删掉已标标签**） |
| `--seed` | `20260906` | 抽样随机种子，便于复现 |
| `--source` | `ml/lie_real_dataset` | 源图目录（需含 `images/{train,val}`） |
| `--output` | `ml/lie_manual_dataset` | 标注工作区 |
| `--config` | `ml/lie_videos_config.json` | 读 `recordings` / `val` 划分 |

行为要点：

- **不拷贝** `lie_real_dataset` 里的机器预标；标签目录从空白开始。
- 按视频 stem 匹配文件名（长名优先，避免「录屏1」误吃「录屏10」）。
- 写出 `classes.txt`（`lie_shape`）和 `subset_summary.json`（每视频抽了多少）。

续标时**不要**加 `--reset`，直接开服务即可；只有想换一批帧才 `--reset`。

##### A.3 启动标注页

```bash
python3 ml/lie_annotator/server.py
# 浏览器打开 http://127.0.0.1:8765
```

服务参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--images` | `ml/lie_manual_dataset/images` | 图片根目录 |
| `--labels` | `ml/lie_manual_dataset/labels` | 标签根目录 |
| `--host` | `127.0.0.1` | 仅本机访问 |
| `--port` | `8765` | 端口 |
| `--load-existing` | off | 即使没有 `.meta.json` 也加载已有 YOLO txt（默认关闭，避免读入坏预标） |

示例：改端口，或标注别的目录：

```bash
python3 ml/lie_annotator/server.py --port 8766
python3 ml/lie_annotator/server.py \
  --images ml/lie_manual_dataset/images \
  --labels ml/lie_manual_dataset/labels
```

##### A.4 页面操作与快捷键

左侧：帧列表 + 进度（已标 / 空帧 / 未标 / 框数）。中间：画布。右侧：当前帧的框列表。

| 操作 | 方式 |
|------|------|
| 画框 | 空白处拖拽；松开后出现框（最小约 4×4 px） |
| 选中 | 点击框，或点右侧列表 |
| 移动 | 选中后拖拽框内部 |
| 缩放 | 选中后拖角点 / 边中点 |
| 删框 | 选中后点「删选中框」，或 `Del` / `Backspace` |
| 保存（有框 → done） | 「保存」或 `S`；无框时会提示改用空帧 |
| 空帧（负样本） | 「标为空帧」或 `E` |
| 取消标注 | 「取消标注」或 `U`（删除该帧标签与 meta） |
| 上一张 / 下一张 | 按钮，或 `A` / `←`、`D` / `→` |

未保存切帧会弹出确认。左侧徽章：数字 = 已标框数；「空」= 负样本；「—」= 未标。

##### A.4.1 标注约定

任务是检测面板里每一个测谎图案实例（单类 `lie_shape`），不是标「谁是真目标」。
按**当前画面里该图案的可见外接矩形**紧贴去画。

- **白色高亮也要框。** 开局全白只是高亮态，仍是 `lie_shape`。高亮 / 淡出 /
  运动阶段都标；绿色鼠标光标任何阶段都不标。
- **框不要强行一样大。** 同局里图案往往差不多大，但不同视频、分辨率、淡出、
  遮挡、贴边都会改变可见区域。框跟着可见外形走，不要为了「看起来一样大」
  故意拉大或缩小。同一张图里各实例也可以大小不同。
- **部分融合时优先标两个框。** 还能分清两个实体（两个中心 / 轮廓还在）→
  两个框，允许大幅重叠。已经糊成一团、肉眼分不出两个实例 → 一个框包住整团。
  拿不准时优先标两个。
- **贴边只标图内可见部分。** 图案超出画面时，框贴齐图像边界，不要脑补画到
  画外。超出多少，框就裁到边缘为止。
- **无图案或整帧无效 → 空帧。** 面板上暂无图案、或已切回普通游戏画面，标
  **空帧**，不要硬画。

简记：紧贴可见外形、一实例一框、越界只标可见、白态也标。

建议规模：每视频约 8–15 张，合计约 **80–150** 张即可启动微调。

##### A.5 磁盘上的标注结果

```text
ml/lie_manual_dataset/
  classes.txt
  subset_summary.json
  images/train/*.jpg
  images/val/*.jpg
  labels/train/<stem>.txt          # YOLO：每行  class cx cy w h（归一化）
  labels/train/<stem>.meta.json    # {"status":"done"|"empty"|"unset", "note":""}
  labels/val/...
```

状态含义：

| status | 含义 | 文件 |
|--------|------|------|
| `done` | 已画框 | `.txt` 有框 + `.meta.json` |
| `empty` | 明确负样本 | 空 `.txt` + `.meta.json` |
| `unset` | 未处理 / 已取消 | 无 meta（取消时会删掉对应 txt/meta） |

训练时只需要 YOLO 的 `images/` + `labels/*.txt`；`.meta.json` 仅供标注页进度，
不影响 Ultralytics。空 `.txt` 表示该图无目标（负样本），应保留。

##### A.6 标完后训练（优先直接用手动集）

配置在 `ml/data_lie_manual.yaml`，指向 `ml/lie_manual_dataset/`。图少，epoch
不必拉到 80；val 掉下去就停。

```bash
python3 ml/train.py \
  --lie-detector \
  --data data_lie_manual.yaml \
  --model yolo11n.pt \
  --epochs 40 \
  --batch 8 \
  --name lie_shape_manual \
  --out models/lie_shape_yolo_manual.pt
```

合成数据是没真实框时的替代，不是这批标注的前置。若再混 V2，真实帧要占到能被
梯度看见的比例（大约 20%+），不要 72 张对 8000 张合成：

```bash
python3 ml/build_lie_dataset_v2.py --reviewed-real ml/lie_manual_dataset
python3 ml/train.py \
  --lie-detector \
  --data data_lie_v2.yaml \
  --model yolo11n.pt \
  --epochs 80 \
  --batch 16 \
  --name lie_shape_v2 \
  --out models/lie_shape_yolo_v2.pt
```

也可日后把同一套标注页指到更大图集（改 `--images` / `--labels`），继续增量标注。

#### 方案 B：Label Studio（预标可用后再审校）

1. 安装并开启本地文件服务（Label Studio 自带 converter）：

   ```bash
   pip install label-studio
   export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
   export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$(pwd)/ml/lie_real_dataset
   label-studio start   # 浏览器打开 http://localhost:8080
   ```

2. 把 YOLO 预标签转成 Label Studio 任务（train / val 各一次）：

   ```bash
   cd ml/lie_real_dataset
   label-studio-converter import yolo -i . -o ls_train.json \
     --image-root-url "/data/local-files/?d=images/train"
   label-studio-converter import yolo -i . -o ls_val.json \
     --image-root-url "/data/local-files/?d=images/val"
   ```

   > converter 需要一个 `classes.txt`（内容一行 `lie_shape`）。`build_lie_real_dataset.py`
   > 已自动生成该文件；若缺失可手动创建。

3. 新建 Project，模板选 **Object Detection with Bounding Boxes**，标签只填
   `lie_shape`；在 Settings → Cloud Storage → Add Source Storage → **Local files**，
   Absolute local path 填 `.../ml/lie_real_dataset/images/train`，Sync；再到
   Data Manager → Import 传 `ls_train.json`，即可看到带预标框的图片开始审校。

4. 复核完 Export → 选 **YOLO**，把导出的 `labels/` 覆盖回
   `ml/lie_real_dataset/labels/train`（val 同理）。

#### 方案 C：CVAT（大批量、快捷键顺手时用，需 Docker）

1. 部署：`git clone https://github.com/cvat-ai/cvat && cd cvat && docker compose up -d`，
   浏览器打开 `http://localhost:8080`。
2. 建 Task，上传 `images/train`；Upload annotations 选 **YOLO 1.1** 格式喂
   `labels/train`，即可在预标框上审校。
3. 复核完 Export annotations 选 **YOLO 1.1**，把 `labels/` 覆盖回对应目录。

若用 B/C 审校预标，每张图只做：删误框、补漏检、修贴边、弃无效帧。

#### 标注后

优先用 `lie_manual_dataset` 直接训练（§A.6 / §3.1）。合成 V2 只在需要补未见几何
时再混；混的时候真实帧占比要够（约 20%+）。之后：微调 → 再预标剩余帧 → 只查
低置信与相邻帧数量突变（主动学习）。

快速冒烟测试：

```bash
python3 ml/build_lie_dataset_v2.py \
  --train 24 --val 12 \
  --background-samples 8 \
  --output ml/lie_dataset_v2_smoke
```

## 3. 训练

### 3.1 手动标注集（当前推荐）

已有 `lie_manual_dataset` 时，直接训，不必先生成合成数据。命令见 **§A.6**。

```bash
python ml/train.py --lie-detector --data data_lie_manual.yaml --model yolo11n.pt --epochs 40 --batch 8 --name lie_shape_manual --out models/lie_shape_yolo_manual.pt
```

### 3.2 合成 V2 模型

```bash
python3 ml/train.py \
  --lie-detector \
  --data data_lie_v2.yaml \
  --model yolo11n.pt \
  --epochs 80 \
  --batch 16 \
  --name lie_shape_v2 \
  --out models/lie_shape_yolo_v2.pt
```

`--lie-detector` 默认使用 `imgsz=768`，关闭 Mosaic/MixUp/Copy-Paste，并将
HSV、缩放和平移增强收窄到与游戏画面相符的范围。需要覆盖默认分辨率时可
显式传入 `--imgsz`。

## 4. V1 基线数据（仅用于对照）

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

## 5. 训练 V1 基线模型

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

## 6. 一次执行 V1 数据合成和训练

```bash
python3 ml/build_lie_dataset.py && \
python3 ml/train.py \
  --data data_lie.yaml \
  --name lie_shape \
  --out models/lie_shape_yolo.pt
```

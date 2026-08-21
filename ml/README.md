# YOLO 怪物检测 — 操作指南

用一个训练好的 YOLO 模型替代模板匹配，大幅减少背景误检（栅栏、招聘海报等），并对客户端画风差异更鲁棒。

类别（`data.yaml`）：当 `auto_scan: true` 时，`synth.py` 会自动扫描 `monster/` 下所有含 `<name>*.png` 的怪种（当前 33 种）并写回 `data.yaml`，无需手动维护。

---

## 两条路线

**路线 A（推荐）：合成数据，零手工标注。**
把绿幕怪物模板贴到真实游戏背景上自动生成图片+标签。快、无需 labelImg，标签绝对准确。

```
准备背景图 → 合成数据(synth.py) → 训练 → 部署 → 开启 yolo 模式
```

**路线 B：真实截图 + 手工标注。** 更贴近真实画面，但费时。

```
采集截图 → (半自动预标注) → labelImg 修正 → 训练 → 部署 → 开启 yolo 模式
```

两条路线可混用（合成数据打底 + 少量真实截图微调，效果最佳）。

---

## 路线 A：合成数据（推荐）

### A1. 准备背景图（你做）

在目标地图截几张 **干净的原始游戏画面**（怪少/无怪最好），放到：

```
ml/backgrounds/*.png
```

建议 5~15 张，覆盖不同场景区域。背景越多样，模型越不会过拟合背景。分辨率与实际跑 bot 一致（1296×700）。

### A2. 生成合成数据（我可代跑）

```bash
python3 ml/synth.py --n 800          # 生成 800 张（自动 80% train / 20% val）
# 可选：--max_mobs 6  每张最多贴几只   --seed 0  复现
```

流程：
- 若 `data.yaml` 里 `auto_scan: true`，先扫描 `monster/*/` 生成类别并写回 `data.yaml`。
- 把随机数量/朝向/缩放的怪物模板贴到随机背景的地面带上。
- 同时写出 YOLO 标签到 `ml/dataset/labels/{train,val}/`（贴哪儿标签就在哪儿，绝对准确）。

⚠️ 只有 1~2 张模板的怪（如 `green_mushroom`、`spike_mushroom`、`skeleton_*`）合成样本偏少、学得弱，必要时补几张真实截图（路线 B）。

生成后可直接跳到 **步骤 4：训练**。

### A3. 只训练固定几种怪（专用模型）

想要一个只认某几种怪的小模型（更快、更少误检），用 `--classes` 指定怪种，并给它**独立的 data.yaml / 数据集 / 输出权重**，与全量模型互不干扰：

```bash
# 生成数据（只含蓝蜗牛/红蜗牛/绿水灵，不含 player）
python ml/synth.py --n 800 --classes blue_snail,red_snail,slime \
    --data data_snails.yaml --dataset dataset_snails --no-player
python ml/synth.py --n 800 --classes slime,green_mushroom,horny_mushroom,blue_mushroom --data data_horny_mushroom.yaml --dataset dataset_horny_mushroom

# 训练（独立 run 名 + 独立输出权重）
python ml/train.py --data data_snails.yaml --name snails \
    --out models/mob_yolo_snails.pt

python ml/train.py --data data_horny_mushroom.yaml --name horny_mushroom --out models/mob_yolo_horny_mushroom.pt
```

使用时在 config 里把 `yolo_model_path` 指向该权重：

```yaml
monster_detect:
  mode: "yolo"
  yolo_model_path: "models/mob_yolo_snails.pt"
```

> 想让专用模型也检测角色，去掉 `--no-player`（需 ml/player/ 有素材）。

---

## 路线 B：真实截图 + 手工标注

整体流程：

```
采集截图 → (半自动预标注) → labelImg 修正 → 训练 → 部署 → 开启 yolo 模式
```

---

### 步骤 1：采集截图（你做）

在目标地图里截 **原始游戏画面**（不是 viz 截图），放到：

```
ml/dataset/images/train/     # 大部分（~80%）
ml/dataset/images/val/       # 一部分（~20%）用于验证
```

建议：
- 每种怪 **100~300 张**，覆盖不同姿态、朝向、背景、数量。
- 分辨率和实际跑 bot 时一致（1296×700）。
- 文件名随意（如 `gk_001.png`）。

### 步骤 2：半自动预标注（可选，我可代跑）

```bash
python3 ml/prelabel.py
```

用现有模板匹配生成初始框，写到 `ml/dataset/labels/{train,val}/*.txt`。

⚠️ 注意：由于 GMS 模板和你的客户端差异较大，预标注可能只标出很少的框，大部分仍需手动补。可以直接跳到步骤 3 全手动。

### 步骤 3：labelImg 修正 / 标注（你做）

```bash
pip install labelImg
labelImg ml/dataset/images/train  ml/data.yaml
```

- 格式选 **YOLO**。
- 给每只怪画框并选类别（blue_snail / red_snail / slime）。
- 保存后会在 `labels/` 生成同名 `.txt`。
- val 目录同样处理。

---

## 共用步骤（两条路线都要）

## 步骤 4：训练（我做，需要你的机器有 GPU）

```bash
pip install ultralytics
python3 ml/train.py --epochs 100          # 有 GPU 自动用 CUDA
# 或指定：python3 ml/train.py --model yolo11n.pt --imgsz 640 --batch 16
```

训练完成后，最佳权重会自动复制到 `models/mob_yolo.pt`。

## 步骤 5：部署 / 开启（改 config）

```yaml
monster_detect:
  mode: "yolo"
  yolo_model_path: "models/mob_yolo.pt"
  yolo_conf_thres: 0.4     # 越高越严格（框越少）
```

重启 bot。日志会打印 `[YOLO] loaded model ...`。

---

## 回退保护

如果模型文件缺失、或没装 `ultralytics`，引擎会**自动回退到模板匹配**（`color` 模式），并在日志里说明原因——不会崩。

## 新增怪种

**若 `data.yaml` 里 `auto_scan: true`（默认）：**
1. 新建 `monster/<name>/` 并放入 `<name>*.png` 绿幕模板。
2. 重跑 `python3 ml/synth.py`（会自动把新怪加进 `data.yaml` 并生成数据）。
3. 重新训练。

**若 `auto_scan: false`（手动冻结类别）：**
1. 在 `data.yaml` 的 `names` **末尾追加**（不要打乱已有序号）。
2. 补该怪的合成/标注数据。
3. 重新训练。

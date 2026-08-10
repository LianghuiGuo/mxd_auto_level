# 小地图定位 / 路线跟随问题排查记录

本文件记录 normal 模式下小地图检测、角色黄点定位、minimap→route 模板匹配相关的
典型问题、根因和解决办法。遇到"角色在小地图上定位不准 / 路线走不对"时先读这里。

---

## 快速自查清单

按顺序检查，多数问题都能定位：

1. **小地图 ROI 是否对齐？** 看 debug 窗口红框是否严丝合缝贴住小地图内容区
   （不含标题栏、不含外边框、不框到游戏画面）。
2. **黄点是否贴合真实角色？** route map debug 里的黄点应落在角色所在平台，
   不应漂到地图中间或最下方。
3. **`minimap→route` 匹配 score 是否 < 0.4？** 看日志
   `get_player_location_on_global_map ... score=...`。≥ 0.4 会触发"屏幕比例投影"
   兜底，定位就会乱（这不是真实匹配）。
4. **运行时与录制时的 `manual_roi` 是否完全一致？** 这是最容易踩的坑，见下。

---

## 问题 1：小地图 ROI 右下角框超（框到游戏画面）

### 现象
debug 窗口红框左上角对齐了，但右边 / 下边远大于真实小地图，把周围游戏画面
（沙滩、泥土、石墙等棕色地形）也框进来了。

### 根因
这张地图（如"明珠港郊外"）的小地图**没有白色边框**，检测走"棕色框回退"路径。
但小地图周围的游戏画面也是棕色地形，棕色连通域会向外蔓延，把周边地形连进来，
导致检测出的宽高远大于真实小地图。

### 解决
- 已在 `src/utils/common.py` 的 `get_minimap_loc_size` 加入**尺寸约束**：
  超出 `auto_max_w_ratio` / `auto_max_h_ratio`（默认帧宽高的 0.18）的连通域，
  宽高会被**裁剪回上限**（而不是整个丢弃，避免"检测不到"）。
- 可在 config 调整 `minimap.auto_max_w_ratio` / `auto_max_h_ratio`。
- **最可靠**：用固定 `manual_roi` 锁死（见问题 3）。

---

## 问题 2：角色黄点定位漂移（跑到地图中间 / 最下方）

### 现象
route map debug 里黄点位置不对，稳定漂在小地图中间或底部。

### 根因（两个）
1. **黄点检测取了所有黄色像素的质心**。小地图里如果有其他黄色物体
   （黄色的树、黄色地形高光），质心会被拉偏。
2. **录制工具没传入客户端的 `player_color`**，用了默认 `(136,255,255)`，
   而实际客户端黄点可能是 `(50,255,238)` 之类，精确匹配失败后掉进泛黄兜底，
   更易被黄树污染。

### 解决
- `get_player_location_on_minimap` 已改为**取最大黄色连通块的质心**
  （`_largest_blob_centroid`），排除分散的黄树 / 黄地形干扰。
- `tools/routeRecorder.py` 已传入 config 的 `minimap.player_color`。
- 确认 config 里 `minimap.player_color` 是你客户端黄点的真实 BGR 值。

---

## 问题 3（最关键）：运行时与录制时 `manual_roi` 不一致 → 匹配失败

### 现象
- 日志：`minimap→route template match was poor (score=0.889 ≥ 0.4).
  Falling back to a screen→route proportional projection`
- 黄点定位乱、路线跟随不准、角色在小地图上位置错误。

### 根因（实测确认）
`map.png` 和匹配算法本身都没问题（从 map.png 自身截块匹配 score=0.0000）。
问题在**运行时裁出的小地图尺寸和录制时不同**：

| 输入 | 尺寸 | 匹配 score | 结果 |
|------|------|-----------|------|
| 运行时实际裁剪（走了自动检测） | 227×83 | 0.3498 | ❌ 失败 → 投影兜底 → 黄点乱 |
| 裁成录制尺寸 | 212×82 | **0.0942** | ✅ 匹配成功 |

运行时多出的右边 ~15px 是小地图外的游戏画面。尺寸 + 内容都对不上，
`find_pattern_sqdiff` 自然匹配失败，score ≥ 0.4 触发兜底，
`loc_player_global` 的 y 被投影到地图底部 → 黄点跑到小地图最下方。

**为什么尺寸不同**：运行时 `manual_roi` 没生效（走了自动检测，尺寸每次会漂移），
而录制 `map.png` 时用的是固定尺寸。

### 解决（核心结论）
**运行时和录制时必须使用完全相同的固定 `manual_roi`。**

```yaml
minimap:
  manual_roi: [5, 41, 212, 82]   # 用你自己 debug 出的准确值；录制与运行必须一致
  debug_dump: false
```

- 用固定 `manual_roi` 杜绝自动检测的尺寸漂移。
- 如果录制 `map.png` 时的 ROI 和现在运行的不一致，**用统一的固定 ROI 重录一次
  `map.png`**，保证"录制 = 运行"。
- 验证：重跑后日志里 `minimap→route ... score` 应降到 **< 0.4**（本例 ~0.09）。

---

## 如何测量准确的 `manual_roi`

不要从 OS 窗口截图里量（分辨率和引擎实际处理帧不同，换算易错）。改用内置调试导出：

1. config 设 `minimap.manual_roi: null`（让自动检测跑）+ `minimap.debug_dump: true`。
2. 确保游戏窗口已激活、小地图展开可见，运行引擎（或 routeRecorder）。
3. 查看：
   - 日志：`[minimap debug] ... manual_roi: [x, y, w, h]`
     —— 这是**引擎自己坐标系**里的准确值，可直接粘贴。
   - `log/minimap_debug_overlay.png` —— 帧上画出红框，肉眼确认贴合。
   - `log/minimap_debug_crop.png` —— 实际裁出的小地图。
4. 把打印的 `[x, y, w, h]` 填回 `minimap.manual_roi`，再把 `debug_dump` 改回 `false`。

---

## 配置文件优先级提醒

`config/config_custom.yaml` 在 `config/config_default.yaml` **之后**加载，会**覆盖**默认值。
所以实际生效的 `manual_roi` / `player_color` / `debug_dump` 以 custom 为准。
改配置时优先改 custom；default 里改了但 custom 有同名项，会被 custom 盖掉。

---

## 何时干脆用 patrol 模式

如果地图满足以下特征，normal 的小地图模板匹配会天然脆弱，**建议直接用 patrol**：

- **水平直路 / 缓坡**，没有需要爬绳索的多层结构。
- **小地图特征重复、纹理单一**（大片相似的海 / 地形），独特特征少，
  `matchTemplate` 容易漂移。

patrol **不依赖小地图匹配**，配合 auto-jump 过缓坡即可，能彻底绕开本文所有问题。
YOLO 怪物检测和角色定位在 patrol 下照常工作（它们不依赖小地图）。

# 新地图录制注意事项（normal 模式）

本文件是 normal 模式下录制小地图（`map.png`）和路线（`route*.png`）的操作指南与避坑清单。
录制前请完整读一遍；遇到黄点定位漂移 / 路线跟随不准，另见
[`README_minimap_troubleshooting.md`](./README_minimap_troubleshooting.md)。

---

## 0. 先判断：这张图该用 normal 还是 patrol？

**不是所有地图都适合 normal。** 如果新地图满足下面任一特征，`matchTemplate` 小地图匹配会
天然脆弱、容易漂移，**建议直接用 patrol 模式**（不依赖小地图匹配，配 auto-jump 过缓坡即可，
YOLO 怪物 / 角色检测照常工作）：

- **水平直路 / 缓坡**，没有需要爬绳索的多层结构。
- **小地图特征重复、纹理单一**（大片相似的海 / 沙 / 地形），独特特征少。

只有**多层结构、地形特征丰富**的地图才推荐 normal。若不确定，先用 patrol 跑一版看效果。

---

## 1. 最关键铁律：录制 = 运行时

> **录制时和运行时必须使用完全相同的固定 `manual_roi`。**

这是黄点漂到小地图底部的**头号根因**：录制时小地图裁剪尺寸和运行时不一致，
`minimap→map` 模板匹配 score ≥ 0.4，触发"屏幕比例投影"兜底 → 定位乱。

所以**录制前先固定 `manual_roi`**，不要用自动检测（`null`）——自动检测每帧尺寸会漂几像素。

在 `config/config_custom.yaml` 里：

```yaml
minimap:
  manual_roi: [x, y, w, h]   # 你 debug 出的准确值，录制与运行必须一致
  player_color: [50, 255, 238]  # 你客户端黄点的真实 BGR（按需改）
  debug_dump: false
```

---

## 2. 如何测量正确的 `manual_roi`

**不要**从 OS 窗口截图里量（OS 分辨率和引擎处理帧不同，换算易错）。用内置调试导出：

1. 临时设 `minimap.manual_roi: null` + `minimap.debug_dump: true`。
2. 确保游戏窗口已激活、小地图已展开可见。
3. 运行引擎（或 routeRecorder）跑一次，查看：
   - 日志：`[minimap debug] ... manual_roi: [x, y, w, h]` —— **引擎自己坐标系**的准确值，可直接粘。
   - `log/minimap_debug_overlay.png` —— 帧上画出红框，肉眼确认严丝合缝贴住小地图内容区
     （**不含**标题栏、**不含**外边框、**不框到**周围游戏画面）。
   - `log/minimap_debug_crop.png` —— 实际裁出的小地图，确认干净。
4. 把打印的 `[x, y, w, h]` 填回 `minimap.manual_roi`，再把 `debug_dump` 改回 `false`。

---

## 3. 确认 `player_color`（黄点颜色）

- config 里的 `minimap.player_color` 必须是**你客户端黄点的真实 BGR**（本项目实测 `[50, 255, 238]`）。
- 值不对会导致精确匹配失败 → 掉进"泛黄兜底" → 被小地图里的黄树 / 黄地形高光污染，黄点漂。
- routeRecorder 会自动读取 config 的 `player_color`，所以只要 config 设对即可。

---

## 4. 录制 map.png

- `map.png` 是**整张地图的小地图全景**（不是游戏画面截图）。
- 用固定的 `manual_roi` 录，保证和运行时裁剪一致。
- 录制时走遍整张地图，让工具拼出完整全景。

---

## 5. 录制路线 route*.png

- 路线画在小地图坐标系上，用**特定颜色码**标注方向和动作。
- **路线要形成逻辑闭环**：normal 模式按 `route1 → route2 → ...` **顺序循环**走
  （索引 `idx % len(routes)`）。路线不闭环，角色走到头会来回震荡或覆盖不全。
- 多条路线时，确保首尾能自然衔接成一个环。

### 颜色码对照表

颜色码在 `config/config_default.yaml` 的 `route` 段定义，命令格式为
`<左右, 上下, 动作>`。**注意 RGB 顺序（不是 BGR）**，画路线时按下表填色。

**主颜色码 `route.color_code`：**

| RGB | 颜色 | 命令 | 含义 |
|---|---|---|---|
| `255,0,0` | 🔴 红 | `left none none` | 向左走 |
| `0,0,255` | 🔵 蓝 | `right none none` | 向右走 |
| `255,127,0` | 🟠 橙 | `left none jump` | 左走 + 跳 |
| `0,255,255` | 🟦 青 | `right none jump` | 右走 + 跳 |
| `127,255,0` | 💚 青柠 | `none down jump` | 下 + 跳 |
| `255,0,255` | 💜 品红 | `none none jump` | 原地跳 |
| `0,127,255` | 橙青 | `none up jump` | 跳 + 上（爬绳：先跳再按上）|
| `0,255,127` | 🟢 浅绿 | `stop stop stop` | 停止 |
| `255,255,0` | 🟨 黄 | `none none goal` | **路点终点**（切下一条路线）|
| `255,0,127` | 🌸 粉 | `none up teleport` | 上 + 传送（法师）|
| `127,0,255` | 🟪 紫 | `none down teleport` | 下 + 传送 |
| `0,127,0` | 🟩 深绿 | `left none teleport` | 左 + 传送 |
| `139,69,19` | 🟫 棕 | `right none teleport` | 右 + 传送 |

**上下颜色码 `route.color_code_up_down`（可与主码叠加使用）：**

| RGB | 颜色 | 命令 | 含义 |
|---|---|---|---|
| `127,127,127` | ⚪ 灰 | `none up none` | 向上 |
| `255,255,127` | 🟡 浅黄 | `none down none` | 向下 |

### 相关参数

- `route.search_range`（默认 10）：在角色位置周围此半径（px）内找最近的路线颜色。
- `edge_teleport.color_code`（默认 `[255,127,127]`）：**平台边缘**独立标记，靠近时法师传送 /
  其他职业跳跃。
- 卡住时 watchdog 会从 `color_code` 里随机挑一个动作解卡。

> 提示：以上是默认配置，若你在 `config_custom.yaml` 里改过 `route.color_code`，以 custom 为准。

---

## 6. 录完必做验证

1. 用录好的图重跑引擎。
2. 看日志 `minimap→route ... score`：应 **< 0.4**（越接近 0 越好，实测好的地图 ~0.09）。
   - 若 **≥ 0.4**：录制 / 运行 ROI 仍不一致，**用统一固定 ROI 重录**。
3. 看 route map debug 里黄点：应**稳定落在角色所在平台**，不漂到地图中间 / 最下方。
4. 观察角色能否按路线正常巡逻、不卡角、不来回震荡。

---

## 7. 配置优先级提醒

`config/config_custom.yaml` 在 `config/config_default.yaml` **之后**加载，会**覆盖**默认值。
所以实际生效的 `manual_roi` / `player_color` / `debug_dump` **以 custom 为准**。
改配置时优先改 custom；default 改了但 custom 有同名项会被盖掉。

---

## 快速检查清单（录制前逐条确认）

- [ ] 已判断该用 normal 还是 patrol（直路 / 纹理单一 → patrol）
- [ ] `manual_roi` 已用 `debug_dump` 测准并固定（非 `null`）
- [ ] `player_color` 是客户端黄点真实 BGR
- [ ] 录制与运行使用**同一** `manual_roi`
- [ ] 路线形成逻辑闭环
- [ ] 录完验证 `score < 0.4`、黄点贴合角色

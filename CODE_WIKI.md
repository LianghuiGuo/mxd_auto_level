# MapleStory AutoLevelUp - Code Wiki

## 目录
- [1. 项目概述](#1-项目概述)
- [2. 整体架构](#2-整体架构)
- [3. 目录结构](#3-目录结构)
- [4. 核心模块详解](#4-核心模块详解)
  - [4.1 Engine 引擎层](#41-engine-引擎层)
  - [4.2 States 状态层](#42-states-状态层)
  - [4.3 Input 输入层](#43-input-输入层)
  - [4.4 UI 用户界面层](#44-ui-用户界面层)
  - [4.5 Utils 工具层](#45-utils-工具层)
- [5. 关键类与函数说明](#5-关键类与函数说明)
- [6. 有限状态机 (FSM) 详解](#6-有限状态机-fsm-详解)
- [7. 配置系统](#7-配置系统)
- [8. 资源目录说明](#8-资源目录说明)
- [9. 依赖关系](#9-依赖关系)
- [10. 项目运行方式](#10-项目运行方式)
- [11. 辅助工具脚本](#11-辅助工具脚本)
- [12. 核心工作流程](#12-核心工作流程)

---

## 1. 项目概述

**MapleStory AutoLevelUp** 是一个基于计算机视觉（Computer Vision）技术的冒险岛（MapleStory Artale）自动练级机器人。

### 核心特性
- **纯视觉驱动**：不读取游戏内存，仅通过屏幕图像识别进行操作
- **模拟键盘输入**：通过模拟真实键盘输入控制角色
- **有限状态机架构**：使用 FSM 管理多种运行状态（狩猎、符文寻找、符文求解等）
- **用户友好 UI**：基于 PySide6 的图形界面
- **符文自动求解**：自动识别并完成游戏中的符文小游戏
- **自动补给**：HP/MP 药水自动使用
- **自动切换频道**：检测到其他玩家时自动切换频道
- **跨平台支持**：Windows / macOS
- **多语言支持**：支持英文 / 繁体中文

### 技术栈
- **语言**：Python 3.12
- **GUI**：PySide6 (Qt 6)
- **图像处理**：OpenCV 4.11
- **自动化**：pyautogui, pynput, windows-capture (Windows)
- **配置**：YAML (PyYAML, ruamel.yaml)
- **打包**：PyInstaller

---

## 2. 整体架构

```
┌───────────────────────────────────────────────────────────────┐
│                        用户界面层 (UI)                         │
│  ┌──────────────┐    ┌──────────────────┐                     │
│  │  MainWindow  │    │ AutoBotController│                     │
│  └──────┬───────┘    └────────┬─────────┘                     │
└─────────┼─────────────────────┼───────────────────────────────┘
          │ Qt Signals          │ 调用/控制
┌─────────▼─────────────────────▼───────────────────────────────┐
│                        引擎层 (Engine)                         │
│  ┌─────────────────────┐    ┌───────────────────────────┐     │
│  │ MapleStoryAutoBot   │    │  FiniteStateMachine       │     │
│  │  (核心控制器)       │───▶│  (状态机管理器)           │     │
│  └────────┬────────────┘    └─────────────┬─────────────┘     │
│           │                                │ 状态切换          │
│  ┌────────▼─────────┐  ┌──────────────────▼─────────────┐    │
│  │  HealthMonitor   │  │  States (6种运行状态)           │    │
│  │  (血量监控)      │  │  - Hunting 狩猎                │    │
│  └──────────────────┘  │  - FindingRune 寻找符文        │    │
│  ┌──────────────────┐  │  - NearRune 接近符文           │    │
│  │  RuneSolver      │  │  - SolvingRune 求解符文        │    │
│  │  (符文求解器)    │  │  - Patrol 巡逻                 │    │
│  └──────────────────┘  │  - Auxiliary 辅助              │    │
│  ┌──────────────────┐  └────────────────────────────────┘    │
│  │  Profiler        │                                          │
│  │  (性能分析)      │                                          │
│  └──────────────────┘                                          │
└───────────────────────────────────────────────────────────────┘
          │
          │ 调用/控制
┌─────────▼─────────────────────────────────────────────────────┐
│                        输入层 (Input)                          │
│  ┌──────────────────────┐     ┌──────────────────────────┐    │
│  │ KeyBoardController   │     │ GameWindowCapturor       │    │
│  │  (键盘控制器)        │     │  (游戏窗口捕获器)        │    │
│  └──────────┬───────────┘     └────────────┬─────────────┘    │
│             │                               │                  │
│  ┌──────────▼───────────┐                   │                  │
│  │ KeyBoardListener     │                   │                  │
│  │  (键盘监听器)        │                   │                  │
│  └──────────────────────┘                   │                  │
└─────────────────────────────────────────────┼──────────────────┘
                                              │
                          ┌───────────────────▼──────────────────┐
                          │          操作系统 / 游戏窗口          │
                          └──────────────────────────────────────┘
```

---

## 3. 目录结构

```
MapleStoryAutoLevelUp/
├── src/                          # 源代码主目录
│   ├── main.py                   # UI模式入口点
│   ├── engine/                   # 核心引擎模块
│   │   ├── MapleStoryAutoLevelUp.py  # 核心控制器类
│   │   ├── FiniteStateMachine.py     # 有限状态机
│   │   ├── HealthMonitor.py          # HP/MP监控线程
│   │   ├── RuneSolver.py             # 符文求解器
│   │   └── Profiler.py               # 性能分析器
│   ├── states/                   # FSM状态定义
│   │   ├── base_state.py             # 状态基类
│   │   ├── hunting.py                # 狩猎状态
│   │   ├── finding_rune.py           # 寻找符文状态
│   │   ├── near_rune.py              # 接近符文状态
│   │   ├── solving_rune.py           # 求解符文状态
│   │   ├── patrol.py                 # 巡逻状态
│   │   └── auxiliary.py              # 辅助状态
│   ├── input/                    # 输入/输出模块
│   │   ├── KeyBoardController.py     # 键盘模拟控制器
│   │   ├── KeyBoardListener.py       # 功能键监听器
│   │   ├── GameWindowCapturor.py     # Windows窗口捕获
│   │   └── GameWindowCapturorForMac.py  # macOS窗口捕获
│   ├── ui/                       # UI界面模块
│   │   ├── ui.py                     # 主窗口UI
│   │   └── AutoBotController.py      # UI与引擎的中间控制器
│   ├── utils/                    # 工具函数
│   │   ├── common.py                 # 通用工具函数
│   │   ├── global_var.py             # 全局变量
│   │   ├── logger.py                 # 日志模块
│   │   └── ui.py                     # UI工具函数
│   └── legacy/                   # 遗留代码（旧版全屏截图方案）
│       ├── mapleStoryAutoLevelUp_legacy.py
│       ├── KeyBoardController_legacy.py
│       └── mapScanner_legacy.py
├── config/                       # 配置文件目录
│   ├── config_default.yaml           # 默认配置
│   ├── config_data.yaml              # 地图/怪物数据
│   ├── config_macOS.yaml             # macOS配置
│   ├── config_cleric.yaml            # 牧师职业配置示例
│   └── legacy/                       # 旧版配置
├── minimaps/                     # 小地图资源（新版方案）
│   ├── <地图名>/
│   │   ├── map.png                   # 地图缩略图
│   │   ├── route1.png, route2.png... # 路线图（颜色编码）
│   │   └── route_rest.png            # 休息路线
├── maps/                         # 旧版全地图资源
├── monster/                      # 怪物模板图片
│   └── <怪物名>/<怪物名>_N.png      # 各帧动画图片
├── rune/                         # 符文相关模板图片
│   ├── rune.png, rune_1.png...       # 符文图标
│   ├── arrow_*.png                   # 方向箭头模板
│   ├── rune_warning_*.png            # 符文警告模板
│   └── rune_enable_*.png             # 符文启用模板
├── misc/                         # 杂项图片（按钮模板等）
├── numbers/                      # 数字识别模板
├── media/                        # README用图片/动图
├── tools/                        # 辅助工具脚本
│   ├── routeRecorder.py             # 路线录制工具
│   ├── mob_maker.py                 # 怪物模板下载工具
│   ├── AutoDiceRoller.py            # 自动掷骰子（角色创建）
│   └── ...其他实验脚本
├── .github/workflows/            # CI构建工作流
├── requirements.txt              # Python依赖
├── Makefile                      # Make构建脚本
├── build.bat                     # Windows打包脚本
├── README.md / README.zh.md      # 项目说明文档
└── LICENSE                       # 许可证
```

---

## 4. 核心模块详解

### 4.1 Engine 引擎层

#### MapleStoryAutoLevelUp.py - 核心控制器
**文件路径**：[src/engine/MapleStoryAutoLevelUp.py](file:///d:/maplestory_autolevelup/MapleStoryAutoLevelUp/src/engine/MapleStoryAutoLevelUp.py)

这是整个项目的核心大脑，负责协调整个自动挂机流程。

**主要职责**：
1. **初始化与配置加载**：`load_config(cfg)` - 加载YAML配置，准备地图、怪物、路线资源
2. **生命周期管理**：`start()` / `pause()` / `terminate_threads()` - 管理所有子线程
3. **图像处理流水线**：`run_once()` - 每帧处理的主函数
4. **玩家定位**：
   - `get_player_location_by_party_red_bar()` - 通过队伍红血条定位（推荐）
   - `get_player_location_by_nametag()` - 通过名字牌定位（已弃用）
   - `get_player_location_on_global_map()` - 通过小地图确定全局坐标
5. **怪物检测**：
   - `get_monsters_in_range()` - 多模式怪物模板匹配
   - `get_nearest_monster()` - 寻找攻击范围内最近的怪物
6. **路线导航**：
   - `get_nearest_color_code()` - 根据路线图颜色编码获取移动指令
   - `update_cmd_by_route()` - 通过路线更新移动命令
7. **战斗决策**：
   - `update_cmd_by_mob_detection()` - 根据怪物检测更新攻击命令
   - `get_attack_direction()` - 判断攻击方向（左/右）
8. **频道管理**：`channel_change()` - 自动切换频道流程
9. **看门狗**：`is_player_stuck()` - 检测角色是否卡死

#### FiniteStateMachine.py - 有限状态机
**文件路径**：[src/engine/FiniteStateMachine.py](file:///d:/maplestory_autolevelup/MapleStoryAutoLevelUp/src/engine/FiniteStateMachine.py)

管理机器人的运行状态切换。

**核心方法**：
| 方法 | 说明 |
|------|------|
| `add_state(state)` | 注册一个状态 |
| `add_transition(from, to)` | 注册合法的状态转换 |
| `set_init_state(name)` | 设置初始状态 |
| `transit_to(to_state)` | 执行状态转换（含1秒防抖） |
| `do_state_stuff()` | 每帧调用：执行当前状态 + 检查转换条件 |

#### HealthMonitor.py - 血量监控线程
**文件路径**：[src/engine/HealthMonitor.py](file:///d:/maplestory_autolevelup/MapleStoryAutoLevelUp/src/engine/HealthMonitor.py)

独立线程监控 HP/MP/EXP 状态，自动喝药。

**核心逻辑**：
1. `get_hp_mp_exp_percent()` - 通过白色轮廓检测UI上的血条/蓝条/经验条，计算填充百分比
2. `_monitor_loop()` - 独立线程循环，当 HP/MP 低于阈值时自动按键喝药
3. 支持无药回城功能

#### RuneSolver.py - 符文求解器
**文件路径**：[src/engine/RuneSolver.py](file:///d:/maplestory_autolevelup/MapleStoryAutoLevelUp/src/engine/RuneSolver.py)

处理游戏中的符文警告、符文定位和符文小游戏求解。

**核心功能**：
| 方法 | 说明 |
|------|------|
| `is_rune_warning()` | 检测"请消除符文"警告消息 |
| `is_rune_enable()` | 检测"符文已生成"消息 |
| `update_rune_location()` | 在角色附近检测符文图标 |
| `is_in_rune_game()` | 检测是否进入符文箭头小游戏 |
| `solve_rune()` | 识别高亮箭头并按对应方向键 |
| `arrow_hsv_binarized()` | HSV阈值化处理箭头区域（支持色相环绕） |

#### Profiler.py - 性能分析器
**文件路径**：[src/engine/Profiler.py](file:///d:/maplestory_autolevelup/MapleStoryAutoLevelUp/src/engine/Profiler.py)

用于调试性能瓶颈，统计各代码段耗时。

---

### 4.2 States 状态层

所有状态继承自 `base_state.py` 中的 `State` 抽象基类。

**State 生命周期钩子**：
```python
class State:
    def on_enter(self):      # 进入状态时调用一次
    def on_exit(self):       # 离开状态时调用一次
    def check_transitions(self):  # 每帧检查，返回要转换的目标状态名或None
    def on_frame(self):      # 每帧执行的业务逻辑
```

#### 状态转换图

```
                  ┌─────────────────┐
          ┌──────▶│    hunting      │◀──────────────┐
          │       │   (狩猎状态)    │               │
          │       └────────┬────────┘               │
          │                │ 检测到符文警告/启用     │
          │                ▼                        │
          │       ┌─────────────────┐               │
          │       │  finding_rune   │               │
          │       │ (寻找符文状态)  │               │
          │       └──┬──────────┬───┘               │
          │  超时    │          │ 检测到符文        │
          │          │          ▼                   │
          │          │  ┌─────────────────┐         │
          │          │  │   near_rune     │         │
          │          │  │  (接近符文状态) │         │
          │          │  └───────┬─────────┘         │
          │          │   超时    │ 进入小游戏       │
          │          │          │                   │
          │          ▼          ▼                   │
          │       ┌─────────────────┐               │
          └───────│  solving_rune   │───────────────┘
           退出   │ (求解符文状态)  │  求解完成
                  └─────────────────┘

  ┌─────────────────┐      ┌─────────────────┐
  │    patrol       │      │      aux        │
  │   (巡逻状态)    │      │   (辅助状态)    │
  └─────────────────┘      └─────────────────┘
  独立运行模式         独立运行模式(占位)
```

#### 各状态详解

| 状态 | 文件 | 说明 |
|------|------|------|
| **HuntingState** | [hunting.py](file:///d:/maplestory_autolevelup/MapleStoryAutoLevelUp/src/states/hunting.py) | 默认主状态。按路线移动 + 检测怪物攻击 + 看门狗。检测到符文时转入 finding_rune |
| **FindingRuneState** | [finding_rune.py](file:///d:/maplestory_autolevelup/MapleStoryAutoLevelUp/src/states/finding_rune.py) | 继续按路线移动但持续搜索符文图标。检测到符文转入 near_rune，进入小游戏转入 solving_rune |
| **NearRuneState** | [near_rune.py](file:///d:/maplestory_autolevelup/MapleStoryAutoLevelUp/src/states/near_rune.py) | 靠近符文后按 '↑' 键尝试触发。超时退回 finding_rune，进入小游戏转入 solving_rune |
| **SolvingRuneState** | [solving_rune.py](file:///d:/maplestory_autolevelup/MapleStoryAutoLevelUp/src/states/solving_rune.py) | 调用 RuneSolver.solve_rune() 逐个按方向键。退出小游戏后返回 hunting |
| **PatrolState** | [patrol.py](file:///d:/maplestory_autolevelup/MapleStoryAutoLevelUp/src/states/patrol.py) | 简化模式。不依赖小地图/路线图，仅在屏幕左右边界间来回走动并定期攻击 |
| **AuxiliaryState** | [auxiliary.py](file:///d:/maplestory_autolevelup/MapleStoryAutoLevelUp/src/states/auxiliary.py) | 占位状态，目前为空实现 |

---

### 4.3 Input 输入层

#### KeyBoardController.py - 键盘模拟控制器
**文件路径**：[src/input/KeyBoardController.py](file:///d:/maplestory_autolevelup/MapleStoryAutoLevelUp/src/input/KeyBoardController.py)

在独立线程中持续将命令字符串转换为实际键盘按键。

**命令格式**：`"left_right up_down action"`（空格分隔的三元组）
- `left_right`: `left` / `right` / `stop` / `none`
- `up_down`: `up` / `down` / `stop` / `none`
- `action`: `jump` / `teleport` / `attack` / `add_hp` / `add_mp` / `goal` / `none`

**核心特性**：
1. `set_command(cmd_str)` - 设置当前要执行的命令
2. `run()` - 主循环：每帧根据当前命令按下/释放对应按键
3. **Buff 技能自动释放**：按配置的冷却周期自动按键
4. **Force Heal**：强制优先喝血（打断攻击）
5. `is_game_window_active()` - 只在游戏窗口为前台时才发送按键，避免干扰其他操作

#### GameWindowCapturor.py - 游戏窗口捕获器
**文件路径**：[src/input/GameWindowCapturor.py](file:///d:/maplestory_autolevelup/MapleStoryAutoLevelUp/src/input/GameWindowCapturor.py) (Windows)
**文件路径**：[src/input/GameWindowCapturorForMac.py](file:///d:/maplestory_autolevelup/MapleStoryAutoLevelUp/src/input/GameWindowCapturorForMac.py) (macOS)

使用 Windows Graphics Capture API (windows-capture 库) 高效截取游戏窗口画面。

**关键流程**：
1. 通过窗口标题关键词查找游戏窗口
2. 调整窗口大小到指定分辨率（1296×759）
3. 注册帧到达回调函数，带锁存入 frame 缓冲区
4. `get_frame()` - 线程安全地获取最新一帧（BGRA→BGR 转换）

#### KeyBoardListener.py - 功能键监听器
**文件路径**：[src/input/KeyBoardListener.py](file:///d:/maplestory_autolevelup/MapleStoryAutoLevelUp/src/input/KeyBoardListener.py)

使用 pynput 库监听全局功能键（F1/F2/F3/F12）。

| 功能键 | 默认行为 |
|--------|----------|
| F1 | 开始/暂停机器人 |
| F2 | 保存截图到 screenshot/ |
| F3 | 开始/停止录像（UI模式） |
| F12 | 终止程序 |

---

### 4.4 UI 用户界面层

#### AutoBotController.py - UI中间控制器
**文件路径**：[src/ui/AutoBotController.py](file:///d:/maplestory_autolevelup/MapleStoryAutoLevelUp/src/ui/AutoBotController.py)

作为 UI 和 Engine 之间的桥梁，基于 QObject 使用 Qt Signal 传递数据。

**核心 Signals**：
- `debug_image_signal` - 发送调试窗口图像到 UI
- `route_map_viz_signal` - 发送路线图可视化图像到 UI

**核心方法**：
| 方法 | 说明 |
|------|------|
| `start_bot(cfg_path)` | 加载配置并启动引擎线程 |
| `pause_bot()` | 暂停引擎 |
| `take_screenshot()` | 截图 |
| `start/stop_recording()` | 录像控制 |
| `enable/disable_bot_viz()` | 切换调试可视化开关 |
| `terminate_bot()` | 终止所有线程 |

#### ui.py - 主窗口
**文件路径**：[src/ui/ui.py](file:///d:/maplestory_autolevelup/MapleStoryAutoLevelUp/src/ui/ui.py)

基于 PySide6 QMainWindow 的四标签页界面。

**标签页结构**：
| 标签页 | 内容 |
|--------|------|
| **Main** | 控制按钮、攻击设置、键位绑定、Buff设置、地图选择、日志输出 |
| **Advanced Settings** | 从 config_default.yaml 动态生成的高级参数表单 |
| **Game Window Viz** | 调试窗口实时画面（含检测框、血条、FPS等） |
| **Route Map Viz** | 路线图 + 玩家位置实时可视化 |

---

### 4.5 Utils 工具层

#### common.py - 通用工具函数
**文件路径**：[src/utils/common.py](file:///d:/maplestory_autolevelup/MapleStoryAutoLevelUp/src/utils/common.py)

大量的图像处理、配置、操作系统级工具函数。

**核心工具函数分类**：

1. **配置操作**：
   - `load_yaml(path)` / `save_yaml(data, path)` - YAML读写
   - `load_yaml_with_comments(path)` - 带注释解析（用于UI高级设置）
   - `override_cfg(base, override)` - 递归合并配置
   - `get_cfg_diff(base, current)` - 计算配置差异

2. **图像处理**：
   - `find_pattern_sqdiff(img, template, ...)` - 带缓存优化的模板匹配（SQDIFF_NORMED）
   - `nms(monsters, iou_threshold)` - 非极大值抑制，去除重叠检测框
   - `get_iou(box1, box2)` - 交并比计算
   - `get_mask(img, chroma_key)` - 基于色键（默认绿色）生成透明遮罩
   - `to_opencv_hsv(hsv)` - 将标准HSV(0-360,0-100,0-100)转换为OpenCV格式
   - `draw_rectangle(...)` - 在图上画带文字的矩形框
   - `get_bar_percent(bar_img)` - 计算UI血条/蓝条填充率
   - `mask_route_colors(...)` - 从地图上清除路线颜色编码像素

3. **小地图处理**：
   - `get_minimap_loc_size(img)` - 自动检测游戏画面中小地图的位置和大小
   - `get_player_location_on_minimap(minimap, player_color)` - 在小地图上找玩家黄点
   - `get_all_other_player_locations_on_minimap(...)` - 找其他玩家红点

4. **操作系统工具**：
   - `is_mac()` / `is_windows()` - 平台判断
   - `activate_game_window(title)` - 激活游戏窗口到前台
   - `resize_window(title, width, height)` - 调整窗口大小
   - `click_in_game_window(title, pos)` - 游戏窗口内坐标点击
   - `screenshot(img, name)` - 保存截图到 screenshot/ 目录

#### global_var.py - 全局常量
**文件路径**：[src/utils/global_var.py](file:///d:/maplestory_autolevelup/MapleStoryAutoLevelUp/src/utils/global_var.py)

```python
WINDOW_WORKING_SIZE = (1282, 693)  # (宽, 高) - 图像处理统一分辨率
```

#### logger.py - 日志模块
**文件路径**：[src/utils/logger.py](file:///d:/maplestory_autolevelup/MapleStoryAutoLevelUp/src/utils/logger.py)

封装 logging 模块，带颜色终端输出 + 文件保存。

---

## 5. 关键类与函数说明

### 5.1 MapleStoryAutoBot 类

**核心成员变量**：

| 变量 | 类型 | 说明 |
|------|------|------|
| `cfg` | dict | 当前生效的配置字典 |
| `fsm` | FiniteStateMachine | 有限状态机实例 |
| `kb` | KeyBoardController | 键盘控制器 |
| `capture` | GameWindowCapturor | 窗口捕获器 |
| `health_monitor` | HealthMonitor | 血量监控线程 |
| `rune_solver` | RuneSolver | 符文求解器 |
| `profiler` | Profiler | 性能分析器 |
| `img_frame` | np.ndarray | 当前游戏帧 (BGR) |
| `img_route` | np.ndarray | 当前路线图 (RGB) |
| `loc_player` | tuple | 角色在游戏画面中的坐标 (x,y) |
| `loc_player_minimap` | tuple | 角色在小地图中的坐标 |
| `loc_player_global` | tuple | 角色在全局路线图中的坐标 |
| `monsters` | list | 当前帧检测到的怪物列表 |
| `cmd_move_x/y/action` | str | 三元组命令 |
| `idx_routes` | int | 当前使用第几条路线图 |
| `t_last_attack` | float | 上次攻击时间戳 |

**核心方法**：

#### `run_once()` - 每帧主处理流水线
位置：[MapleStoryAutoLevelUp.py#L1539-L1754](file:///d:/maplestory_autolevelup/MapleStoryAutoLevelUp/src/engine/MapleStoryAutoLevelUp.py#L1539-L1754)

这是核心中的核心，每帧按序执行以下步骤：

```
1. 图像预处理
   └─ get_img_frame() → 获取游戏帧 + 尺寸校验 + resize
   └─ 转灰度图

2. 小地图提取
   └─ get_minimap_loc_size() → 检测小地图位置/大小
   └─ 截取 img_minimap

3. 角色定位
   └─ get_player_location_by_party_red_bar() → 队伍红条定位
   └─ get_player_location_on_minimap() → 小地图上黄点定位
   └─ get_player_location_on_global_map() → 小地图模板匹配全局定位

4. 频道切换检查
   └─ is_need_change_channel() → 其他玩家检测
   └─ is_time_to_change_channel() → 定时切换

5. 攻击看门狗
   └─ 检查是否太久没攻击 → 切换频道/回城

6. 状态行为 (FSM)
   └─ fsm.do_state_stuff() → 当前状态的 on_frame + check_transitions

7. 调试可视化 (可选)
   └─ update_info_on_img_frame_debug() → 画所有调试信息
   └─ 视频写入 (如果录像中)
   └─ Profiler 报告输出
```

---

## 6. 有限状态机 (FSM) 详解

### 6.1 FSM 初始化 (MapleStoryAutoBot.__init__)
位置：[MapleStoryAutoLevelUp.py#L123-L138](file:///d:/maplestory_autolevelup/MapleStoryAutoLevelUp/src/engine/MapleStoryAutoLevelUp.py#L123-L138)

```python
# 注册6种状态
self.fsm.add_state(HuntingState    ("hunting"     , self))
self.fsm.add_state(FindingRuneState("finding_rune", self))
self.fsm.add_state(NearRuneState   ("near_rune"   , self))
self.fsm.add_state(SolvingRuneState("solving_rune", self))
self.fsm.add_state(AuxiliaryState  ("aux"         , self))
self.fsm.add_state(PatrolState     ("patrol"      , self))

# 注册合法转换
self.fsm.add_transition("hunting",      "finding_rune")
self.fsm.add_transition("finding_rune", "hunting")
self.fsm.add_transition("finding_rune", "near_rune")
self.fsm.add_transition("finding_rune", "solving_rune")
self.fsm.add_transition("near_rune",    "finding_rune")
self.fsm.add_transition("near_rune",    "solving_rune")
self.fsm.add_transition("solving_rune", "hunting")

self.fsm.set_init_state("hunting")
```

### 6.2 各状态转换条件详解

| 源状态 | 目标状态 | 触发条件 |
|--------|----------|----------|
| hunting | finding_rune | 屏幕出现"符文已生成"或"请消除符文"消息 |
| finding_rune | hunting | 超时未找到符文 |
| finding_rune | near_rune | RuneSolver.loc_rune 不为 None（检测到符文图标） |
| finding_rune | solving_rune | 检测到进入箭头小游戏 |
| near_rune | finding_rune | near_rune_duration（默认10秒）超时 |
| near_rune | solving_rune | 检测到进入箭头小游戏 |
| solving_rune | hunting | 退出了箭头小游戏（符文解完或失败） |

---

## 7. 配置系统

### 7.1 配置文件层级

配置采用**多层覆盖**机制，后加载的覆盖先加载的：

```
1. config_default.yaml       (默认基础配置，不应直接修改)
      │
      ▼ 覆盖
2. config_macOS.yaml         (仅 macOS 生效)
      │
      ▼ 覆盖
3. config_<cfg名>.yaml       (用户自定义，--cfg 参数指定或 UI 保存)
```

### 7.2 主要配置节 (config_default.yaml)

| 配置节 | 说明 |
|--------|------|
| `bot` | 运行模式 (`normal`/`aux`/`patrol`)、攻击模式 (`directional`/`aoe_skill`)、地图名 |
| `key` | 游戏内各功能键位映射（技能/跳/药/回城等） |
| `buff_skill` | Buff技能列表和冷却时间 |
| `directional_attack` / `aoe_skill` | 攻击范围、冷却参数 |
| `health_monitor` | 自动喝药阈值、冷却、无药回城等 |
| `teleport` / `edge_teleport` | 瞬移走路、边缘瞬移参数 |
| `party_red_bar` | 队伍红血条检测参数 (HSV范围、偏移量) |
| `monster_detect` | 怪物检测模式 (`color`/`grayscale`/`contour_only`/`template_free`)、阈值 |
| `channel_change` | 检测到其他玩家换频道 |
| `scheduled_channel_switching` | 定时换频道 |
| `route` | 路线颜色编码表、搜索范围 |
| `rune_warning_cn/eng` | 符文警告消息检测ROI和阈值 |
| `rune_detect` / `rune_find` / `rune_solver` | 符文相关检测参数 |
| `minimap` | 小地图玩家颜色、其他玩家颜色 |
| `game_window` | 窗口标题、目标分辨率、标题栏高度 |
| `ui_coords` | 游戏内UI按钮坐标（菜单、频道、角色选择等） |
| `system` | 各线程FPS限制、服务器(TW/NA)、语言(eng/cn) |
| `profiler` | 性能分析开关 |

### 7.3 路线颜色编码表 (route.color_code)

在 `minimaps/<地图名>/route*.png` 图片上，用不同颜色像素表示移动指令：

| RGB颜色 | 指令 (x y action) | 含义 |
|---------|-------------------|------|
| 255,0,0 | left none none | 向左走 |
| 0,0,255 | right none none | 向右走 |
| 255,127,0 | left none jump | 向左跳 |
| 0,255,255 | right none jump | 向右跳 |
| 127,255,0 | none down jump | 向下跳（跳下平台） |
| 255,0,255 | none none jump | 原地跳 |
| 0,255,127 | stop stop stop | 停止 |
| 255,255,0 | none none goal | 路线终点，切换到下一条路线 |
| 255,0,127 | none up teleport | 向上瞬移 |
| 127,0,255 | none down teleport | 向下瞬移 |
| 0,127,0 | left none teleport | 向左瞬移 |
| 139,69,19 | right none teleport | 向右瞬移 |

**补充上下方向的颜色**：
| RGB颜色 | 指令 |
|---------|------|
| 127,127,127 | none up none |
| 255,255,127 | none down none |

---

## 8. 资源目录说明

### 8.1 minimaps/ - 小地图和路线 (推荐使用)

```
minimaps/<地图名>/
├── map.png          # 该地图的小地图全景图（用于全局定位模板匹配）
├── route1.png       # 第1条路线图（在map.png基础上画颜色编码点）
├── route2.png       # 第2条路线图
├── route_rest.png   # 休息/补充路线（可选）
└── upper_route*.png # 上层路线（如空屋地图）
```

### 8.2 monster/ - 怪物模板

```
monster/<怪物英文名>/
├── <怪物名>_1.png
├── <怪物名>_2.png
└── ...
```
- 每个怪物的多种动画帧模板
- 自动加载水平翻转版本以覆盖两个朝向
- 使用绿色 (0,255,0) 作为色键透明背景

### 8.3 rune/ - 符文模板

| 文件 | 用途 |
|------|------|
| rune.png / rune_1.png / rune_2_cn.png / rune_3.png | 符文本体的分段模板（用于定位符文） |
| rune_warning_cn.png / rune_warning_eng.png | 符文警告消息模板 |
| rune_enable_cn.png / rune_enable_eng.png | 符文启用消息模板 |
| arrow_<方向>_<N>.png | 四个方向各3种变体的箭头模板 |

---

## 9. 依赖关系

### 9.1 requirements.txt

| 包名 | 用途 |
|------|------|
| `opencv-python` | 图像处理、模板匹配、颜色检测、轮廓检测 |
| `numpy` | 矩阵运算、图像数组操作 |
| `pyautogui` | 模拟键盘按键（keyDown/keyUp/press） |
| `pynput` | 监听全局功能键（F1/F2等） |
| `requests` | 下载怪物模板图片（mob_maker工具） |
| `pyyaml` | YAML配置文件读写 |
| `PySide6` | Qt6图形界面（主程序UI） |
| `ruamel.yaml` | 保留注释的YAML解析（用于UI高级设置） |
| `pyinstaller` | 打包为Windows .exe可执行文件 |
| `pywin32` | **Windows专属** - Win32 API调用（窗口操作） |
| `windows-capture` | **Windows专属** - 高效游戏窗口捕获（Graphics Capture API） |
| `pyobjc-framework-Quartz` | **macOS专属** - macOS窗口管理和辅助功能 |

### 9.2 模块内部依赖图

```
src.main (UI入口)
    ├── src.ui.ui (MainWindow)
    │     └── src.ui.AudioBotController
    │           ├── src.engine.MapleStoryAutoLevelUp (MapleStoryAutoBot)
    │           │     ├── src.engine.FiniteStateMachine
    │           │     │     └── src.states.* (6个状态)
    │           │     ├── src.engine.HealthMonitor
    │           │     ├── src.engine.RuneSolver
    │           │     ├── src.engine.Profiler
    │           │     ├── src.input.KeyBoardController
    │           │     ├── src.input.GameWindowCapturor(ForMac)
    │           │     └── src.utils.*
    │           └── src.input.KeyBoardListener
    └── PySide6

src.engine.MapleStoryAutoLevelUp (CLI入口 __main__)
    └── 同上（不经过 UI/AutoBotController）
```

---

## 10. 项目运行方式

### 10.1 环境要求
- **操作系统**：Windows 11 或 macOS
- **Python**：3.12
- **游戏设置**：
  - 窗口模式 (Windowed Mode)
  - 调整为最小分辨率（1296×759 不含标题栏）
  - 左上角打开小地图
  - 创建队伍（按P键，角色头上出现红血条）
  - 角色移动到目标地图

### 10.2 开发者运行

#### 1) 安装依赖
```bash
pip install -r requirements.txt
```

#### 2) 运行 UI 模式（推荐）
```bash
python -m src.main
```
- 启动后在图形界面中配置参数，点击"Start"或按 F1 开始

#### 3) 运行命令行模式（无UI）
```bash
# 基础运行（使用 config/config_default.yaml）
python -m src.engine.MapleStoryAutoLevelUp

# 使用自定义配置
python -m src.engine.MapleStoryAutoLevelUp --cfg cleric

# 禁用调试可视化窗口（省资源）
python -m src.engine.MapleStoryAutoLevelUp --disable_viz

# 录制调试窗口视频
python -m src.engine.MapleStoryAutoLevelUp --record
```

#### 4) 命令行热键
| 热键 | 功能 |
|------|------|
| F1 | 暂停/继续 |
| F2 | 保存截图 (screenshot/) |
| F12 | 终止脚本 |

#### 5) Makefile (类Unix系统)
```bash
make setup    # 创建虚拟环境并安装依赖
make run      # 以 UI 模式运行
```

#### 6) Windows 打包
```batch
build.bat
```
使用 PyInstaller 将项目打包为独立 .exe 文件，输出到 `dist/` 目录。

---

## 11. 辅助工具脚本

### 11.1 routeRecorder.py - 路线录制工具
**运行**：
```bash
python -m tools.routeRecorder --new_map <新地图目录名>
```
**功能**：监听键盘输入，在小地图上记录角色轨迹并生成路线图颜色编码。

| 热键 | 功能 |
|------|------|
| F1 | 暂停/继续录制 |
| F2 | 截图 |
| F3 | 保存当前路线并开始新路线 |
| F4 | 将当前扫描地图保存为 map.png |

**使用流程**：
1. 先让角色走遍整个地图以生成 map.png（按F4保存）
2. 按F3开始录制具体路线
3. 录制完成后可用画图工具手动微调 route*.png

### 11.2 mob_maker.py - 怪物模板下载工具
**运行**：
```bash
python tools/mob_maker.py
```
**功能**：从 maplestory.io API 自动下载指定怪物的所有动画PNG图片，过滤死亡动画帧，保留 stand/move/hit/skill 等动作帧。

### 11.3 AutoDiceRoller.py - 角色创建自动掷骰子
**运行**：
```bash
# 要求 INT=13, 其他=4 (法师)
python -m tools.AutoDiceRoller --attribute 4,4,13,4

# 允许 ? 占位（该属性任意）
python -m tools.AutoDiceRoller --attribute 4,4,?,?
```
属性顺序：`STR, DEX, INT, LUK`

---

## 12. 核心工作流程

### 12.1 启动流程

```
用户点击Start / 命令行运行
    │
    ├─ 加载多层YAML配置（default → macOS → custom）
    ├─ 加载地图/怪物/符文模板图片
    ├─ 启动窗口捕获线程 (GameWindowCapturor)
    ├─ 启动键盘控制器线程 (KeyBoardController)
    ├─ 启动血量监控线程 (HealthMonitor)
    ├─ 初始化 FSM（默认进入 hunting 状态）
    ├─ 确保队伍已创建（检测不到红条则自动按P创建队伍）
    │
    └─ 进入主循环 (loop / run_once)
        │
        ├─ 截取游戏窗口帧
        ├─ 提取小地图
        ├─ 定位角色（红血条 → 小地图黄点 → 全局匹配）
        ├─ 检测其他玩家（判断是否换频道）
        ├─ 执行当前 FSM 状态的 on_frame
        │    ├─ 按路线颜色编码更新移动命令
        │    ├─ 检测怪物并更新攻击命令
        │    └─ 看门狗（卡壳随机动作、攻击超时处理）
        ├─ 检查并执行状态转换
        └─ 输出调试可视化图像（OpenCV窗口或Qt Signal）
```

### 12.2 正常狩猎时的单次决策流程

```
每帧 run_once():
    │
    ├─ [角色定位] loc_player / loc_player_global
    │
    ├─ [FSM on_frame - Hunting]
    │    │
    │    ├─ update_cmd_by_route()
    │    │    └─ get_nearest_color_code()
    │    │         └─ 在路线图玩家周围搜索最近的颜色编码点
    │    │         └─ 解析为 cmd_move_x / cmd_move_y / cmd_action
    │    │
    │    ├─ check_reach_goal()
    │    │    └─ 如果 action == "goal" → idx_routes++（切下一条路线）
    │    │
    │    ├─ update_cmd_by_mob_detection()
    │    │    └─ get_monsters_in_range() → 模板匹配怪物
    │    │         ├─ contour_only: 黑色轮廓模板匹配（默认推荐）
    │    │         └─ NMS 去除重叠框
    │    │    └─ get_nearest_monster(left/right) → 两侧各找最近
    │    │    └─ get_attack_direction() → 决定左/右攻击
    │    │    └─ cmd_action = "attack"（满足冷却条件时）
    │    │
    │    └─ is_player_stuck() → 长时间不动则随机动作
    │
    ├─ kb.set_command(cmd_x + cmd_y + cmd_action)
    │    └─ [KeyBoardController 线程异步执行按键]
    │
    └─ [FSM check_transitions]
         └─ 检测符文消息 → transit_to finding_rune
```

### 12.3 符文处理完整流程

```
hunting (正常狩猎)
    │
    ├─ is_rune_enable() 或 is_rune_warning() 返回 True
    │
    ▼
finding_rune (寻找符文)
    │  ├─ reset RuneSolver
    │  ├─ 继续按路线移动 + 攻击（除非警告要求停止）
    │  └─ 每帧 update_rune_location() → 在角色附近找符文图标
    │
    ├─ 检测到符文 (loc_rune != None) → near_rune
    │  或
    ├─ 直接进入了小游戏 (is_in_rune_game()) → solving_rune
    │  或
    └─ 找不到 → 保持 hunting (超时返回)
    │
    ▼
near_rune (接近符文)
    │  ├─ 计算玩家与符文距离
    │  └─ 足够近时反复按 ↑ 键触发
    │
    ├─ 进入小游戏 → solving_rune
    │  或
    └─ 10秒超时 → finding_rune（重新找）
    │
    ▼
solving_rune (求解符文)
    │  ├─ 停止所有按键，释放所有键
    │  └─ 每帧 solve_rune():
    │       ├─ HSV 二值化箭头区域
    │       ├─ HoughCircles 检测当前高亮的箭头（圆形）
    │       ├─ 4向箭头模板匹配得分最低的方向
    │       └─ press_key(方向键, 0.5s) → 按对应方向
    │
    └─ 不再检测到3个以上圆圈（退出小游戏）→ hunting
```

---

*文档版本：v1.0 | 生成日期：2026-08-04*

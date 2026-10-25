# 红蓝双拼天块检测 / 评测流水线

YOLOv8-Seg 2D 抠图 + RealSense RGB-D 3D 姿态 + HSV 红蓝状态 + 单点激光兜底，
最后输出一个**“这是哪个面”**的最终标签，并配套离线评测与 GitHub CI。

> 当前状态：数据转换已用你的 `recommended/` 跑通；代码已通过单元测试和语法检查。
> 机器上还没安装 `torch / ultralytics / open3d / pyrealsense2`，也没有 NVIDIA GPU，
> 所以**训练和 RealSense 实测尚未执行**。

---

## 1. 目录结构

```
sky_block_pipeline/
├── src/skyblock/                 # 可测试的核心逻辑（不依赖 ultralytics）
│   ├── types.py                  # Intrinsics / Frame / Detection 等数据结构
│   ├── sources.py                # RealSense / 摄像头 / 离线 RGB-D
│   ├── geometry.py               # 反投影、RANSAC 平面、交线、姿态分类
│   ├── color.py                  # HSV 红蓝比例与排布
│   ├── laser.py                  # 单点激光兜底接口
│   ├── decision.py               # 最终面标签 + 判定表（你要改这里）
│   ├── pipeline.py               # 单帧完整流水线
│   ├── visualize.py              # 画 mask / 最小外接矩形 / 3D 中心
│   └── evaluation.py             # 指标计算与报告
├── scripts/
│   ├── coco_to_yolo_seg.py       # COCO JSON -> YOLOv8-seg 数据集
│   ├── train_yolov8_seg.py       # 训练 + 验证
│   ├── infer_realsense.py        # 实时推理 CLI（JSONL 输出）
│   ├── dump_intrinsics.py        # 导出 RealSense 内参 JSON
│   └── evaluate.py               # 离线评测，生成报告
├── tests/                        # pytest（决策表、几何、颜色、评测）
├── docs/evaluation.md            # 预测/GT schema、指标定义、验收线
├── examples/gt_example.json      # GT 示例
├── datasets/                     # 已生成（.gitignore 排除）
├── weights/  runs/               # 训练产物（.gitignore 排除）
├── requirements.txt              # 全量运行时依赖
├── pyproject.toml                # 安装 + ruff + pytest 配置
├── Makefile
└── .github/workflows/ci.yml      # GitHub Actions：ruff + pytest
```

---

## 2. 安装

```bash
cd ~/sky_block_pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# 运行 + 开发（不含 torch/open3d/RealSense）
pip install -e ".[dev]"

# 真正训练/推理时再装全量（有 GPU 时先按 pytorch.org 装 CUDA 版 torch）
pip install -r requirements.txt
# 或者：
pip install -e ".[train,realtime]"
```

RealSense USB 权限：

```bash
sudo apt install librealsense2-udev-rules 2>/dev/null || true
# 或从 librealsense 源码：sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
python -c "import pyrealsense2 as rs; print(rs.context().devices)"
```

---

## 3. 数据转换（已完成，可重跑）

输入是 `~/Downloads/sky_block_labeled_datasets_20261025 (2)/recommended/`：

```bash
python scripts/coco_to_yolo_seg.py --overwrite
# 若要把图片复制进项目（而不是软链接到 Downloads）：
python scripts/coco_to_yolo_seg.py --copy-images --overwrite
# 额外生成标注预览图：
python scripts/coco_to_yolo_seg.py --preview --overwrite
```

输出：

```
datasets/sky_block/
├── images/{train,val}/
├── labels/{train,val}/       # 0 x1 y1 x2 y2 ...（归一化多边形）
├── data.yaml
└── dataset_stats.json
```

已验证：train 23 张 / 18 实例 / 5 张负样本，val 2 张 / 2 实例；多边形面积与
COCO 标注误差 < 0.0001%；负样本生成空 `.txt`（正确）。

> 注意：默认用软链接，**不要直接把 `datasets/` 提交到 GitHub**（会变成坏链接）。
> `.gitignore` 已排除。

### Roboflow（可选）

1. 新建 **Instance Segmentation** 项目；
2. 上传 `recommended/train|val/images` 与各自的 `annotations.json`；
3. 负样本图片也要保留（没有标注是正常的）；
4. 保持现有 train/val 划分，验证集不要增强、不要重新随机划分；
5. 导出 YOLOv8 格式。

数据量很小，建议直接用本地脚本，避免 Roboflow 重新划分。

---

## 4. 训练

```bash
# GPU
python scripts/train_yolov8_seg.py --model yolov8n-seg.pt --epochs 200 --batch 8

# 只有 CPU（会很慢）
python scripts/train_yolov8_seg.py --model yolov8n-seg.pt --epochs 100 --batch 2 \
    --workers 2 --device cpu
```

产出：

```
runs/segment/sky_block_yolov8n_seg/weights/best.pt
weights/best.pt          # 自动复制，推理默认读这里
```

训练结束会打印 mask/box mAP，并自动跑一次 val。

---

## 5. 推理

```bash
# RealSense（RGB + 对齐深度）
python scripts/infer_realsense.py --weights weights/best.pt

# 普通摄像头（无深度，只做 2D + 颜色）
python scripts/infer_realsense.py --weights weights/best.pt --source 0

# 离线 RGB-D
python scripts/dump_intrinsics.py --out realsense_640x480.json
python scripts/infer_realsense.py --weights weights/best.pt \
    --image test.jpg --depth depth.npy --intrinsics realsense_640x480.json \
    --sample-id test_0001 --json-out predictions.jsonl --no-show

# 激光测试
python scripts/infer_realsense.py --weights weights/best.pt \
    --laser-left 0.42 --laser-right 0.51
```

终端每行输出最终面、姿态、颜色、3D 中心和激光状态，例如：

```
det0 | conf=0.93 | 面=SIDE(0.95) | 姿态=折角双面 | (...) | 颜色=红左蓝右 R61%/B39% \
     | 中心=(X=+0.012, Y=-0.034, Z=0.512)m | 激光=折角
```

`--json-out` 写出的 JSONL 就是评测输入。`--sample-id` 用于把预测和 GT 对起来；
流模式下默认是 `frame_000123`。

### 3D 逻辑

1. mask 内像素按深度反投影到相机系（米）：`X=(u-cx)Z/fx, Y=(v-cy)Z/fy`；
2. 深度裁剪 `--zmin/--zmax`，体素下采样，统计离群点剔除；
3. Open3D 迭代 RANSAC：
   - 1 个平面：法向与光轴夹角 ≤ `--top-angle`（默认 35°）→ **顶面**，否则 **单侧面**；
   - 2 个平面：法向夹角 ≥ `--min-dihedral`（默认 15°）→ **折角双面**，并输出交线方向；
4. 3D 中心：mask 内有效点中位数（比均值稳健）。

### 颜色

mask 腐蚀 1 px → HSV → 剔除高光/低饱和（`--highlight-v`、`--min-sat`）→
红 `H∈[0,10]∪[170,180]`、蓝 `H∈[95,135]` → 按质心差判断红左/右、上/下。

### 激光接口

```python
from skyblock import LaserProbe, laser_fallback

state = laser_fallback(D_left, D_right, diff_threshold=0.03)   # "折角" / "单平面" / None

probe = LaserProbe(diff_threshold=0.03)
probe.update(d_left, d_right)      # 机械臂移动时持续刷新
state = probe.classify()
```

真正接硬件时改 `LaserProbe.read()`（串口 / ROS / socket）。3D 点云不足时，
决策表会使用激光结果作为兜底；3D 与激光冲突时会在 reasons 里标记。

### 外参

默认输出 RealSense 光学系（X 右、Y 下、Z 前，米）。有相机→机械臂基座的 4×4
矩阵时：

```bash
python scripts/infer_realsense.py --weights weights/best.pt --extrinsic cam_to_base.npy
```

只变换 3D 中心；姿态分类仍在相机系。若要把法向也转到基座系，在
`geometry.classify_pose` 之后对 `Plane.normal` 乘旋转即可。

---

## 6. 最终面标签（需要你补充/纠正）

最终面只按**几何**分类，定义在 `src/skyblock/decision.py::FaceLabel`：

| 3D 姿态 | 激光 | 最终面 |
|---|---|---|
| 顶面 | – | `TOP` |
| 单侧面 | – | `SIDE` |
| 折角双面 | – | `SIDE`（折角仍属于侧面） |
| 折角双面 | 单平面 | `SIDE`（降置信 + 冲突原因） |
| 未知 | 折角 | `SIDE`（低置信，激光兜底） |
| 未知 | 单平面 | `UNKNOWN`（无法区分顶/侧） |
| 其它 | | `UNKNOWN` |

颜色不再塞进面标签：`color.state`、`red_ratio`、`blue_ratio` 单独输出，
不同红蓝比例以后可以再细分，不会导致标签爆炸。`pose` 仍保留
`顶面 / 单侧面 / 折角双面` 的几何细节。

> 这就是对上一版 `FOLD_A / FOLD_B` 的纠正：它们把“红左蓝右/红右蓝左”编进了
> 面标签，既不好解释，也和“折角属于侧面”冲突。现在统一为 `SIDE`，颜色作为
> 独立字段。若以后要更细的标签（如 `SIDE_RED_LEFT_BLUE_RIGHT`），只需扩展
> `FaceLabel` 和 `decide_face()`，评测与报告会自动适配。

---

## 7. 评测（新数据）

### 7.1 冻结独立测试集

- 新采集，覆盖不同高度/角度/灯光/背景，并包含负样本；
- 按**采集场次**划分 train/val/test，不能按帧随机分（避免同场景泄漏）；
- 每个样本保存 `rgb / depth / intrinsics / gt`；
- 3D 中心真值优先用机械臂 TCP 触碰或标定板；没有绝对真值就只报告重复性 σ，
  不要叫“精度”。

### 7.2 跑评测

```bash
python scripts/evaluate.py \
    --pred predictions.jsonl \
    --gt gt_test.json \
    --out reports/test_2026_11_01 \
    --pick best \
    --center-tolerance-mm 5 10 20
```

输出 `metrics.json`、`report.md`、`confusion_matrix.csv`、`per_sample.csv`。
指标包括：

- 最终面 accuracy / macro-F1 / 混淆矩阵；
- 姿态 accuracy、颜色 accuracy；
- 3D 中心 MAE/RMSE（每轴）、欧氏距离 mean/median/p95、5/10/20 mm 成功率；
- 平面法向角误差、折角夹角误差；
- 激光 accuracy、激光与 3D 一致率；
- 推理延迟 mean/median/p95。

Schema、公式和建议验收线见 [`docs/evaluation.md`](docs/evaluation.md)，
示例 GT 见 [`examples/gt_example.json`](examples/gt_example.json)。

2D 分割的 mAP 仍由 Ultralytics 负责：

```bash
yolo segment val model=weights/best.pt data=datasets/sky_block/data.yaml imgsz=640
```

---

## 8. 测试与 CI

```bash
pytest          # 决策表 / 几何 / 颜色 / 评测 / COCO 转换
ruff check .
```

CI（`.github/workflows/ci.yml`）只跑轻量单元测试和 lint，不跑 RealSense/训练。

---

## 9. 建立 GitHub 仓库

本地已经执行过 `git init -b main`（未提交）。首次提交：

```bash
cd ~/sky_block_pipeline
git add .
git status                 # 确认没有 datasets/ weights/ runs/ *.pt
git commit -m "Initial sky-block detection + evaluation pipeline"
```

到 github.com 新建空仓库，然后：

```bash
git remote add origin git@github.com:<你的用户名>/sky-block-pipeline.git
git push -u origin main
```

本机没有 `gh`，需要的话：

```bash
sudo apt install gh
gh auth login
gh repo create sky-block-pipeline --private --source=. --remote=origin --push
```

数据与权重策略：

- `best.pt` 放 **GitHub Releases**，不进 git；
- 数据集用 DVC / 网盘 / 对象存储，仓库只保留 `data.yaml`、标签、`CHECKSUMS.sha256`；
- 每次评测把 `commit hash + 权重 hash + 数据集 hash + metrics.json` 存档。

---

## 10. 已知限制

1. 验证集只有 2 张且与训练集同场景，不能代表真实精度；必须用新测试集。
2. “2 个平面 = 折角”是按需求实现的；普通立方块斜视角下也会看到顶面+侧面，
   也可能被判成折角。若要区分，改 `decide_face`：两个平面都为侧面才算折角。
3. 顶面/单侧面依赖相机大致俯视；有安装角时调 `--top-angle` 或接外参。
4. 颜色阈值受灯光影响，先用 `--json-out` 看 `red_ratio/blue_ratio` 再调。
5. 深度边缘飞点已用腐蚀 mask、深度裁剪、离群点剔除；仍不稳时优先调
   `--plane-dist`、`--min-plane-points`、`--zmax`。
6. 当前机器无 GPU，训练会很慢；可把 `datasets/sky_block` 拷到 GPU 机器训练，
   再把 `best.pt` 拿回来。

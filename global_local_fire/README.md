# Global–Local Fire Network

此目录独立于 `fire_cascade` 和 `fire_mil`。本提交仅实现代码和合成验收，**未运行正式训练、验证集推理或官方测试**。候选模型以图像级可见火焰 F1 为准，不使用检测 mAP。基础提交：`36a5234e6216f4476e71c26037d2041ec50caa67`。

## 模型与复用

- 使用固定版本 [Ultralytics 8.4.0](https://github.com/ultralytics/ultralytics/tree/v8.4.0) 的 `DetectionModel`、[YOLOv8 P2 配置](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8-p2.yaml)（`m` scale）、`Detect`、`LetterBox`、TAL 分配和检测 box/cls/DFL 损失。上游 Ultralytics 为 AGPL-3.0；部署或分发时需按上游许可要求处理。
- 原生 P2/P3/P4/P5 共享步长 `4/8/16/32` 特征。每层全局平均池化后拼接，共享场景 MLP 产生全局 Fire logit 和逐层通道门控；门控从单位倍率起步，调制 Detect 的输入。
- 对检测头输出的 NMS 前原始 Fire/Smoke logits 按类别做跨尺度 Top-16 均值。Fire Top-K 自带图像级监督；可微融合 MLP 同时接收全局、Fire 和 Smoke 证据。检测、全局、MIL、融合权重分别为 `1/0.5/0.5/1`。图像推理仅用分类分支与融合 logit，不运行框解码或 NMS。
- 未找到由 Ultralytics 官方发布的完整 YOLOv8m-P2 `.pt`；使用[官方 COCO `yolov8m.pt`](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m.pt)，严格逐层迁移配置和张量匹配的连续前缀，P2 新路径、变化后的 PAN、双类头和全局/融合头正常初始化。官方资产下载后实测 **52,136,884 字节**，完整 SHA-256 为 `5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5`。不是完整 P2 预训练，也不使用第三方转换权重。

## 数据与阻塞项

`data/best_f1_07957_20261122` 是 JSON 清单与分类子目录，不是 YOLO TXT 导出。只读取 `train.json`（`images/train/{fire,no_fire}`，8707 张，4164 有火）和 `validation.json`（`images/validation/{fire,no_fire}`，259 张，71 有火）。不重划分、不使用 `calibration` 或 `holdout`。检查路径、文件声明哈希和来源组无 train/validation 交集；正式训练入口还须逐图核对字节 SHA-256。

训练 `boxes` 是归一化 Fire `[x1,y1,x2,y2]`，检测内部编码 Fire=0、Smoke=1；图像级 `y=1` 表示 Fire，烟雾或背景为 `0`。7625 个 Fire 框覆盖 4163 张正图；余下一张有火但无框，仅用图像级正标签，不将其 Fire 检测分支当背景。输入中小量出界四舍五入坐标仅在内存中裁剪。训练与推理均使用同样的 EXIF 校正、RGB 和 768 方形 letterbox。

**当前训练阻塞：数据集没有真实 Smoke 框。** `--smoke-manifest` 必须是经人工核实的独立 JSON，不改原数据。键使用训练清单的 `path` 原字符串，值为该图完整、归一化的 Smoke `xyxy` 框数组；只有确已核实无烟的图才写 `[]`，未复核图不写键。训练要求至少存在真实 Smoke 框；缺失 Smoke 监督时掩蔽该图 Smoke 检测分类损失，不推断负标签、不造框。Smoke-only 图仍是图像级负样本；要学到稳健的 Smoke 检测需要有代表性的已审样本，而不仅是通过最低程序检查。

没有独立 `test`；`holdout` 不是官方测试。沿用 `fire_cascade` 的 `data/test/images`（5313 张）、文件名 → 整数 `0/1` 的提交格式。未来官方推理先做文件名和训练/验证字节去重检查，不访问测试标签。字节检查不能排除重编码、同视频近帧或来源层面的关联；数据集目录名中的历史外部 F1 不是本模型成绩。

## 命令（正式训练须先获批准并备齐真实 Smoke 标注）

从仓库根目录使用现有 CUDA PyTorch 环境；依赖见 `requirements.txt`。将官方权重放到 `global_local_fire/assets/yolov8m.pt`，将真实 Smoke sidecar 放到本地、不提交 GitHub。运行目录必须新建，不能覆盖已有实验。

```powershell
.venv\Scripts\python.exe -m global_local_fire.check --pretrained global_local_fire/assets/yolov8m.pt
.venv\Scripts\python.exe -m global_local_fire.train --pretrained global_local_fire/assets/yolov8m.pt --smoke-manifest global_local_fire/assets/train_smoke_boxes.json --run global_local_fire/runs/run01
.venv\Scripts\python.exe -m global_local_fire.infer --run global_local_fire/runs/run01 --output global_local_fire/runs/run01/submission.json
```

固定 seed=42、100 epoch、768px、native BF16、microbatch=2、8 次累积（通常有效 batch=16）。每轮按验证集固定 0.50 的图像级 F1 更新 `best.pt`（并列保留较早轮次）；每轮保存 `last.pt`、训练 CSV 和完整配置。训练完成后严格重新加载 `best.pt`，仅用同一验证集从固定网格选最终阈值（并列先最接近 0.50，再选较小阈值），与最佳权重摘要一起写入 `validation_threshold.json`。这一验证 F1 不作为独立成绩。官方推理只接受已完成 100 轮且验证阈值与权重匹配的运行；同一运行只允许开始一次，不做测试阈值扫描。

## 本轮验收与风险

- RTX 5060 Laptop 实测一次合成 CUDA BF16 前向与反向：P2–P5 为 `192²/96²/48²/24²`，共 48,960 个位置；检测 logits 为 `[2, 2, 48960]`，四个图像级证据均为 `[2]`。16 个上游层严格迁移（10 层完整骨干和 6 层共享 neck 前缀）。四项损失有限，融合头、只通过融合监督的 Smoke 汇聚输出、全局头、P2 门控、检测分类头和共享特征均有有限非零梯度。
- 合成批次 CUDA 峰值约 **1.64 GiB 已分配 / 1.75 GiB 保留**，不含优化器状态和真实密集目标。满训练保守预计 PyTorch 进程约 **2–4 GiB**，不含桌面/GPU 驱动占用；未跑真实批次或优化器更新，正式峰值未实测，密集目标和同卡程序仍可能导致 OOM。
- Smoke 标注缺失是数据阻塞，不用空标记代替。验证集仅 259 张（71 张正类），阈值和最佳轮次可能对来源分布敏感。是否提升真实场景 F1 尚无证据。

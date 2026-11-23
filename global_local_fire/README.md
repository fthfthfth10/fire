# Global–Local Fire Network

此目录独立于 `fire_cascade` 和 `fire_mil`。本实现以图像级可见火焰 F1 为准，不使用检测 mAP。单类修复基于 `fc8aecaaadb4c1d17be4930c7a769ae3b3482183`；原双类权重不兼容。

## 模型与复用

- 使用固定版本 [Ultralytics 8.4.0](https://github.com/ultralytics/ultralytics/tree/v8.4.0) 的 `DetectionModel`、[YOLOv8 P2 配置](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8-p2.yaml)（`m` scale）、`Detect`、`LetterBox`、TAL 分配和检测 box/cls/DFL 损失。上游 Ultralytics 为 AGPL-3.0；部署或分发时需按上游许可要求处理。
- 原生 P2/P3/P4/P5 共享步长 `4/8/16/32` 特征。每层全局平均池化后拼接，共享场景 MLP 产生全局 Fire logit 和逐层通道门控；门控从单位倍率起步，调制 Detect 的输入。
- 对单类检测头输出的 NMS 前原始 Fire logits 做跨尺度 Top-16 均值。Fire Top-K 自带图像级监督；可微融合 MLP 接收全局 Fire logit 和局部 Fire 证据。检测、全局、MIL、融合权重分别为 `1/0.5/0.5/1`。图像推理不运行框解码或 NMS。
- 使用[官方 COCO `yolov8m.pt`](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m.pt)，严格逐层迁移配置和张量匹配的连续前缀，P2 新路径、变化后的 PAN、单类头和全局/融合头正常初始化。官方资产为 **52,136,884 字节**，SHA-256 为 `5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5`；并非完整 P2 预训练权重。

## 数据

`data/best_f1_07957_20261122` 是 JSON 清单与分类子目录，不是 YOLO TXT 导出。只读取 `train.json`（`images/train/{fire,no_fire}`，8707 张，4164 有火）和 `validation.json`（`images/validation/{fire,no_fire}`，259 张，71 有火）。不重划分、不使用 `calibration` 或 `holdout`。检查路径、文件声明哈希和来源组无 train/validation 交集；正式训练入口还须逐图核对字节 SHA-256。

训练 `boxes` 是归一化 Fire `[x1,y1,x2,y2]`，检测只有类 0 Fire；图像级 `y=1` 表示 Fire，烟雾或背景为 `0`。7625 个 Fire 框覆盖 4163 张正图；余下一张有火但无框，保留三项图像级损失，屏蔽其检测分类负监督。所有无火图（包括仅有烟雾）均为 Fire 检测负样本。不读取 Smoke 框文件，不生成伪标签。输入中小量出界坐标仅在内存中裁剪；训练与推理均使用同样的 EXIF 校正、RGB 和 768 方形 letterbox。

没有独立 `test`；`holdout` 不是官方测试。沿用 `fire_cascade` 的 `data/test/images`（5313 张）、文件名 → 整数 `0/1` 的提交格式。未来官方推理先做文件名和训练/验证字节去重检查，不访问测试标签。字节检查不能排除重编码、同视频近帧或来源层面的关联；数据集目录名中的历史外部 F1 不是本模型成绩。

## 命令

从仓库根目录使用现有 CUDA PyTorch 环境；依赖见 `requirements.txt`。将官方权重放到 `global_local_fire/assets/yolov8m.pt`。运行目录必须新建，不能覆盖已有实验。

```powershell
.venv\Scripts\python.exe -m global_local_fire.check --pretrained global_local_fire/assets/yolov8m.pt
.venv\Scripts\python.exe -m global_local_fire.train --pretrained global_local_fire/assets/yolov8m.pt --run global_local_fire/runs/run02
.venv\Scripts\python.exe -m global_local_fire.infer --run global_local_fire/runs/run02 --output global_local_fire/runs/run02/submission.json
```

固定 seed=42、100 epoch、768px、native BF16、microbatch=2、8 次累积（通常有效 batch=16）。每轮使用验证集固定网格计算图像级 F1，以最高 F1 选 `best.pt`；并列优先最接近 0.50，再选较小阈值，完全并列保留较早轮次。每轮保存 `last.pt`、训练 CSV 和完整配置；结束后严格加载 `best.pt`，将对应阈值和权重摘要写入 `validation_threshold.json`。验证 F1 不是独立测试成绩。

官方推理只接受已完成 100 轮且验证阈值与权重匹配的运行；不做测试阈值搜索。同一运行只允许开始一次。封存提交后，若有用户确认的 `data/test/testA_标注.json`（文件名 → `0/1`，`1` 表示 Fire），精确匹配全部键后生成 `score.json`。也可显式传入 `--confirmed-labels` 指定已核实的 JSON 或 YOLO 标签目录（YOLO 类 0=Fire、类 1=Smoke）。没有确认的真值不生成成绩。

合成验收要求 P2–P5 合计 48,960 个位置，单类检测 logits 为 `[2, 1, 48960]`；检测、Global、MIL、Fusion 都须有有效梯度。验证集仅 259 张（71 张正类），其阈值和最佳轮次可能对来源分布敏感。

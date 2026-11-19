# 图像级可见火焰识别

本项目复用参考仓库 D3 的 COCO 预训练 YOLOv8n 和 D5 的 YOLO 候选、区域上下文裁剪、冻结 OpenAI CLIP ViT-B/32、L2 逻辑回归与级联门控。参考仓库只报告了 YOLOv8n 的正式检测器实验。D3 notebook 的已保存输出使用 Ultralytics 8.4.37，训练 50 轮并从验证结果选择最佳权重。D5 notebook 在 Kaggle 中载入该权重，按原数据的 0=烟、1=火提取裁剪，另有内部验证划分、测试集消融及提示词负样本实验。 [D3 训练 notebook](https://github.com/27100340/dl-forest-fire-detection/blob/main/deliverable3_baseline/baseline.ipynb) · [D5 验证器 notebook](https://github.com/27100340/dl-forest-fire-detection/blob/main/deliverable5_improvement2_probe/improvement2_final.ipynb) · [最终报告](https://github.com/27100340/dl-forest-fire-detection/blob/main/final_report/main.tex)

本地训练集的检测标注仅含类别 0 可见火焰，因此 YOLO 使用单类头，所有 1100 张图参与训练，236 张无火图作为背景。若输入标注含类别 1，YOLO 将它视为背景，CLIP 也只会把无火图里的烟框当作负样本。CLIP 线性头使用全部真实火焰框、每张无火训练图的随机区域及最终 YOLO 在无火训练图上的误检区域。保留 class-balanced、C=1 的 LBFGS 逻辑回归，将拟合后的系数保存为 PyTorch `last.pt`。训练阶段没有验证数据加载器，以第 50 轮的 EMA `last.pt` 为检测器，预测要求类别 0。

固定门槛写在 `config.json`。YOLO 候选置信度为 0.25，NMS IoU 为 0.45，置信度达到 0.70 的火焰框依据 D5 直接通过，其余火焰框的宽高各扩大 20% 后由 CLIP 判别，火焰概率达到 0.50 即通过。任一候选通过即输出整图 1。训练挖掘误检时使用 D5 的 0.10 置信度门槛，并单独记录达到推理门槛的误检数量。

原始数据使用副本入训，COCO ID、文件名、解码尺寸、标注坐标和图像级标注都经过核对。4 个框裁到图像边缘，1 张图片按 EXIF 旋转后写入训练副本。训练和测试间有 5 对同名、画面近乎相同的图片，且文件 SHA-256 不同。按照指定训练和测试清单保留这些图，解读测试 F1 时应考虑重叠。

## 运行

运行环境为 Python 3.10、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128、Ultralytics 8.4.0 和 scikit-learn 1.7.2。使用工作区 `.venv`，并将官方 COCO `yolov8n.pt` 和 OpenAI `ViT-B-32.pt` 放在 `fire_cascade/assets`。YOLO 训练使用 BF16 混合精度。CLIP 使用 OpenAI 原版模型代码，许可证随文件提供。 [Ultralytics 预训练权重](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt) · [OpenAI ViT-B/32 权重](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) · [OpenAI CLIP 源码](https://github.com/openai/CLIP/blob/main/clip/clip.py)

将指定数据放入 `data/train/images` 和 `data/test/images`，将 `train_coco.json`、`train_image.json` 放在 `data/train`。可用于评分的 YOLO 格式测试标注放在 `data/test/labels`。从仓库根目录运行以下命令。已有对应 CUDA PyTorch 虚拟环境时直接执行后两行。

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r fire_cascade\requirements.txt
.\.venv\Scripts\python.exe fire_cascade\run.py prepare --run fire_cascade\runs\new_run
.\fire_cascade\run_pipeline.ps1 -Run fire_cascade\runs\new_run
```

每阶段的日志位于运行目录。输出为 `yolo/weights/last.pt`、`clip/last.pt`、`submission.json`。提交按照完整原始文件名映射到整数 0 或 1。测试推理只有在两个最后权重校验通过后才运行，目录内的测试标记防止重复进行整集推理。提交落盘并核验后，评分步骤才读取 `data/test/labels`，将含类 0 框的图片转为图像级真值，记录本地 F1 与混淆计数。官方平台分数需要独立的正式评分记录。

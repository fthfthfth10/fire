# DINOv2-S/14 Reg4 + ABMIL 可见火焰分类

输入一张图像，输出整数 0 或 1。类别 0 的可见火焰框定义图像正类，只有类别 1 烟雾框或无框的图像为负类。后续正式评估只采用图像级可见火焰 F1。

## 复用与改动

- 骨干使用 Meta 的 `dinov2.hub.backbones.dinov2_vits14_reg`，固定提交 `7764ea0f912e53c92e82eb78a2a1631e92725fc8`。严格加载 LVD-142M ViT-S/14 Reg4 官方原生 checkpoint，取冻结模型的 384 维归一化 CLS 特征。 [固定源码](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8) · [官方模型发布](https://github.com/facebookresearch/dinov2#pretrained-backbones-via-pytorch-hub)
- 图像级聚合和分类直接复用 Apache-2.0 的 `torchmil==1.0.2` 中 `ABMIL`，启用其现成门控注意力。同一个无偏置、单位矩阵初始化的 384→384 线性投影为 ABMIL 和线性实例头提供特征。实例损失可以更新图像分类实际使用的投影，DINO 特征仍可冻结缓存。 [torchmil 1.0.2](https://pypi.org/project/torchmil/1.0.2/) · [ABMIL API](https://torchmil.readthedocs.io/en/main/api/models/abmil/)

`config.json` 固定全图和局部输入为 224×224，Patch14 对应 16×16 个图像补丁。全图保持比例补边，局部使用 224 和 448 原图像素的方块、75% 步长，每个方向覆盖到图像末端。训练和推理共用 EXIF 方向处理、裁剪顺序、缩放和归一化。输出火焰阈值固定为 0.50。 [设计和验收详情](AUDIT.md)

## 环境与官方权重

从仓库根目录运行。将 Meta 公开发布的 `dinov2_vits14_reg4_pretrain.pth` 放到下面的 `$dinoWeights`。加载器校验 88,291,785 字节、完整 SHA-256 `f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb` 和源码指纹，再以 `strict=True` 加载。 [官方原生权重](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth)

```powershell
$dinoCommit = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
$dinoLocal = 'fire_mil/.local'
New-Item -ItemType Directory -Force -Path $dinoLocal, 'fire_mil/weights' | Out-Null
py -3.10 -m venv "$dinoLocal/venv"
$milPython = "$dinoLocal/venv/Scripts/python.exe"
& $milPython -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
& $milPython -m pip install -r fire_mil\requirements.txt

Invoke-WebRequest "https://codeload.github.com/facebookresearch/dinov2/zip/$dinoCommit" -OutFile "$dinoLocal/dinov2.zip"
Expand-Archive -LiteralPath "$dinoLocal/dinov2.zip" -DestinationPath $dinoLocal
$dinoSource = "$dinoLocal/dinov2-$dinoCommit"
$dinoWeights = 'fire_mil/weights/dinov2_vits14_reg4_pretrain.pth'
```

## 静态和合成验收

用合成特征检查实例损失的反向路径，用合成文件名检查提交 JSON。原生权重验收记录完整 SHA-256、严格加载结果和合成图 BF16 特征形状。

```powershell
$env:FIRE_MIL_DINOV2_SOURCE = (Resolve-Path $dinoSource).Path
$env:FIRE_MIL_DINOV2_WEIGHTS = (Resolve-Path $dinoWeights).Path
$env:FIRE_MIL_TEST_TMP = (Resolve-Path $dinoLocal).Path
& $milPython -m unittest discover -s fire_mil -p test_mil.py -v
```

## 正式运行命令

以下命令留待 ChatGPT 复审通过且用户明确批准后执行。训练目录需包含 `images/`、`train_coco.json` 和 `train_image.json`，三者必须配对全部 1100 张图。固定 seed 42 和 50 轮。每轮原子更新同一个 `last.pt` 和 `training.csv`，只记录训练损失和耗时。此分支使用配置版本 3 和 `features-v2` 缓存。同版本以相同参数加 `--resume` 可复用缓存并从最后一个完整轮次继续。旧 DINOv3 运行清单与 checkpoint 会被拒绝。

```powershell
$milRun = 'fire_mil/runs/dinov2-reg4'
& $milPython fire_mil\train.py --train-data data/train --dinov2-source $dinoSource --weights $dinoWeights --run-dir $milRun
```

独立推理入口只接收图像和已完成 50 轮的 `last.pt`。单张输出 0/1。目录模式要求官方图像目录恰有 5313 个文件名唯一的图像。输出键须与扫描到的文件名逐一对应，值须为整数 0/1。暂存 JSON 会完整回读并核验，通过后才发布 `submission.json`。推理入口无需测试真值。

```powershell
& $milPython fire_mil\infer.py --image path/to/image.jpg --dinov2-source $dinoSource --weights $dinoWeights --last "$milRun/last.pt"
& $milPython fire_mil\infer.py --images path/to/images --dinov2-source $dinoSource --weights $dinoWeights --last "$milRun/last.pt" --output "$milRun/submission.json"
```

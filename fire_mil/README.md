# DINOv3-S/16 + ABMIL 可见火焰分类

输入一张图像，输出整数 0 或 1。类别 0 的可见火焰框定义图像正类，只有类别 1 烟雾框或无框的图像为负类。后续正式评估只采用图像级可见火焰 F1。

## 复用与改动

- 骨干直接调用 Meta 的 `dinov3.hub.backbones.dinov3_vits16`，固定提交 `6876159a11b4df116f30f667f8c9888617df0751`。加载 Meta 的 LVD-1689M ViT-S/16 原生 checkpoint `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`，取冻结骨干的 384 维归一化 CLS 特征。源码和权重适用 DINOv3 License。仓库仅提交本项目代码，源码和权重由各自官方入口获取。 [固定源码](https://github.com/facebookresearch/dinov3/tree/6876159a11b4df116f30f667f8c9888617df0751) · [官方权重申请说明](https://github.com/facebookresearch/dinov3#pretrained-models) · [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/LICENSE.md)
- 图像级聚合和分类直接复用 Apache-2.0 的 `torchmil==1.0.2` 中 `ABMIL`，启用其现成门控注意力。同一个无偏置、单位矩阵初始化的 384→384 线性投影为 ABMIL 和线性实例头提供特征。实例损失可以更新图像分类实际使用的投影，DINO 特征仍可冻结缓存。 [torchmil 1.0.2](https://pypi.org/project/torchmil/1.0.2/) · [ABMIL API](https://torchmil.readthedocs.io/en/main/api/models/abmil/)

`config.json` 固定全图和局部输入为 224×224。全图保持比例补边，局部使用 224 和 448 原图像素的方块、75% 步长，每个方向覆盖到图像末端。训练和推理共用 EXIF 方向处理、裁剪顺序、缩放和归一化。输出火焰阈值固定为 0.50。 [设计和验收详情](AUDIT.md)

## 环境与官方权重

从仓库根目录运行。向 Meta 取得下载授权后，把原生 `.pth` 放到下面的 `$dinoWeights`。加载器读取完整 SHA-256，检查 Meta 发布文件名中的 `08c60483` 前缀，再以 `strict=True` 加载。本轮原生端点返回 HTTP 403，官方 Hugging Face 发布需要授权，文件列表提供的是 Transformers Safetensors。原生 checkpoint 的真实加载及合成前向待验收。 [Meta 权重访问 FAQ](https://github.com/facebookresearch/dinov3/issues/108) · [官方 Hugging Face 文件列表](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m/tree/main)

```powershell
$dinoCommit = '6876159a11b4df116f30f667f8c9888617df0751'
$dinoLocal = 'fire_mil/.local'
New-Item -ItemType Directory -Force -Path $dinoLocal, 'fire_mil/weights' | Out-Null
py -3.10 -m venv "$dinoLocal/venv"
$milPython = "$dinoLocal/venv/Scripts/python.exe"
& $milPython -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
& $milPython -m pip install -r fire_mil\requirements.txt

Invoke-WebRequest "https://codeload.github.com/facebookresearch/dinov3/zip/$dinoCommit" -OutFile "$dinoLocal/dinov3.zip"
Expand-Archive -LiteralPath "$dinoLocal/dinov3.zip" -DestinationPath $dinoLocal
$dinoSource = "$dinoLocal/dinov3-$dinoCommit"
$dinoWeights = 'fire_mil/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
```

## 静态和合成验收

常规用例用合成特征检查实例损失的反向路径，用合成文件名检查提交 JSON。原生权重验收用例需要获准的 Meta 文件，验收时记录完整 SHA-256、严格加载结果和合成 BF16 前向形状。

```powershell
$env:FIRE_MIL_DINOV3_SOURCE = (Resolve-Path $dinoSource).Path
$env:FIRE_MIL_TEST_TMP = (Resolve-Path $dinoLocal).Path
# 已有获准的真实权重时设置下一行，额外校验官方权重加载和合成图前向。
# $env:FIRE_MIL_DINOV3_WEIGHTS = (Resolve-Path $dinoWeights).Path
& $milPython -m unittest discover -s fire_mil -p test_mil.py -v
```

## 正式运行命令

以下命令留待原生权重验收、ChatGPT 复审通过且用户明确批准后执行。训练目录需包含 `images/`、`train_coco.json` 和 `train_image.json`，三者必须配对全部 1100 张图。固定 seed 42 和 50 轮。每轮原子更新同一个 `last.pt` 和 `training.csv`，只记录训练损失和耗时。使用相同参数加 `--resume` 可复用本版本已写入的特征缓存，并从最后一个完整轮次继续。共享投影使用配置版本 2，恢复和推理均拒绝旧版 checkpoint。

```powershell
$milRun = 'fire_mil/runs/first'
& $milPython fire_mil\train.py --train-data data/train --dinov3-source $dinoSource --weights $dinoWeights --run-dir $milRun
```

独立推理入口只接收图像和已完成 50 轮的 `last.pt`。单张输出 0/1。目录模式要求官方图像目录恰有 5313 个文件名唯一的图像。输出键须与扫描到的文件名逐一对应，值须为整数 0/1。暂存 JSON 会完整回读并核验，通过后才发布 `submission.json`。推理入口无需测试真值。

```powershell
& $milPython fire_mil\infer.py --image path/to/image.jpg --dinov3-source $dinoSource --weights $dinoWeights --last "$milRun/last.pt"
& $milPython fire_mil\infer.py --images path/to/images --dinov3-source $dinoSource --weights $dinoWeights --last "$milRun/last.pt" --output "$milRun/submission.json"
```

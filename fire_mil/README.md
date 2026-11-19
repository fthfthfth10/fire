# DINOv3-S/16 + ABMIL 可见火焰分类

输入一张图像，输出整数 0 或 1。类别 0 的可见火焰框定义图像正类，只有类别 1 烟雾框或无框的图像为负类。后续正式评估只采用图像级可见火焰 F1。

## 复用与改动

- 骨干直接调用 Meta 的 `dinov3.hub.backbones.dinov3_vits16`，固定提交 `6876159a11b4df116f30f667f8c9888617df0751`。加载 Meta 的 LVD-1689M ViT-S/16 原生 checkpoint `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`，取冻结骨干的 384 维归一化 CLS 特征。源码和权重适用 DINOv3 License。仓库仅提交本项目代码，源码和权重由各自官方入口获取。 [固定源码](https://github.com/facebookresearch/dinov3/tree/6876159a11b4df116f30f667f8c9888617df0751) · [官方权重申请说明](https://github.com/facebookresearch/dinov3#pretrained-models) · [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/LICENSE.md)
- 图像级聚合和分类直接复用 Apache-2.0 的 `torchmil==1.0.2` 中 `ABMIL`，启用其现成门控注意力。本项目增加共享图块规则、训练框到图块的监督、独立线性实例头、特征缓存和训练及提交入口。 [torchmil 1.0.2](https://pypi.org/project/torchmil/1.0.2/) · [ABMIL API](https://torchmil.readthedocs.io/en/main/api/models/abmil/)

`config.json` 固定全图和局部输入为 224×224。全图保持比例补边，局部使用 224 和 448 原图像素的方块、75% 步长，每个方向覆盖到图像末端。训练和推理共用 EXIF 方向处理、裁剪顺序、缩放和归一化。输出火焰阈值固定为 0.50。 [设计和验收详情](AUDIT.md)

## 环境与官方权重

从仓库根目录运行。已用 Python 3.10、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128 和 RTX 5060 Laptop 验证合成前向。向 Meta 取得下载授权后，把原生 `.pth` 放到下面的 `$dinoWeights`。Hugging Face 发布的 Safetensors 对应不同的加载接口。

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

## 仅做合成验收

```powershell
$env:FIRE_MIL_DINOV3_SOURCE = (Resolve-Path $dinoSource).Path
$env:FIRE_MIL_TEST_TMP = (Resolve-Path $dinoLocal).Path
# 已有获准的真实权重时设置下一行，额外校验官方权重加载和合成图前向。
# $env:FIRE_MIL_DINOV3_WEIGHTS = (Resolve-Path $dinoWeights).Path
& $milPython -m unittest discover -s fire_mil -p test_mil.py -v
```

## 正式运行命令

以下命令留待 ChatGPT 审核通过且用户明确批准后执行。训练目录需包含 `images/`、`train_coco.json` 和 `train_image.json`，三者必须配对全部 1100 张图。固定 seed 42 和 50 轮。每轮原子更新同一个 `last.pt` 和 `training.csv`，只记录训练损失和耗时。使用相同参数加 `--resume` 可复用已写入的特征缓存，并从最后一个完整轮次继续。

```powershell
$milRun = 'fire_mil/runs/first'
& $milPython fire_mil\train.py --train-data data/train --dinov3-source $dinoSource --weights $dinoWeights --run-dir $milRun
```

独立推理入口只接收图像和已完成 50 轮的 `last.pt`。单张输出 0/1。目录输出按原始文件名映射整数 0/1 的 `submission.json`。

```powershell
& $milPython fire_mil\infer.py --image path/to/image.jpg --dinov3-source $dinoSource --weights $dinoWeights --last "$milRun/last.pt"
& $milPython fire_mil\infer.py --images path/to/images --dinov3-source $dinoSource --weights $dinoWeights --last "$milRun/last.pt" --output "$milRun/submission.json"
```

# 设计与验收审计

## 来源与兼容性

| 来源 | 核对结果及实现决定 |
| --- | --- |
| 现有 `fire_cascade/` | 项目已有 COCO 和图像级标签一致性检查，确认按 EXIF 定向尺寸配对。本目录采用相同的数据约定，YOLO+CLIP 的检测、裁剪和结果文件都独立于新方案。 |
| Meta DINOv2 `7764ea0f` | 使用官方 `dinov2_vits14_reg` 工厂与 `forward_features()["x_norm_clstoken"]`。确认 ViT-S/14 为 14 像素补丁、384 维、4 个寄存器令牌。224×224 输入产生 256 个图像补丁。加载器校验源码指纹、权重字节数和完整 SHA-256，再严格匹配整个 state dict。 |
| torchmil 1.0.2 | 直接复用 `ABMIL(in_shape=(384,), att_dim=128, gated=True)`。ABMIL 和线性实例头接收同一个 384→384 线性投影输出 `[B,N,384]`。ABMIL 还接收布尔有效掩码 `[B,N]`，返回 `[B]` 图像 logits。实例头返回 `[B,N]`。PyPI 标注 Python ≥3.10 和 Apache-2.0。 |
| AttentionDeepMIL 参考实现 | 核对了原始论文配套仓库。torchmil 已有相同方法的可安装实现及批量掩码接口，适配本项目的变长图块袋。 |

[现有主分支](https://github.com/fthfthfth10/fire/tree/main/fire_cascade) · [DINOv3 基准提交](https://github.com/fthfthfth10/fire/commit/df252e135f4f114a64aa7bde85156a85f88203c2) · [Meta DINOv2 工厂源码](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/hub/backbones.py) · [torchmil ABMIL](https://torchmil.readthedocs.io/en/main/api/models/abmil/) · [AttentionDeepMIL](https://github.com/AMLab-Amsterdam/AttentionDeepMIL)

## 权重下载完整性

原下载残片为 65,782,002 字节。Meta 官方端点返回 `206 Partial Content` 和 `Content-Range: bytes 65782002-88291784/88291785`。独立接收的尾段为 22,509,783 字节，响应与长度核验后再拼接成新文件。原残片 SHA-256 复核一致。完整文件为 88,291,785 字节，ZIP 记录 CRC 校验通过，实测 SHA-256 为 `f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb`，已写入配置。 [Meta 官方发布](https://github.com/facebookresearch/dinov2#pretrained-backbones-via-pytorch-hub) · [权重下载源](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth)

## 坐标、标签和训练路径

1. EXIF 转正后的原图坐标为共同参考系。由 COCO 的左上角和宽高转换到 `xyxy` 并裁到图像边界。训练加载时校验解码尺寸。网格边界锚定保证每个原图像素属于至少一个局部块。
2. 图块对火焰框的覆盖率为相交面积除以火焰框面积。达到 50% 的局部块标为正，部分相交块忽略，完全不相交块标为负。若某个框没有达到 50% 的图块，用覆盖率最大的局部块提供正监督。不同框冲突时正标签优先。全图只参与图像级 ABMIL。
3. 图像 BCE 由图像标签监督，正类权重按训练侧负正比例计算。辅助实例 BCE 对批内存在的正负两类分别求均值后合并，权重固定为 0.30。烟框不参与判正。推理以图像 logit 的 sigmoid 和固定 0.50 门槛生成 0/1。
4. 编码器冻结，每个训练图像的特征只需生成一次。运行清单绑定配置、训练文件内容和权重。训练使用全部样本和固定 50 轮，按轮次记录损失。断点保留模型、优化器、完整轮次和日志记录，恢复时重新生成相同轮次顺序。训练脚本没有验证集、早停、best 选择或测试读取路径。
5. 沿用审核基准的无偏置、单位矩阵初始化的共享投影。图像损失和实例损失都可更新投影，实例损失权重为 0.30，投影输出维数为 384。DINOv2 配置为版本 3，特征缓存目录为 `features-v2`。DINOv3 运行身份和旧 checkpoint 在恢复及推理时被拒绝。
6. 目录模式在加载模型前检查 5313 张图和文件名唯一性，大小写变体也视为冲突。提交键集合须和输入文件名完全一致，值须为整数 0/1。暂存 JSON 在发布前检查重复键、缺项、多余项及内容一致性。

## 本轮检查结果

| 检查 | 结果 |
| --- | --- |
| 静态检查 | Python 源码解析、训练和推理 CLI 帮助通过。对照基准确认 seed、50 轮、全部 1100 张、`last.pt`、阈值 0.50、图块几何、MIL 和损失配置沿用。训练入口改用 `--dinov2-source` 和新缓存目录。 |
| 合成用例 | 14 项全部通过。实例损失单独反传时共享投影收到有限非零梯度。钩子确认 ABMIL 和实例头接收同一个投影输出。验证旧版 checkpoint 和运行清单被拒绝、错误权重大小和 SHA-256 被拒绝。提交校验使用 5313 个合成文件名。 |
| 官方预训练 checkpoint | `strict=True` 加载通过，缺失键和意外键均为零。RTX 5060 Laptop 上的 16×3×224×224 合成批次在 CUDA BF16 autocast 下通过，补丁卷积输出为 BF16，归一化 CLS 输出为 FP32 `[16,384]` 且元素均有限。该前向峰值分配显存为 145.3 MiB。1024×512 合成图经集成适配器得到有限的 BF16 `[25,384]` 缓存特征。 |
| 执行边界 | 反向传播只针对合成特征的实例损失。未运行 `optimizer.step`、真实图像训练、测试集推理或 F1 计算。 |

## 尚存风险

- 审核基准按训练标注尺寸估算 BF16 特征净体积约 66.0 MiB。编码器按 16 块分批，MIL 每批上传 8 袋的特征。新增投影后的实际训练显存及全量缓存耗时待验证。
- 规则保证火焰区域进入局部块及训练标签，极小火焰仍可能只占 ViT 补丁很少一部分。冻结骨干的火焰可分性和跨监控背景的泛化尚未评估。
- 本机未安装可选 xFormers，合成验收使用官方模型的 PyTorch 注意力路径。
- 正式训练需等待 ChatGPT 复审和用户明确批准。

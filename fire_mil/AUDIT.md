# 设计与验收审计

## 来源与兼容性

| 来源 | 核对结果及实现决定 |
| --- | --- |
| 现有 `fire_cascade/` | 项目已有 COCO 和图像级标签一致性检查，确认按 EXIF 定向尺寸配对。本目录采用相同的数据约定，YOLO+CLIP 的检测、裁剪和结果文件都独立于新方案。 |
| Meta DINOv3 `6876159a` | 核实 ViT-S/16 为 16 像素补丁、384 维、4 个寄存器令牌。使用官方 `dinov3_vits16` 工厂和 `forward_features()["x_norm_clstoken"]`。直接导入同一工厂以避开 `hubconf.py` 的无关检测、分割和文本依赖。生产加载校验源码指纹、原生文件的官方 SHA-256 前缀及完整 state dict 键和形状。运行清单另记完整权重 SHA-256。 |
| torchmil 1.0.2 | 直接复用 `ABMIL(in_shape=(384,), att_dim=128, gated=True)`。ABMIL 和线性实例头接收同一个 384→384 线性投影输出 `[B,N,384]`。ABMIL 还接收布尔有效掩码 `[B,N]`，返回 `[B]` 图像 logits。实例头返回 `[B,N]`。PyPI 标注 Python ≥3.10 和 Apache-2.0。 |
| AttentionDeepMIL 参考实现 | 核对了原始论文配套仓库。torchmil 已有相同方法的可安装实现及批量掩码接口，适配本项目的变长图块袋。 |

[现有主分支](https://github.com/fthfthfth10/fire/tree/main/fire_cascade) · [Meta 工厂源码](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/hub/backbones.py) · [Meta 权重访问 FAQ](https://github.com/facebookresearch/dinov3/issues/108) · [torchmil ABMIL](https://torchmil.readthedocs.io/en/main/api/models/abmil/) · [AttentionDeepMIL](https://github.com/AMLab-Amsterdam/AttentionDeepMIL)

## 坐标、标签和训练路径

1. EXIF 转正后的原图坐标为共同参考系。由 COCO 的左上角和宽高转换到 `xyxy` 并裁到图像边界。训练加载时校验解码尺寸。网格边界锚定保证每个原图像素属于至少一个局部块。
2. 图块对火焰框的覆盖率为相交面积除以火焰框面积。达到 50% 的局部块标为正，部分相交块忽略，完全不相交块标为负。若某个框没有达到 50% 的图块，用覆盖率最大的局部块提供正监督。不同框冲突时正标签优先。全图只参与图像级 ABMIL。
3. 图像 BCE 由图像标签监督，正类权重按训练侧负正比例计算。辅助实例 BCE 对批内存在的正负两类分别求均值后合并，权重固定为 0.30。烟框不参与判正。推理以图像 logit 的 sigmoid 和固定 0.50 门槛生成 0/1。
4. 编码器冻结，每个训练图像的特征只需生成一次。运行清单绑定配置、训练文件内容和权重。训练使用全部样本和固定 50 轮，按轮次记录损失。断点保留模型、优化器、完整轮次和日志记录，恢复时重新生成相同轮次顺序。训练脚本没有验证集、早停、best 选择或测试读取路径。
5. 审核基准 `4cf12520ea106765bba023f4d97b5c47127ae26a` 中，实例头只更新自身参数。新增无偏置、单位矩阵初始化的共享投影，使图像损失和实例损失可共同更新投影。损失权重仍为 0.30，投影输出维数仍为 384。配置版本 2 使旧 checkpoint 在恢复及推理时被拒绝。
6. 目录模式在加载模型前检查 5313 张图和文件名唯一性，大小写变体也视为冲突。提交键集合须和输入文件名完全一致，值须为整数 0/1。暂存 JSON 在发布前检查重复键、缺项、多余项及内容一致性。

## 本轮检查结果

| 检查 | 结果 |
| --- | --- |
| 静态检查 | Python 源码通过解析，推理 CLI 帮助正常。训练脚本无改动，固定 50 轮、全部 1100 张与单一 `last.pt` 协议沿用审核基准。 |
| 合成用例 | 12 项通过。仅实例损失反传后共享投影梯度有限、非零。前向钩子确认 ABMIL 和实例头接收同一个投影输出。5313 个合成文件名的 JSON 往返通过，并验证数量错误、重复文件名、缺项和多余项、非法标签、重复 JSON 键及损坏内容均被拒绝。 |
| 官方预训练 checkpoint | 1 项跳过。Meta 原生端点返回 HTTP 403，本地未发现匹配文件或可用的 Hugging Face 授权凭据。官方 Hugging Face 文件列表受访问限制，权重为 Transformers Safetensors。原生文件的完整 SHA-256、严格加载和合成编码前向仍待验收。本轮使用无效文件验证摘要拒绝路径，未使用随机骨干代替。 |
| 执行边界 | 反向传播只针对合成特征的实例损失。未运行 `optimizer.step`、真实图像训练、测试集推理或 F1 计算。 |

[Meta 官方授权渠道](https://github.com/facebookresearch/dinov3/issues/108) · [官方 Hugging Face 权重格式与访问状态](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m/tree/main)

## 尚存风险

- 审核基准按训练标注尺寸估算 BF16 特征净体积约 66.0 MiB。编码器按 16 块分批，MIL 每批上传 8 袋的特征。新增投影后的实际训练显存及全量缓存耗时待验证。
- 规则保证火焰区域进入局部块及训练标签，极小火焰仍可能只占 ViT 补丁很少一部分。冻结骨干的火焰可分性和跨监控背景的泛化尚未评估。
- 真正训练还取决于官方权重加载检查、ChatGPT 审核和用户明确批准。

# UDAF-Net

不确定性分布感知的脑电-表情融合情感识别

Uncertainty Distribution Alignment and Fusion Network for EEG–Face Emotion Recognition

## 包含内容

- EEGNet + ResNet18 分阶段编码；融合训练阶段冻结编码器参数和 BatchNorm。
- UDA：残差均值、输入相关对角方差、梯度隔离、试次语义中心、Gaussian MLS。
- RGF：样本内容与对数相对方差共同压缩，双路 sigmoid 通道门控。
- 均值锚点与高斯对偶采样；先平均类别概率，再聚合同一试次的观测。
- full / base / uda / rgf 四种结构配置，对应论文表4的配置定义。
- EAV / DEAP 数据准备接口、原始记录分组检查、SD / LOSO 清单生成。
- 分阶段训练、验证集或固定轮次选模、epoch边界续跑、单折评价、跨折汇总。
- 单元测试和不下载数据/权重的 CPU 合成数据全流程检查。

不包含原始人脸视频、脑电数据、模型权重、私人实验日志、论文PDF、已发表结果表或其他研究分支。
本包未打包论文表2/3的全部外部对比方法，也不宣称恢复了历史checkpoint。

## 安装与快速检查

建议 Python 3.10；验证环境的准确版本见 `docs/VALIDATION.md`。
先按自己的CPU/CUDA环境安装匹配的 PyTorch / torchvision，再在解压目录执行：

```bash
python -m pip install -e '.[test]'
python -m pytest tests/ -x -q
python -m udaf_net.smoke --output /tmp/udaf_synthetic_demo
```

`--output` 应为不存在的新目录；合成检查默认CPU且不下载ImageNet权重。
其分数没有科研含义，不能写入论文。真实训练默认需访问torchvision ImageNet权重或已有本地缓存。

## 数据

先阅读 [数据格式与协议](docs/DATA.md)。统一输入是一行一个情感试次的CSV，以及每个试次一个NPZ。
训练脚本按试次逐个读取数据，不会一次加载全部被试到内存。清单中必须保留原始20s/60s记录编号，
程序会拒绝原始记录跨集合、重复试次或LOSO被试泄漏。数据缺失清单、原始分组编号需由数据持有者核实。

EAV：

```bash
python -m udaf_net.prepare eav --source /path/to/EAV --subject 1 \
  --group-map /path/to/subject01_groups.csv --output data/eav_s01
python -m udaf_net.splits --dataset eav --protocol subject_dependent --subject 1 \
  --inventories data/eav_s01/manifest.csv --output manifests/eav_s01.csv
```

DEAP（valence和arousal分别准备/运行）：

```bash
python -m udaf_net.prepare deap --source /path/to/s01.dat --subject 1 \
  --frames /path/to/subject01_center_frames --target valence --output data/deap_v_s01
python -m udaf_net.splits --dataset deap --protocol subject_dependent --subject 1 \
  --inventories data/deap_v_s01/manifest.csv --output manifests/deap_v_s01.csv
```

如果选用验证集选模，在 `splits` 中显式添加 `--validation-ratio 0.2`（只是使用示例，**不是论文已报告的比例**），
并将独立YAML的 `selection` 改成 `validation_accuracy`。
没有验证集时，默认 `final_epoch`，不读取测试集来选择epoch；不得与历史test-as-validation分数混称。

## 训练与评价

```bash
python -m udaf_net.train --config configs/eav.yaml \
  --manifest manifests/eav_s01.csv --output runs/eav_s01 --device cuda
```

正式长任务可使用：

```bash
nohup python -m udaf_net.train --config configs/eav.yaml \
  --manifest manifests/eav_s01.csv --output runs/eav_s01 --device cuda \
  > eav_s01.log 2>&1 &
```

中断后重跑同一命令，从最近完成的epoch续跑；已完成阶段直接载入。配置或CSV改变时必须使用新的输出目录。
不要同时向同一个输出目录启动两个任务。使用GPU前检查其他任务占用。

```bash
python -m udaf_net.train --config configs/eav.yaml \
  --manifest manifests/eav_s01.csv --output runs/eav_s01 --device cuda --evaluate-only
python -m udaf_net.summarize runs/eav_s*/test_metrics.json
```

LOSO：向 `splits` 传入全部42人（EAV）或22人（DEAP）的清单，设置 `--protocol loso --subject 留出编号`。
复制配置并把 `protocol` 改成 `loso`。每个留出被试使用独立运行目录，编码器也必须在本折训练被试上重新训练。
不应复用被试内协议的编码器。全折结果以百分数、样本标准差 `ddof=1` 汇总，勿将少数折当全数据集结果。

## 消融

复制对应数据集YAML，修改 `model.variant`：

| variant | 概率表示 | 融合 |
|---|---|---|
| `base` | 无 | 双向交叉注意力 |
| `uda` | 有 | 双向交叉注意力处理分布样本 |
| `rgf` | 无 | 内容驱动通道门控，方差分支禁用 |
| `full` | 有 | 内容与相对方差条件通道门控 |

相同清单的消融可通过 `--encoders-from runs/eav_s01` 复用同一组已训练编码器；
该参数读取源运行的 `face_finetune/selected.pt`，程序验证清单哈希完全一致。
每个变体仍需自己的新输出目录。主模型与消融是本次整理实现，未复现原表4数值。

## 公式与文件

| 论文公式 | 实现 |
|---|---|
| (1) 编码与投影 | `encoders/`、`model.UDAFNet` |
| (2)–(3) 后验与方差 | `model.Posterior` |
| (4)–(5) 试次中心/均值/异方差 | `losses.objectives` |
| (6)–(7) Gaussian MLS/关系损失 | `losses.objectives`、`losses.relation_loss` |
| (8) 对偶采样 | `model.antithetic` |
| (9)–(10) RGF | `model.ReliabilityGuidedFusion` |
| (11) 概率集成 | `model.UDAFNet.forward`、`metrics.classification_metrics` |
| (12) 联合损失 | `losses.objectives` |

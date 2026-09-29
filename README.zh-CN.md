# EMO-R3 + SEPM 启发的自由比例视觉 TokenDrop

这是准备上传 GitHub 的**源码包**，不含数据、模型权重或实验输出。
主方法是：本地按 EMO-R3 原奖励训练的检查点 → 训练集正确且严格
SET 的伪标签 → 0/10/20/30/40% 真随机丢弃的 SFT 冷启动 →
同一轮 RL 同时更新 actor 和逐视觉 token 的 Bernoulli 选择器。
丢弃比例由各图的 token 动作自然产生，不预先限定四个档位。
离散四档代码仅保留作探索性消融。

**起点身份必须如实写明。** 现有本地检查点只按原奖励训练了
100 步，不是作者论文的完整 EMO-R3 权重。它可用于“相同本地起点、
加不加 TokenDrop”的受控比较，不能冒充官方模型，更不能据此声称
超过论文报告的完整模型。

## 上传与运行

直接把本目录内容作为一个 GitHub 仓库上传。`.gitignore` 已屏蔽
`data/` 中的真实样本、`models/`、`checkpoints/`、`outputs/`、
日志、缓存和 `*.pt` 等权重；上传前仍请运行一次
`git status --short` 人工复核，绝不要使用 `git add -f` 强行加入
这些文件。项目自己的源码尚未由用户选择开源许可证；官方奖励函数
与模板的 Apache-2.0 来源见 `THIRD_PARTY.md`。

在 GPU 服务器设置：

```bash
export EMOR3_PROJECT_ROOT=/你的/仓库/绝对路径
export EMOR3_PYTHON=/你的/环境/bin/python
export EMOR3_BASE_MODEL=/你的/本地EMO-R3起点权重目录
mkdir -p "$EMOR3_PROJECT_ROOT/logs" "$EMOR3_PROJECT_ROOT/outputs"
```

安装匹配 CUDA 的 PyTorch 后，再安装 `requirements.txt`。授权获取
EmoSet 数据并按 `data/README.md` 放置；运行
`convert_emoset_full.py`，只在 TRAIN 生成伪标签和开发集。
主方法训练与评估代码分别是
`affectprune_joint_grpo_free.py` 和
`affectprune_joint_eval_free.py`。可配置的提交入口是
`slurm/generate_teacher_train.slurm`、
`slurm/train_free_portable.slurm`、
`slurm/eval_free_portable.slurm`。
详尽环境变量、检查点恢复、逐图同预算比较及方法学边界请看英文
`README.md`。

当前已记录的历史训练中，教师伪标签由更早的评估路径生成；本包
`generate_teacher_train.slurm` 是**无旧选择器依赖的重新生成方案**，
不能保证逐字复现那批伪标签。精确复现实验数值还需原数据划分、
原始伪标签、起点权重和检查点；它们均不适合直接放进 GitHub 源码包。

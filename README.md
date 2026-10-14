# LIBERO-Plus 记忆引导修复 + 七维全参数 SFT

本分支是原有实验的一个 fork，着重实现**LIBERO-Plus** 上运行的 “修复 → SFT” 流程。

仓库存放**代码 + 七类干扰的清单**。模型权重、仿真资源、运行数据，获取方式见[第 4 节](#4-仓库外依赖)。

## 1. 流水线的内容

```
原始 policy 在 7 个扰动维度上跑基线        →  记录失败 case
        ↓
诊断：失败落在哪个技能阶段，以及接管时刻 t*
        ↓
回放失败前缀  →  Memory 引导修正（选段 + 目标位姿 + cuRobo 规划执行）  →  交回原 policy 继续
        ↓
保存完整成功轨迹  →  按同 base task 的 regenerated 成功 demo 的 P95 长度筛选
        ↓
全参数 SFT  →  导出单文件 policy  →  关闭 Memory 重新 rollout 评测
```

## 2. 七个扰动维度（suite = libero_10）

| 类别 | key | case 数 |
|---|---|---:|
| Sensor Noise | `sensor_noise` | 449 |
| Camera Viewpoints | `camera_viewpoints` | 419 |
| Robot Initial States | `robot_initial_states` | 393 |
| Language Instructions | `language_instructions` | 383 |
| Objects Layout | `objects_layout` | 312 |
| Background Textures | `background_textures` | 289 |
| Light Conditions | `light_conditions` | 274 |
| **合计** | | **2519** |

清单在 `experiments/libero_plus/task_lists/libero_10/`（`index.json` + 7 个类别文件）。
`scripts/run_libero_plus_batch.py:81` 会逐个校验 sha256；`index.json` 中 `total_tasks == 2519`
且类别数 `== 7` 是硬编码断言。

## 3. 仓库内容

| 路径 | 内容 |
|---|---|
| `cosmos_policy/` | 原本cosmospolicy的框架，其中对于训练和评估有部分修改（e.g., LoRA） |
| `memory_system/` | Memory 的构建、检索、规划与控制（`tta/`、`execute/`、`offline/`、`pointcloud_action/`） |
| `memory/` | **原本实现libero-pro的memory guided correction，现在和libero-plus没有合并**。libero-pro的分支暂未迁移到当前fork...|
| `scripts/` | 脚本（见第 6 节） |
| `tests/tta/archived_lorafix/train_tta_sft_fixed.py` | 训练入口按字符串路径动态加载的 `mutate_config` |
| `skill_memory_test/libero_10/` | 修复路径读取的 3 个 Memory 产物 |
| `experiments/libero_plus/task_lists/libero_10/` | 七维任务清单（`load_plan()` 的输入） |
| `configs/libero10_experiment_tasks.json` | base task ↔ 合法扰动变体的映射 |
| `docs/repair/` | 各种记录 |

## 4. 仓库外依赖

| 依赖 | 大小 | 来源 |
|---|---:|---|
| **LIBERO-plus 代码** | 119 MB（不含 assets） | `git clone https://github.com/Constructor-Sun/LIBERO-plus.git` 到 `<repo>` **同级**。**不要用上游** `sylvestf/LIBERO-plus`：它缺 `benchmark/__init__.py` 里的 `get_ids_by_category()`，七个维度无法枚举。 |
| **LIBERO-plus assets** | 9.5 GB | 🤗 `Sylvest/LIBERO-plus` → 下载 `assets.zip`，解压到 `LIBERO-plus/libero/libero/` |
| **基础 policy** | 3.8 GB | 🤗 `nvidia/Cosmos-Policy-LIBERO-Predict2-2B`，三个文件都在同一个 repo：`hf download nvidia/Cosmos-Policy-LIBERO-Predict2-2B --local-dir ../checkpoints/Cosmos-Policy-LIBERO-Predict2-2B` |
| **language T5 增量缓存** | 375 MB（374 条） | language 分支所需的 text embedding。可以通过仓库内脚本生成：`python bin/make_libero_plus_t5_embeddings.py --suite libero_10 --all-base-tasks --device cuda --output experiments/cache/libero_plus_language_t5_libero10.pkl`。它取 LIBERO-plus `libero_10` 的 `Language Instructions` 类（383个task，去重后374），用 `CosmosT5TextEncoder` 编码；需要 google/t5-11b，路径由 `--model-path` 额外指定。生成物是 `--extra-t5` 的默认输入。 |
| **成功 demo（P95 参考）** | 3.6 GB | **可选**。P95 阈值已随 `experiments/tta/repair_libero_plus_435/success_task_p95/summary.json`（6 KB）提交。重算 P95（换 base task 或换 demo 集）时需要 🤗 `nvidia/LIBERO-Cosmos-Policy` 的 `success_only/libero_10_regen/`（`--demo-dir` 的默认值）进行重算。|
| **运行数据** | — | 基线 rollout、修复产物、SFT 训练集，迁移时需要重跑 |
| **cuRobo** | — | 未列入 `pyproject.toml` / `uv.lock`，需单独安装到环境 |

### 目录约定

将本仓库 clone 到 `<A>/<B>/<repo>`：

```bash
git clone --branch memory_guided_ttt --single-branch \
    https://github.com/Constructor-Sun/cosmos-policy.git <A>/<B>/<repo>
```

代码全部按**相对位置**找外部目录，所以配套资源的位置由仓库位置唯一确定：

```
<A>/
├── checkpoints/                            同时是 SFT 的输出根目录，必须可写
│   ├── Cosmos-Policy-LIBERO-Predict2-2B/   基础 policy：.pt + libero_dataset_statistics.json + libero_t5_embeddings.pkl
│   ├── Cosmos-Predict2-2B-Video2World/     model-480p-16fps.pt + tokenizer/tokenizer.pth
│   ├── t5-11b/                             只在生成 language T5 缓存时用
│   └── huggingface-hub/                    自动生成，不用手工建
└── <B>/
    ├── <repo>/                             上面 clone 的本仓库（分支 memory_guided_ttt）
    └── LIBERO-plus/                        clone 自 Constructor-Sun/LIBERO-plus，与仓库同级
```

| 路径 | 由谁读取路径 | 调整路径时使用的参数 |
|---|---|---|
| `<A>/checkpoints/` | `Path(repo).parents[1]/checkpoints`（`run_libero_plus_batch.py:119`、`train_libero_plus_sft190.py:22`） | `--policy-dir`、`--ckpt` |
| `<repo>` 与 `LIBERO-plus` 同级 | `run_libero_plus_batch.py:229`、`run_libero_smoke_test.sh:60` | `COSMOS_LIBERO_ROOT` |
| `<A>/checkpoints/t5-11b/` | `get_t5_emb.py:26` | `COSMOS_T5_PATH` |

## 5. 环境

三件事：**cosmospolicy 的 Python 依赖 + LIBERO-plus + cuRobo**。

```bash
# ① 装 cosmos-policy 的依赖（Python >= 3.10）
uv sync --extra cu128 --group libero

# ② cuRobo —— 单独装
git clone https://github.com/NVlabs/curobo && cd curobo && pip install -e .

# ③ LIBERO-plus 不安装，靠 PYTHONPATH 盖过 pip 装的那个 libero 0.1.1
export COSMOS_LIBERO_ROOT=<B>/LIBERO-plus
export PYTHONPATH=$COSMOS_LIBERO_ROOT:$PWD:$PWD/scripts

# ④ libero 的路径配置 —— config.yaml 缺失时 libero 会交互式提问，没人答就 EOFError
export LIBERO_CONFIG_PATH=$PWD/libero_config
mkdir -p $LIBERO_CONFIG_PATH
echo N | python -c "import libero.libero"
```

本人实跑环境是仓库外的一个 conda 环境 `cosmospolicy`（不是 `.venv`）。
`~/.libero/config.yaml` 容易被别的 LIBERO 项目改写，每次运行前或许应显式设置 `LIBERO_CONFIG_PATH`。

## 6. 运行

```bash
# 阶段 0：自检（CPU，不需要 GPU）
python scripts/run_libero_plus_batch.py --validate-only  # 校验 2519 个 case 的 BDDL、初始状态、T5 覆盖

# 阶段 1：基线 rollout（7 维 2519 case）
python scripts/run_libero_plus_batch.py --gpus 0 1 2 3 4 5 6 7

# 阶段 2：收集失败清单 → experiments/tta/repair_libero_plus_435/inputs/failed_435.json
python scripts/collect_libero_plus_failures.py

# 阶段 3：Memory 引导修复
#   默认只跑 4 个 smoke case；全量要显式指定工作目录并排除已完成的：
python scripts/run_libero_plus_repair.py \
    --work-dir          experiments/tta/repair_libero_plus_435/smoke_4 \
    --gpus 0 1 2 3 4 5 6 7
python scripts/run_libero_plus_repair.py --all-cases \
    --work-dir          experiments/tta/repair_libero_plus_435/remaining_431 \
    --exclude-completed-from experiments/tta/repair_libero_plus_435/smoke_4 \
    --gpus 0 1 2 3 4 5 6 7

# 阶段 4：P95 筛选 → success_task_p95/
python scripts/select_libero_plus_success_p95.py

# 阶段 5：SFT 训练 → 导出 → 评测（一条龙，含 smoke、checkpoint 校验、逐 epoch 评测）
python scripts/run_libero_plus_sft190_pipeline.py \
    --job-name <name> --full-finetune --epochs 8 --gpus 0 1 2 3 4 5 6 7
```

`run_libero_plus_repair.py` 也支持 `--task-ids` 选单个 case，以及 `--resume` / `--retry-errors`。

## 7. 协议常量

| 项 | 值 |
|---|---|
| `action_seed` | 42 |
| `environment_seed` | 10000 |
| `settle_steps` | 30（不计入保存的轨迹长度） |
| `max_steps` | 700（含前缀回放与 Memory 干预，但是经过长度筛选后的训练样本均会回到原本libero设定的520 steps以内） |
| `num_trials_per_task` | 1 |
| `init_state_index` | 0 |
| `rng_mode` | `openwam` |

轨迹长度定义为完整的 `actions`，不含前 30 步全 0的settling。

## 8. 不在仓库里的东西

- **全部运行产物**：`experiments/libero_plus/cosmospolicy_full_7dims/`（2519 条基线结果）、
  `experiments/tta/repair_libero_plus_435/`（435 个修复 case）、`experiments/libero_plus/sft190_p95/`（逐 case 评测）
- **SFT 训练集**：`training/`。注意其中的 HDF5 用 `h5py.ExternalLink` **以绝对路径**指向原始修复产物，
  只拷贝 `training/<data_dir>` 不会得到独立可用的数据。
- **导出权重**：`training/<job>_iter<N>_full.pt`
- **未包含的模块**：`scripts_memory/`、`memory/` 重构树的其余部分、`scripts/libero_pro/` 的其余部分
  （只保留了 `precompute_t5.py`）。
- **包含但本流水线走不到的模块**：`memory_system/` 是**整个保留**的（116 个文件），
  其中 `pointcloud_action/`（42 个）、`offline/`（11 个）、`execute/planner/`（14 个）属于另外的实验分支。
  它们唯一的静态入口是 `scripts/libero_pro/oracle_ready_eval.py`，而该模块只在
  `run_libero_eval.py:390` 的一个环境变量门（`COSMOS_SKILL_READY_MEMORY`）之后被导入，
  本流水线不设这个变量，因此这些代码不会执行。它们保留是为了不破坏 `memory_system` 包内的相互引用。

## 9. 已有的实验结果

以下数字由本分支的代码在原作者机器上产生，**产物未包含在本仓库**。

评测集是该维度**全部基线失败 case**（基线在这些 case 上恒为 0），所以这些数字只说明"修复了多少"，
**不反映在基线已通过的 2084 个 case 上是否发生遗忘** —— 那些 case 从未在微调后的模型上跑过。

五个已训维度统一取 **epoch 2**：

| 维度 | 训练轨迹 | 训练集（模型见过） | heldout（没见过） | 合计 |
|---|---:|---|---|---|
| Robot Initial | 90 | 68.9% (62/90) | 62.1% (18/29) | 67.2% (80/119) |
| Camera | 24 | 79.2% (19/24) | 32.9% (47/143) | 39.5% (66/167) |
| Background | 29 | 69.0% (20/29) | 22.0% (9/41) | 41.4% (29/70) |
| Sensor Noise | 22 | 63.6% (14/22) | 20.0% (1/5) | 55.6% (15/27) |
| Light | 14 | 85.7% (12/14) | 0.0% (0/2) | 75.0% (12/16) |
| **合计** | **179** | **70.9% (127/179)** | **34.1% (75/220)** | **50.6% (202/399)** |

"训练集" = 喂给 SFT 的修复轨迹；"heldout" = 同一维度的其余基线失败 case。

各 epoch 的合计（取自 5 个 job 的 `evaluation_curve.csv`）：

| epoch | 训练集 | heldout | 合计 |
|---|---:|---:|---:|
| 1 | 74.9% (134/179) | 28.6% (63/220) | 49.4% (197/399) |
| **2** | **70.9% (127/179)** | **34.1% (75/220)** | 50.6% (202/399) |
| 4 | 76.5% (137/179) | 32.7% (72/220) | 52.4% (209/399) |
| 8 | 74.9% (134/179) | 35.0% (77/220) | 52.9% (211/399) |

七个维度里只有五类跑过 SFT。**Language 与 Objects 从未训练**（它们通过 P95 筛选的轨迹分别只有 8 条和 3 条）。

读表注意：

- **Light（heldout n=2）和 Sensor Noise（n=5）的 heldout 列没有统计意义**；
  能单独解读的是 Camera / Robot Initial / Background 三行和合计行。
- 训练集成功率**不是 100%** —— 模型重新 rollout 自己见过的 case 也会失败一部分。
- 训练集成功率低不代表更差：epoch 2 在训练集上最低但 heldout 最高之一，说明它过拟合最轻。
  选 epoch 看 heldout 和合计，不看训练集。

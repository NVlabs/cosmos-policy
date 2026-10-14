# LIBERO-Plus 记忆引导修复 + 七维全参数 SFT

本分支是 [NVIDIA Cosmos Policy](https://github.com/NVIDIA/cosmos-policy) 的一个 fork，在其上实现了一套在
**LIBERO-Plus** 上运行的「记忆引导修复 → 全参数 SFT」流水线。

仓库存放**代码 + 少量固定的 Memory 产物 + 七维任务清单**。模型权重、仿真资源、运行数据都不在这里，
获取方式见[第 4 节](#4-仓库外依赖)。

---

## 1. 流水线做什么

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

Memory 只提供**选哪一段**和**目标位姿**；接管动作由规划器在线生成，不含 memory 中存储的动作块。
因此 SFT 学到的是「修正之后如何继续」，而不是「复现控制器输出的动作」。

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
| `cosmos_policy/` | policy 模型、imaginaire 框架、数据集、训练器、eval 入口 |
| `memory_system/` | Memory 的构建、检索、规划与控制（`tta/`、`execute/`、`offline/`、`pointcloud_action/`） |
| `memory/` | 仅 3 个文件：`libero_joint_control.py` 用到的 `densify_joint_path` 及其 `types.py` |
| `scripts/` | 流水线入口（见第 6 节） |
| `tests/tta/archived_lorafix/train_tta_sft_fixed.py` | 训练入口按字符串路径动态加载的 `mutate_config` |
| `skill_memory_test/libero_10/` | 修复路径读取的 3 个 Memory 产物 |
| `experiments/libero_plus/task_lists/libero_10/` | 七维任务清单（`load_plan()` 的输入） |
| `configs/libero10_experiment_tasks.json` | base task ↔ 合法扰动变体的映射 |
| `docs/repair/` | 流水线的协议、失败清单与核对记录 |

三条 `.gitignore` 规则会命中本仓库需要的文件，它们是用 `git add -f` 加进来的，**不要重新生成索引**：

- `.gitignore:10` 的 `experiments/`（无锚点，误伤 `cosmos_policy/experiments/`）→ `libero_joint_control.py`
- `.gitignore:13` 的 `skill_memory_test/` → 那 3 个 Memory 产物
- `.gitignore` 末行 `tests/tta/archived_lorafix/` → `train_tta_sft_fixed.py`
- `experiments/libero_plus/task_lists/` 同样被 `experiments/` 命中

## 4. 仓库外依赖

| 依赖 | 大小 | 来源 |
|---|---:|---|
| **LIBERO-plus 代码** | 119 MB（不含 assets） | 上游 `https://github.com/sylvestf/LIBERO-plus.git` |
| **LIBERO-plus assets** | 9.5 GB | 🤗 `Sylvest/LIBERO-plus` → 下载 `assets.zip`，解压到 `LIBERO-plus/libero/libero/` |
| **基础 policy checkpoint** | 3.7 GB | 🤗 `nvidia/Cosmos-Policy-LIBERO-Predict2-2B`（见 `LIBERO.md`） |
| **`libero_t5_embeddings.pkl`** + `libero_dataset_statistics.json` | 109 MB | 🤗 `nvidia/LIBERO-Cosmos-Policy`：`hf download nvidia/LIBERO-Cosmos-Policy --repo-type dataset --local-dir LIBERO-Cosmos-Policy` |
| **language T5 增量缓存** | 375 MB（374 条） | **无下载源**。这是 `--extra-t5` 的默认输入，运行时与 checkpoint 目录下的基础缓存**合并**后才完整（`run_libero_plus_batch.py:187-190`）。仓库里没有生成它的脚本；`scripts/libero_pro/precompute_t5.py` 是通用工具，可按 suite 枚举指令并补算缺失项（需加载 9 GB T5 编码器，未实测） |
| **成功 demo（P95 参考）** | 3.6 GB | 🤗 `nvidia/LIBERO-Cosmos-Policy` 下的 `success_only/libero_10_regen/`（每个 base task 一个 `*_demo.hdf5`），即 `select_libero_plus_success_p95.py` 的 `--demo-dir` 默认值。同目录的 `dataset_statistics*.json` 也在 SFT 阶段被链接。整个数据集约 80 GB，但只需要这一部分 |
| **运行数据** | — | 基线 rollout、修复产物、SFT 训练集，均需自己重跑生成 |
| **cuRobo** | — | 未列入 `pyproject.toml` / `uv.lock`，需单独安装 |

### 版本 pin

`run_libero_plus_batch.py:188` 会校验 LIBERO-plus 的 `benchmark/task_classification.json`：

```
期望 sha256 = faa87cce3e3ba434da01df7c77523a391b5f2912e4774330b0aa1be5f6a999e6
```

对不上会直接报 `LIBERO-plus classification differs from the pinned task manifests`。
本流水线使用的 LIBERO-plus 相对上游还有三处本地修改，需从配套分支获取：

```
 M libero/libero/benchmark/__init__.py      (+90 行)
 M libero/libero/envs/env_wrapper.py        (+30 行)
 ?? libero/libero/bddl_files/libero_10/counterfactual_plusonly_*.bddl   (10 个)
```

### 目录约定

代码按**相对位置**解析外部资源，仓库必须放在 `<A>/<B>/<repo>`，其中：

```
<A>/checkpoints/Cosmos-Policy-LIBERO-Predict2-2B/     ← run_libero_plus_batch.py 用 root.parents[1]
<B>/LIBERO-plus/                                       ← 或设置 COSMOS_LIBERO_ROOT 指向别处
```

`scripts/run_libero_smoke_test.sh` 另外用 `$REPO_ROOT/../../../checkpoints` 定位 checkpoint，
`${COSMOS_LIBERO_ROOT:-../../LIBERO-plus}` 定位 LIBERO 包，两者都可用环境变量覆盖。

## 5. 环境准备

```bash
# 1) LIBERO-plus 及其 assets 就位后，生成 libero 配置。
#    注意：config.yaml 缺失时 libero 会交互式提问，无人应答会 EOFError。
export LIBERO_CONFIG_PATH=$PWD/libero_config
mkdir -p $LIBERO_CONFIG_PATH
echo N | python -c "import libero.libero"     # 生成指向本机 LIBERO-plus 的绝对路径配置

# 2) 环境变量
export COSMOS_LIBERO_ROOT=$PWD/../LIBERO-plus
export PYTHONPATH=$COSMOS_LIBERO_ROOT:$PWD:$PWD/scripts
```

`~/.libero/config.yaml` 可能被其他 LIBERO 分支的实验改写，**每次运行前显式设置 `LIBERO_CONFIG_PATH`**
比依赖全局配置可靠。

## 6. 运行

```bash
# 阶段 0：自检（CPU，不需要 GPU）
python scripts/run_libero_plus_batch.py --dry-run        # 校验任务清单并打印计划
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

### 已知限制

`run_libero_plus_sft190_pipeline.py:171` 的 `--evaluation-scope` 目前只覆盖
`robotinit-failures`、`robotinit-heldout-failures`、`camera-failures`、`camera-memory-failures`、
`background-heldout-failures`、`sensor-noise-failures`。**Background 全量、Light、Language、Objects
四类当时是用 `scripts/experiments/libero_plus_sft190/_plans/<job>/` 下的局部 pipeline 跑的**，
那些文件没有带进本分支。要在这些维度上评测，需要先把对应的 scope 补进公共 pipeline。

## 7. 协议常量

| 项 | 值 |
|---|---|
| `action_seed` | 42 |
| `environment_seed` | 10000 |
| `settle_steps` | 30（不计入保存的轨迹长度） |
| `max_steps` | 700（含前缀回放与 Memory 干预） |
| `num_trials_per_task` | 1 |
| `init_state_index` | 0 |
| `rng_mode` | `openwam` |
| P95 规则 | 成功 **且** `full action_steps <= 同 base task 的 regenerated 成功 demo 的 P95`（numpy linear） |

轨迹长度定义为完整的 `actions`（基线前缀 + Memory 干预 + policy 续跑），不含 30 步 settling。

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

以下数字由本分支的代码在原作者机器上产生，**产物未包含在本仓库**。评测集是该维度全部基线失败 case
（即基线成功率在这些 case 上恒为 0），因此不反映在基线已通过的 case 上是否发生遗忘。

| 维度 | 训练轨迹 | 评测 case | 训练集成功率(e8) | heldout 成功率(e1 → e8) | heldout n |
|---|---:|---:|---:|---|---:|
| Robot Initial | 90 | 119 | 76.7% | 58.6% → 55.2% | 29 |
| Camera | 24 | 167 | 79.2% | 24.5% → 32.9% | 143 |
| Background | 29 | 70 | 75.9% | 19.5% → 24.4% | 41 |
| Sensor Noise | 22 | 27 | 59.1% | 40% → 60% | 5 |
| Light | 14 | 16 | 78.6% | 50% → 50% | 2 |
| Language | — | — | 未训练 | — | — |
| Objects | — | — | 未训练 | — | — |

七个维度中只有五类跑过 SFT；Language 与 Objects 从未训练（它们通过 P95 筛选的轨迹分别只有 8 条和 3 条）。

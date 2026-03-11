# MolBlender 统一 API 使用指南

## 概述

MolBlender 提供统一的 API 层 (`molblender.api`)，整合了表征生成、模型筛选、Dashboard 等所有核心功能。

## API 层级

MolBlender 提供多层级 API，用户可以根据需求选择合适的入口：

### 推荐入口：molblender.api（统一 Facade）

**适用场景**：新代码、快速上手、常用功能

`molblender.api` 是统一的便利入口（Facade 模式），提供最常用功能的简洁接口：

```python
from molblender.api import (
    # 表征
    get_featurizer,
    list_featurizers,
    # 模型筛选
    screen_models,
    load_results,
    # 可视化
    run_dashboard,
)
```

**优点**：
- 简洁易用
- 统一入口
- 向后兼容保证

### 领域 API：更丰富的功能

对于需要更多控制或高级功能的场景，可以直接使用领域 API：

#### molblender.models（ML 筛选领域 API）

**适用场景**：需要完整的 ML 筛选功能

```python
from molblender.models import (
    # 基础筛选（与 molblender.api 相同）
    screen_models,
    quick_screen,
    thorough_screen,
    # 分析功能（richer API）
    analyze_results,
    compare_models,
    compare_representations,
    # 可视化
    plot_screening_results,
    create_performance_dashboard,
)
```

**额外功能**：
- 多种筛选策略（quick/thorough/interpretable）
- 结果分析和对比
- 统计检验
- 性能可视化

#### molblender.representations（表征领域 API）

**适用场景**：需要详细的表征器信息

```python
from molblender.representations import (
    # 基础功能（与 molblender.api 相同）
    get_featurizer,
    list_available_featurizers,
    # 详细信息（richer API）
    get_featurizer_info,
    print_available_featurizers,
    # 类别选择
    get_category_info,
    select_featurizers_by_category,
    # 子模块访问
    fingerprints,
    descriptors,
    graph,
)
```

**额外功能**：
- 表征器详细元数据
- 类别浏览和筛选
- 直接访问子模块

#### molblender.drawings（静态绘图工具）

**适用场景**：生成出版物质量的静态图表

```python
from molblender.drawings import (
    # 核心配置
    PlotConfig,
    set_plot_style,
    # 绘图函数
    plot_histogram,
    plot_scatter_fit,
    plot_heatmap,
    # 主题
    set_scientific_publication_style,
    set_presentation_style,
)
```

**定位**：静态绘图工具（matplotlib/seaborn），不等同于 interactive dashboard

#### molblender.dashboard（交互式探索）

**适用场景**：交互式数据探索和结果分析

```python
from molblender.dashboard import run_dashboard

# 启动交互式 Dashboard
run_dashboard()
```

**定位**：基于 Streamlit 的交互式 Web UI

### 顶层 molblender（最常用功能）

顶层 `import molblender` 暴露最常用的函数：

```python
import molblender

# 推荐使用（unified facade）
molblender.screen_models(...)
molblender.get_featurizer(...)
molblender.run_dashboard(...)

# 兼容入口（直接从子模块导入）
molblender.list_available_featurizers(...)
molblender.analyze_results(...)  # 需要更丰富的 API
```

### 选择指南

| 需求 | 推荐入口 | 备选方案 |
|------|----------|----------|
| **新代码/快速上手** | `molblender.api` | 顶层 `molblender` |
| **完整 ML 筛选** | `molblender.models` | `molblender.api.screen_models` |
| **表征器详细信息** | `molblender.representations` | `molblender.api.get_featurizer_info` |
| **静态图表** | `molblender.drawings` | - |
| **交互式探索** | `molblender.dashboard` | - |

## 快速开始

### 安装

```bash
pip install molblender
```

### 基本使用

```python
from molblender.api import (
    get_featurizer,
    list_featurizers,
    screen_models,
    create_screener,
    load_results,
    run_dashboard,
    load_dashboard_data,
)
```

## API 模块

### 1. 表征 API

#### list_featurizers()

列出所有可用的分子表征生成器。

```python
from molblender.api import list_featurizers

# 列出所有表征器
all_featurizers = list_featurizers()
print(f"可用表征器数量: {len(all_featurizers)}")

# 列出特定类别的表征器
fingerprint_featurizers = list_featurizers(category="fingerprints")
protein_featurizers = list_featurizers(category="protein")
```

**返回**: `list[str]` - 表征器名称列表

#### get_featurizer()

获取分子表征生成器实例。

```python
from molblender.api import get_featurizer

# 获取指纹表征器
morgan_featurizer = get_featurizer("morgan_fp")
rdkit_featurizer = get_featurizer("rdkit_fp")

# 获取蛋白质表征器
prot_featurizer = get_featurizer("prot_bert")

# 生成表征
smiles = ["CCO", "c1ccccc1", "CC(C)CC"]
features = morgan_featurizer.featurize(smiles)
print(f"特征形状: {features.shape}")
```

**参数**:
- `name` (`str`): 表征器名称

**返回**: `BaseFeaturizer` - 表征器实例

#### get_featurizer_info()

获取表征器的详细信息。

```python
from molblender.api import get_featurizer_info

# 获取表征器信息
info = get_featurizer_info("morgan_fp")
print(f"名称: {info['name']}")
print(f"类别: {info['category']}")
print(f"输出形状: {info['output_shape']}")
print(f"描述: {info['description']}")
```

**返回**: `dict` - 包含表征器详细信息的字典

### 2. 模型 API

#### screen_models()

执行 ML 模型筛选（推荐使用）。

```python
from molblender.api import screen_models

# 简单筛选
results = screen_models(
    smiles_data=["CCO", "c1ccccc1", "CC(C)CC", ...],
    representations=["morgan_fp", "rdkit_fp"],
    task_type="regression",
    target_values=[0.5, 1.2, 0.8, ...],
)

# 高级筛选
results = screen_models(
    smiles_data=smiles_list,
    representations=["morgan_fp", "chemberta"],
    models=["random_forest", "xgboost"],
    task_type="regression",
    target_values=activity_values,
    split_strategy="random",
    test_size=0.2,
    max_cpu_cores=-1,
    max_workers_per_model=1,
    enable_hpo=False,
    combinations="auto",
)

# 保存结果到数据库
results = screen_models(
    ...,
    save_path="screening_results.db",
    session_name="my_screening_session",
)
```

**参数**:
- `smiles_data` (`list[str]`): SMILES 分子列表
- `representations` (`list[str]`): 分子表征列表
- `models` (`list[str]` | `None`): 模型列表（None表示自动选择）
- `task_type` (`str`): 任务类型（"classification" 或 "regression"）
- `target_values` (`list[float]`): 目标值
- `save_path` (`str | None`): 结果保存路径
- `session_name` (`str | None`): 会话名称
- 其他参数参见 `ScreeningConfig`

并行参数约定:
- `max_cpu_cores`: 整个筛选流程可用的总 CPU 预算
- `max_workers_per_model`: 单个模型内部可使用的 worker 上限
- `n_jobs`: 旧接口兼容别名，新代码不再推荐

**返回**: `dict` - 筛选结果字典

#### create_screener()

创建筛选器实例（高级用法）。

```python
from molblender.api import create_screener
from molblender.models.api.core import ScreeningConfig

# 创建配置
config = ScreeningConfig(
    task_type="regression",
    representations=["morgan_fp"],
    models=["random_forest"],
    max_cpu_cores=-1,
    max_workers_per_model=1,
)

# 创建筛选器
screener = create_screener(config)

# 准备数据
screener.prepare_data(
    smiles_data=["CCO", "c1ccccc1"],
    target_values=[0.5, 1.2],
)

# 运行筛选
results = screener.run_screening()
```

**参数**:
- `config` (`ScreeningConfig`): 筛选配置

**返回**: `BaseScreener` - 筛选器实例

#### load_results()

从数据库加载筛选结果。

```python
from molblender.api import load_results

# 加载结果
results = load_results(
    db_path="screening_results.db",
    session_name="my_screening_session",
)

# 获取最佳结果
best_result = results.get_best_result(metric="r2_score")
print(f"最佳模型: {best_result['model_name']}")
print(f"最佳表征: {best_result['representation_name']}")
print(f"最佳分数: {best_result['primary_metric']}")
```

**参数**:
- `db_path` (`str`): 数据库路径
- `session_name` (`str | None`): 会话名称（None表示加载所有会话）

**返回**: `ResultsDatabase` - 结果数据库对象

### 3. Dashboard API

#### run_dashboard()

启动交互式 Dashboard。

```python
from molblender.api import run_dashboard

# 启动 Dashboard（默认端口 8501）
run_dashboard()

# 指定数据库和端口
run_dashboard(
    db_path="screening_results.db",
    port=8502,
)
```

**参数**:
- `db_path` (`str | None`): 数据库路径
- `port` (`int`): Web 服务端口

#### load_dashboard_data()

加载 Dashboard 数据（用于自定义分析）。

```python
from molblender.api import load_dashboard_data

# 加载数据
data = load_dashboard_data("screening_results.db")

# 访问数据
sessions = data['sessions']
results = data['results']
models = data['models']
representations = data['representations']
```

**返回**: `dict` - 包含所有 Dashboard 数据的字典

## 完整示例

### 示例1: 简单回归任务

```python
from molblender.api import screen_models, load_results, run_dashboard

# 1. 准备数据
smiles = [
    "CCO",  # 乙醇
    "c1ccccc1",  # 苯
    "CC(C)CC",  # 异丁烷
    # ... 更多分子
]
activities = [0.5, 1.2, 0.8, ...]  # 生物活性值

# 2. 运行筛选
results = screen_models(
    smiles_data=smiles,
    representations=["morgan_fp", "rdkit_fp"],
    task_type="regression",
    target_values=activities,
    save_path="results.db",
    session_name="regression_screening",
)

# 3. 查看结果
print(f"完成 {len(results)} 个模型-表征组合筛选")

# 4. 启动 Dashboard 可视化
run_dashboard(db_path="results.db")
```

### 示例2: 分类任务

```python
from molblender.api import screen_models

# 分类数据
smiles = [...]
labels = [0, 1, 0, 1, ...]  # 0: 不活性, 1: 活性

# 运行分类筛选
results = screen_models(
    smiles_data=smiles,
    representations=["morgan_fp", "chemberta"],
    models=["random_forest", "xgboost", "logistic_regression"],
    task_type="classification",
    target_values=labels,
    save_path="classification_results.db",
    session_name="classification_screening",
    metric_type="roc_auc",  # 使用 AUC 作为评估指标
)
```

### 示例3: 高级筛选配置

```python
from molblender.api import create_screener
from molblender.models.api.core import ScreeningConfig

# 创建高级配置
config = ScreeningConfig(
    task_type="regression",
    representations=["morgan_fp", "chemberta", "unimol_cls"],
    models=None,  # 自动选择兼容模型
    cv_folds=5,
    test_size=0.2,
    split_strategy="scaffold_split",  # 使用骨架分割
    n_jobs=-1,
    enable_hpo=False,
    combinations="auto",  # 使用主要路径 + 备用路径
    auto_resource_optimization=True,  # 自动资源优化
)

# 创建并运行筛选器
screener = create_screener(config)
screener.prepare_data(
    smiles_data=smiles,
    target_values=activities,
)
results = screener.run_screening(
    save_path="advanced_results.db",
    session_name="advanced_screening",
)
```

### 示例4: 结果分析

```python
from molblender.api import load_results

# 加载结果
db = load_results("results.db", session_name="my_screening")

# 获取所有结果
all_results = db.get_all_results()
print(f"总结果数: {len(all_results)}")

# 获取最佳结果
best = db.get_best_result()
print(f"最佳组合: {best['model_name']} + {best['representation_name']}")
print(f"最佳分数: {best['primary_metric']:.3f}")

# 筛选特定结果
rf_results = db.get_results(
    model_name="random_forest",
    representation_name="morgan_fp",
)

# 获取统计信息
stats = db.get_statistics()
print(f"平均R²: {stats['mean_primary_metric']:.3f}")
print(f"标准差: {stats['std_primary_metric']:.3f}")
```

## API 设计原则

### 1. 简洁性

- 最少的参数，合理的默认值
- 一个函数完成常见任务
- 自动检测和验证

### 2. 一致性

- 统一的命名约定
- 一致的参数顺序
- 统一的返回类型

### 3. 向后兼容性

- 旧导入路径仍然可用
- 逐步迁移，无需立即重写
- 平滑的升级路径

### 4. 可扩展性

- 支持高级配置
- 支持自定义扩展
- 支持插件机制

## 错误处理

### 常见错误

```python
# 1. 不兼容的模型-表征组合
try:
    results = screen_models(
        smiles_data=smiles,
        representations=["canonical_smiles"],  # STRING 类型
        models=["random_forest"],  # 不能处理 STRING
        task_type="regression",
        target_values=activities,
    )
except ValueError as e:
    print(f"兼容性错误: {e}")

# 2. 无效的表征器名称
try:
    featurizer = get_featurizer("invalid_featurizer")
except ValueError as e:
    print(f"表征器错误: {e}")

# 3. 数据格式错误
try:
    results = screen_models(
        smiles_data=["INVALID_SMILES"],
        representations=["morgan_fp"],
        task_type="regression",
        target_values=[0.5],
    )
except Exception as e:
    print(f"数据错误: {e}")
```

### 错误处理最佳实践

```python
from molblender.api import screen_models
import logging

logger = logging.getLogger(__name__)

def safe_screening(smiles, targets, reprs, models):
    """带错误处理的筛选"""
    try:
        results = screen_models(
            smiles_data=smiles,
            representations=reprs,
            models=models,
            task_type="regression",
            target_values=targets,
        )
        return results
    except ValueError as e:
        logger.error(f"配置错误: {e}")
        return None
    except Exception as e:
        logger.error(f"筛选失败: {e}")
        return None
```

## 性能优化

### 1. 并行处理

```python
# 自动并行（推荐）
results = screen_models(
    smiles_data=large_smiles_list,
    representations=["morgan_fp"],
    models=["random_forest"],
    n_jobs=-1,  # 使用所有CPU核心
)
```

### 2. 缓存优化

```python
import os
from molblender.config import config_manager

# 设置缓存目录
config_manager.set_cache_dir("representations", "./cache")

# 启用缓存
os.environ["MOLBLENDER_CACHE_ENABLED"] = "true"

# 筛选会自动使用缓存
results = screen_models(
    smiles_data=smiles,
    representations=["morgan_fp"],
    task_type="regression",
    target_values=activities,
)
```

### 3. 资源优化

```python
from molblender.models.api.core import ScreeningConfig

# 自动资源优化
config = ScreeningConfig(
    task_type="regression",
    representations=["morgan_fp"],
    auto_resource_optimization=True,  # 启用自动优化
)

# 创建筛选器
screener = create_screener(config)
```

## 相关文档

- [迁移指南](migration_guide.md) - 从旧 API 迁移
- [ConfigManager 指南](config_manager_guide.md) - 配置管理
- [快速开始](quickstart.md) - 新手入门
- [架构诊断报告](../../Archive/ARCHITECTURE_DIAGNOSIS.md) - 架构详情

---

**最后更新**: 2026-03-10
**版本**: 1.0.0

## Advanced: Tool Registry API

MolBlender 提供了统一的表征器注册表系统，用于高级元数据查询和筛选。

### ToolRegistry

`ToolRegistry` 提供了统一的表征器发现和查询接口。

```python
from molblender.representations.tool_registry import ToolRegistry, ToolInfo

# 获取注册表实例
registry = ToolRegistry()

# 按类别筛选
molecular_featurizers = registry.list(category="molecular")
protein_featurizers = registry.list(category="protein")

# 按标签筛选
gpu_featurizers = registry.list(tags=["gpu"])
experimental_featurizers = registry.list(tags=["experimental"])

# 获取详细元数据
info: ToolInfo = registry.get("ecfp")
print(f"Description: {info.description}")
print(f"Dependencies: {info.dependencies}")
print(f"Output shape: {info.output_shape}")

# 搜索表征器
results = registry.search("fingerprint")
for info in results:
    print(f"{info.name}: {info.description}")
```

### ToolInfo

`ToolInfo` 包含表征器的完整元数据：

```python
@dataclass
class ToolInfo:
    name: str                      # 表征器名称
    category: str                  # 类别 (molecular, protein, etc.)
    description: str               # 描述
    source: str                    # 来源 (rdkit, deepchem, etc.)
    tags: list[str]                # 标签 (gpu, experimental, etc.)
    input_type: str                # 输入类型 (smiles, sdf, etc.)
    output_type: str               # 输出类型 (vector, matrix, etc.)
    output_shape: tuple            # 输出形状
    default_kwargs: dict           # 默认参数
    is_available: bool             # 是否可用（依赖检查）
    dependencies: list[str]        # 依赖列表
```

### 使用示例

```python
from molblender.representations.tool_registry import get_tool_registry

# 获取全局注册表实例
registry = get_tool_registry()

# 列出所有可用的 GPU 表征器
gpu_tools = registry.list(tags=["gpu"])
for tool in gpu_tools:
    if tool.is_available:
        print(f"✅ {tool.name}: {tool.description}")
    else:
        print(f"❌ {tool.name}: Missing dependencies {tool.dependencies}")

# 查找特定表征器
info = registry.get("morgan_fp")
if info and info.is_available:
    print(f"Morgan fingerprint is available!")
    print(f"Default radius: {info.default_kwargs.get('radius', 'N/A')}")
    print(f"Default n_bits: {info.default_kwargs.get('n_bits', 'N/A')}")
```


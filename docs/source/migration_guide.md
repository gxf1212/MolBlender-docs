# MolBlender API 迁移指南

## 概述

本文档帮助您从旧的 MolBlender API 迁移到新的统一 API 层。新 API 提供更一致的接口和更好的向后兼容性。

## API 变更总览

### 旧 API（分散入口）

```python
# 表征相关
from molblender.representations import get_featurizer

# 模型相关
from molblender.models import screen_models

# Dashboard相关
from molblender.dashboard import run_dashboard
```

### 新 API（统一入口）

```python
# 所有功能从统一入口导入
from molblender.api import (
    get_featurizer,
    screen_models,
    run_dashboard,
)
```

## 详细迁移指南

### 1. 表征 API 迁移

#### 旧方式

```python
from molblender.representations import get_featurizer
from molblender.representations import list_available_featurizers
from molblender.representations import get_featurizer_info
```

#### 新方式（推荐）

```python
from molblender.api import (
    get_featurizer,
    list_featurizers,  # 注意：函数名略有变化
    get_featurizer_info,
)
```

**变更说明**：
- `list_available_featurizers` → `list_featurizers`（简化命名）
- 其他函数保持不变
- 旧导入路径仍然可用（向后兼容）

### 2. 模型 API 迁移

#### 旧方式

```python
from molblender.models.api.screening import screen_models
from molblender.models.api.screening import create_screener
from molblender.models.api.utils import load_results
```

#### 新方式（推荐）

```python
from molblender.api import (
    screen_models,
    create_screener,
    load_results,
)
```

**变更说明**：
- 所有模型相关 API 从统一入口导入
- 函数名保持不变
- 功能完全向后兼容

### 3. Dashboard API 迁移

#### 旧方式

```python
from molblender.dashboard.app import run_dashboard
from molblender.dashboard.data.loaders import load_dashboard_data
```

#### 新方式（推荐）

```python
from molblender.api import (
    run_dashboard,
    load_dashboard_data,
)
```

**变更说明**：
- Dashboard API 从统一入口导入
- 函数名保持不变
- 功能完全向后兼容

### 4. 配置管理迁移

#### 旧方式

```python
from molblender.config import get_cache_dir, set_cache_dir
from molblender.config import EFFECTIVE_HF_HOME, EFFECTIVE_TORCH_HOME
```

#### 新方式（推荐）

```python
from molblender.config import config_manager

# 获取缓存目录
cache_dir = config_manager.get_cache_dir("torch")

# 设置缓存目录
config_manager.set_cache_dir("hf", "/custom/path")

# 获取配置对象
cache_config = config_manager.get_cache_config()
print(cache_config.torch_home)
print(cache_config.hf_home)
```

**变更说明**：
- 新增 `ConfigManager` 单例模式
- 提供更结构化的配置管理
- 旧方式仍然可用（标记为 `_legacy`）

### 5. Infrastructure 层迁移

#### 旧方式（已删除 / removed）

```python
from molblender.models.api.core.evaluation.utilities import timeout_context
from molblender.models.api.utils.resource_scheduler import IntelligentResourceScheduler
```

#### 新方式（推荐）

```python
from molblender.models.api.infrastructure import (
    ExecutionContext,
    ResourcePolicy,
)

# 使用 ExecutionContext
ctx = ExecutionContext.from_screening_config(config)
with ctx.timeout_context(timeout=300):
    # 你的代码
    pass

# 使用 ResourcePolicy
policy = ResourcePolicy.from_screening_config(config)
parallel_config = policy.determine_parallel_config(
    n_models=10,
    has_heavy_models=True,
)
```

**变更说明**：
- `timeout_context` 已迁移到 `ExecutionContext`
- `resource_scheduler` 已删除（功能整合到 `ResourcePolicy`）
- 新的 Infrastructure 层提供更统一的资源管理

## 迁移检查清单

### 阶段1：准备（无需代码修改）

- [ ] 阅读本文档，了解 API 变更
- [ ] 运行现有测试套件，确保所有测试通过
- [ ] 备份当前代码和数据库

### 阶段2：渐进式迁移

#### Step 1: 安装最新版本

```bash
cd /data/gxf1212/work/MolBlender
pip install -e .
```

#### Step 2: 更新导入语句（可选）

```python
# 逐步替换旧导入
# 从 molblender.representations import get_featurizer  # 旧
from molblender.api import get_featurizer  # 新
```

#### Step 3: 测试验证

```bash
# 运行测试套件
pytest tests/ -v

# 运行特定模块测试
pytest tests/representations/ -v
pytest tests/models/ -v
pytest tests/dashboard/ -v
```

### 阶段3：完全迁移（推荐但非必需）

- [ ] 将所有导入更新为新的统一 API
- [ ] 删除对旧 API 的直接依赖
- [ ] 更新项目文档和示例代码

## 向后兼容性保证

### 仍然支持的旧 API

```python
# ✅ 仍然可用
from molblender.representations import get_featurizer
from molblender.models.api.screening import screen_models
from molblender.dashboard.app import run_dashboard

# ✅ 配置管理旧方式仍然可用
from molblender.config import get_cache_dir as get_cache_dir_legacy
```

### 已删除的 API

```python
# ❌ 已删除（2026-03-10）
from molblender.models.api.utils.resource_scheduler import IntelligentResourceScheduler
from molblender.models.api.core.evaluation.utilities import timeout_context

# ✅ 替代方案
from molblender.models.api.infrastructure import (
    ExecutionContext,  # 替代 timeout_context
    ResourcePolicy,    # 替代 IntelligentResourceScheduler
)
```

## 常见迁移问题

### Q1: 我必须立即迁移到新 API 吗？

**A**: 取决于你使用的是哪一层 API。

- 顶层 workflow API（如 `molblender.api`、`molblender.models`）仍然可用，通常可以渐进迁移
- 但本文列出的低层内部入口里，有些已经删除，例如 `timeout_context` 和 `resource_scheduler`
- 如果你的代码直接依赖这些已删除入口，就需要立即切到 `ExecutionContext` / `ResourcePolicy`

### Q2: 迁移后性能会变化吗？

**A**: 对大多数用户工作流来说，行为目标不变；但运行时管理已经统一到了 Infrastructure 层，因此导入路径和内部执行组织方式确实发生了变化。收益主要是：

1. 更一致的资源策略
2. 更清晰的错误和 telemetry 语义
3. 更轻量的顶层导入

### Q3: 如何处理已删除的 API？

**A**:
1. `timeout_context` → 使用 `ExecutionContext.timeout_context()`
2. `resource_scheduler` → 使用 `ResourcePolicy`
3. 参见上面的"Infrastructure 层迁移"部分

### Q4: 测试失败了怎么办？

**A**:
1. 检查是否使用了已删除的 API
2. 查看错误消息，找到需要更新的导入
3. 参考本文档的迁移示例

## 示例：完整迁移脚本

### 旧代码

```python
# old_script.py
from molblender.representations import get_featurizer, list_available_featurizers
from molblender.models.api.screening import screen_models
from molblender.config import get_cache_dir, set_cache_dir

def main():
    # 设置缓存
    set_cache_dir("representations", "./cache")

    # 获取表征器
    featurizer = get_featurizer("morgan_fp")
    smiles = ["CCO", "c1ccccc1"]

    # 生成表征
    features = featurizer.featurize(smiles)

    # 运行筛选
    results = screen_models(
        smiles_data=smiles,
        representations=["morgan_fp"],
        task_type="regression",
    )

    return results
```

### 新代码（推荐）

```python
# new_script.py
from molblender.api import (
    get_featurizer,
    list_featurizers,
    screen_models,
)
from molblender.config import config_manager

def main():
    # 设置缓存（使用 ConfigManager）
    config_manager.set_cache_dir("representations", "./cache")

    # 获取表征器（API相同）
    featurizer = get_featurizer("morgan_fp")
    smiles = ["CCO", "c1ccccc1"]

    # 生成表征（API相同）
    features = featurizer.featurize(smiles)

    # 运行筛选（API相同）
    results = screen_models(
        smiles_data=smiles,
        representations=["morgan_fp"],
        task_type="regression",
    )

    return results
```

**变更总结**：
- 导入路径改变：`molblender.api` 替代多个分散路径
- 函数名略有变化：`list_featurizers` 替代 `list_available_featurizers`
- 核心功能 API 完全相同：无需修改业务逻辑

## 相关文档

- [ConfigManager 使用指南](config_manager_guide.md) - 配置管理详情
- [架构概览](development/architecture.md) - 当前架构与迁移背景
- [快速开始指南](quickstart.md) - 新手入门指南

---

**最后更新**: 2026-03-10
**版本**: 1.0.0

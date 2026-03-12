# ConfigManager 使用指南

## 概述

`ConfigManager` 是 MolBlender 的统一配置管理器，采用单例模式确保全局配置一致性。它集中管理缓存路径、模型参数和日志配置。

## 快速开始

### 基本使用

```python
from molblender.config import ConfigManager, config_manager

# 方式1: 使用全局单例
cache_config = config_manager.get_cache_config()
print(f"Torch cache: {cache_config.torch_home}")
print(f"HuggingFace cache: {cache_config.hf_home}")

# 方式2: 创建新实例（仍然返回全局单例）
cm = ConfigManager()
model_config = cm.get_model_config()
print(f"Default n_jobs: {model_config.default_n_jobs}")
print(f"Default CV folds: {model_config.default_cv_folds}")
```

### 配置验证

```python
# 验证配置完整性
errors = config_manager.validate()
if errors:
    for error in errors:
        print(f"配置错误: {error}")
else:
    print("配置验证通过")
```

## 配置类型

### 1. CacheConfig（缓存配置）

```python
@dataclass
class CacheConfig:
    """缓存目录配置"""
    torch_home: str              # PyTorch Hub 缓存路径
    hf_home: str                 # HuggingFace Hub 缓存路径
    hf_endpoint: str | None      # HuggingFace 镜像端点
    representation_cache: str    # 分子表征缓存路径
    cache_enabled: bool          # 是否启用缓存
```

**默认值**：
- `torch_home`: `~/.cache/molblender/torch_hub`
- `hf_home`: `~/.cache/molblender/huggingface_hub`
- `representation_cache`: `.mbl_cache`（当前工作目录）
- `cache_enabled`: `True`

### 2. ModelConfig（模型配置）

```python
@dataclass
class ModelConfig:
    """模型相关配置"""
    default_n_jobs: int = -1                    # 默认并行任务数（-1表示使用所有核心）
    default_random_state: int = 42              # 默认随机种子
    default_cv_folds: int = 5                   # 默认交叉验证折数
    model_timeout: int | None = None            # 模型训练超时（秒）
    base_model_timeout: int = 300               # 基础模型超时（秒）
    min_model_timeout: int = 60                 # 最小模型超时（秒）
    max_model_timeout: int = 3600               # 最大模型超时（秒）
```

### 3. LoggingConfig（日志配置）

```python
@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "WARNING"                      # 日志级别
    format_string: str = "..."                  # 日志格式
    enable_deduplication: bool = True           # 是否启用日志去重
```

## 运行时配置管理

### 修改缓存目录

```python
# 修改特定类型的缓存目录
config_manager.set_cache_dir("torch", "/custom/path/to/torch")
config_manager.set_cache_dir("hf", "/custom/path/to/huggingface")
config_manager.set_cache_dir("representations", "/custom/path/to/repr")

# 获取缓存目录路径
torch_path = config_manager.get_cache_dir("torch")
hf_path = config_manager.get_cache_dir("hf")
repr_path = config_manager.get_cache_dir("representations")
```

**支持的缓存类型**：
- `"torch"` 或 `"torch_home"`: PyTorch Hub 缓存
- `"hf"`, `"hf_home"`, `"huggingface"`: HuggingFace Hub 缓存
- `"representations"`, `"repr"`, `"representation_cache"`: 分子表征缓存

## 环境变量配置

所有配置都可以通过环境变量覆盖：

### 缓存路径

```bash
# PyTorch Hub 缓存
export TORCH_HOME=/custom/path/to/torch

# HuggingFace Hub 缓存
export HF_HOME=/custom/path/to/huggingface

# HuggingFace 镜像端点
export HF_ENDPOINT=https://hf-mirror.com

# 分子表征缓存
export MOLBLENDER_CACHE_DIR=/custom/path/to/repr

# 启用/禁用缓存
export MOLBLENDER_CACHE_ENABLED=true
```

### 模型参数

```bash
# 并行任务数（-1表示使用所有核心）
export MOLBLENDER_DEFAULT_N_JOBS=-1

# 随机种子
export MOLBLENDER_RANDOM_STATE=42

# 交叉验证折数
export MOLBLENDER_DEFAULT_CV_FOLDS=5

# 模型训练超时（-1表示无超时）
export MOLBLENDER_MODEL_TIMEOUT=600

# 基础模型超时
export MOLBLENDER_BASE_MODEL_TIMEOUT=300

# 最小/最大模型超时
export MOLBLENDER_MIN_MODEL_TIMEOUT=60
export MOLBLENDER_MAX_MODEL_TIMEOUT=3600
```

### 日志配置

```bash
# 日志级别
export MOLBLENDER_LOG_LEVEL=WARNING

# 启用日志去重
export MOLBLENDER_LOG_DEDUPLICATION=true
```

## 向后兼容性

`ConfigManager` 完全兼容旧的 `settings` 模块。旧的导入路径仍然可用：

```python
# 旧方式（仍然支持，但标记为 _legacy）
from molblender.config import get_cache_dir as get_cache_dir_legacy
from molblender.config import set_cache_dir as set_cache_dir_legacy

# 新方式（推荐）
from molblender.config import config_manager
cache_dir = config_manager.get_cache_dir("torch")
```

## 最佳实践

### 1. 在应用启动时配置

```python
# app.py
from molblender.config import config_manager

def initialize_app():
    """应用启动时初始化配置"""
    # 验证配置
    errors = config_manager.validate()
    if errors:
        raise RuntimeError(f"配置错误: {errors}")

    # 设置自定义缓存路径（可选）
    config_manager.set_cache_dir("representations", "./cache/repr")

    # 获取配置
    cache_config = config_manager.get_cache_config()
    model_config = config_manager.get_model_config()

    return cache_config, model_config
```

### 2. 在测试中使用临时配置

```python
# test_module.py
import os
import pytest
from molblender.config import config_manager

@pytest.fixture
def temp_cache_dir(tmp_path):
    """使用临时缓存目录"""
    old_path = os.environ.get("MOLBLENDER_CACHE_DIR")
    os.environ["MOLBLENDER_CACHE_DIR"] = str(tmp_path)

    # 注意：由于单例模式，ConfigManager需要在测试中谨慎使用
    # 建议在测试间重新创建进程或使用环境变量

    yield tmp_path

    # 恢复原环境变量
    if old_path is not None:
        os.environ["MOLBLENDER_CACHE_DIR"] = old_path
    else:
        os.environ.pop("MOLBLENDER_CACHE_DIR", None)
```

### 3. 配置优先级

配置读取优先级（从高到低）：
1. 运行时设置：`config_manager.set_cache_dir()`
2. 环境变量：`export MOLBLENDER_*=...`
3. 默认值：硬编码在ConfigManager中

## 常见问题

### Q: 为什么使用单例模式？

A: 确保全局配置一致性。多个组件访问相同的配置实例，避免配置不一致导致的问题。

### Q: 如何在多进程环境中使用？

A: 每个进程有自己的ConfigManager实例，但它们读取相同的环境变量，因此配置保持一致。

### Q: 缓存目录不存在会怎样？

A: ConfigManager会在`__post_init__`中自动创建缓存目录。如果创建失败，会记录错误日志。

### Q: 如何禁用缓存？

A: 设置环境变量：
```bash
export MOLBLENDER_CACHE_ENABLED=false
```

## 相关文档

- [架构概览](development/architecture.md) - 当前架构与分层说明
- [API使用指南](api_guide.md) - 统一API层使用
- [迁移指南](migration_guide.md) - 从旧API迁移到新API

---

**最后更新**: 2026-03-10
**版本**: 1.0.0

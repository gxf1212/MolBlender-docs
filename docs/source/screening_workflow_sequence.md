# Screening运行链路与回调持久化时序图

## 概述

本文档描述`StandardScreener`的完整运行流程，重点关注回调持久化机制和数据库存储时序。

## 组件架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         StandardScreener                          │
├─────────────────────────────────────────────────────────────────┤
│  • data_handler: ProfessionalDataHandler                         │
│  • evaluator: StandardEvaluator                                  │
│  • result_processor: ProfessionalResultProcessor                  │
│  • model_registry: ModelRegistry                                  │
│  • result_callback: Optional[Callable]                            │
│  • db_manager: Optional[DatabaseManager]                          │
└─────────────────────────────────────────────────────────────────┘
```

## 主流程时序图

```
User -> StandardScreener: run_screening()
StandardScreener -> DataHandler: prepare_dataset()
DataHandler --> StandardScreener: prepared_data

StandardScreener -> ModelRegistry: get_models()
ModelRegistry --> StandardScreener: models

loop For each representation
    StandardScreener -> DataHandler: split_data()
    DataHandler --> StandardScreener: X_train, X_test, y_train, y_test
    
    StandardScreener -> Database: check_existing()
    Database --> StandardScreener: skip_if_exists
    
    par For each model
        StandardScreener -> Evaluator: evaluate_model()
        Evaluator --> StandardScreener: ModelResult
        
        StandardScreener -> ResultCallback: callback(result, modality)
        ResultCallback -> Database: insert_result()
        ResultCallback --> StandardScreener: success=True/False
        
        StandardScreener -> ResultProcessor: process()
    end
    
    StandardScreener -> StandardScreener: _cleanup_representation_data()
end

StandardScreener --> User: all_results
```

## 关键方法调用链

### 1. 数据准备阶段

```python
# run_screening() 入口
prepared_data = self.data_handler.prepare_dataset(
    dataset,
    target_column,
    representation_names,
)
# 返回: {'representations': {name: X}, 'targets': y}
```

### 2. 回调持久化阶段

```python
def _save_result_with_callback(self, result, X_train, repr_name):
    if not self.result_callback:
        return
    
    try:
        # ① 推断数据模态
        modality = self._infer_modality(X_train)
        # 'vector', 'matrix', 'image', 'unknown'
        
        # ② 调用用户回调
        self.result_callback(result, modality)
        
    except Exception as cb_error:
        model_name = getattr(result, "model_name", "Unknown")
        logger.warning(
            f"Result callback failed for {model_name} + {repr_name}: {cb_error}"
        )
```

### 3. 内存清理阶段

```python
def _cleanup_representation_data(self, repr_name, X_train, X_test, 
                                  y_train, y_test, X=None):
    try:
        del X_train, X_test, y_train, y_test
        if X is not None:
            del X
        import gc
        gc.collect()
    except Exception as cleanup_e:
        logger.debug(f"Memory cleanup failed: {cleanup_e}")
```

## 回调机制详解

### 回调接口定义

```python
ResultCallback = Callable[[ModelResult, str], bool]

# 参数:
#   result: ModelResult - 评估结果对象
#   modality: str - 数据模态 ('vector', 'matrix', 'image')
# 返回:
#   bool - True=保存成功, False=保存失败
```

### 典型回调实现

```python
def database_persistence_callback(db_manager, session_id):
    """创建数据库持久化回调函数"""
    
    def callback(result: ModelResult, modality: str) -> bool:
        try:
            db_manager.insert_model_result(
                session_id=session_id,
                model_name=result.model_name,
                representation_name=result.representation_name,
                primary_metric=result.primary_metric,
                all_metrics=result.all_metrics,
                cv_scores=result.cv_scores,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save result: {e}")
            return False
    
    return callback
```

## 数据库存储时序

```
StandardScreener -> Callback: callback(result, modality)

alt 回调成功
    Callback -> Database: BEGIN TRANSACTION
    Callback -> Database: INSERT INTO model_results
    Callback -> Database: UPDATE session_summary
    Callback -> Database: COMMIT
    Callback --> StandardScreener: True
else 回调失败
    Callback -> Callback: raise Exception or return False
    Callback --> StandardScreener: False
end
```

## 性能优化要点

### 1. 结果缓存

```python
# 检查数据库中是否已有此结果
if should_skip_existing_result(
    self.db_manager,
    self.session_id,
    model_name,
    repr_name,
):
    logger.info(f"Skipping {model_name} + {repr_name} (already exists)")
    continue
```

### 2. 内存管理

```python
# 每个representation评估完后立即清理
for repr_name, X in prepared_data["representations"].items():
    # ... 评估所有模型 ...
    
    # 清理当前representation的数据
    self._cleanup_representation_data(
        repr_name, X_train, X_test, y_train, y_test, X
    )
```

## 相关文档

- `CLAUDE.md` (项目根目录): 整体架构
- `models/CLAUDE.md`: Models模块规范
- `models/api/CLAUDE.md`: API接口规范

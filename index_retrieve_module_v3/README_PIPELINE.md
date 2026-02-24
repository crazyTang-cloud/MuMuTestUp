# 测试修复辅助信息检索流水线 (Test Fix RAG Pipeline)

## 概述

这是一个成本优先、多阶段的自动化流水线，用于渐进式检索辅助信息以帮助修复失败的测试用例。流水线采用漏斗型设计，在每个阶段都设有"熔断器"机制，以避免不必要的计算和成本。

## 设计理念

### 1. 低耦合 (Low Coupling)
- 流水线封装为独立模块，通过明确定义的接口与外部交互
- 每个阶段都是独立的类，可以单独测试和优化
- 使用标准的数据结构进行阶段间通信

### 2. 成本优先 (Cost-Aware)
- 漏斗型流程：随着流程深入，时间、算力和金钱开销逐渐增加
- 每个阶段都有"熔断/跳出"机制，避免不必要的计算
- Phase 1: 纯 LLM 分析
- Phase 2: 轻量级 SQL 元数据检索
- Phase 3: 完整向量嵌入和语义搜索（成本）

## 总架构概览

![mermaid-diagram (2)](README_PIPELINE.assets/mermaid-diagram (2).png)

## 更细节一点的架构，图不好画，可以看下面的文字部分理解

![mermaid-diagram (1)](README_PIPELINE.assets/mermaid-diagram (1).png)

## 流水线阶段详解

### Phase 0: 环境准备 (Git Reset)

**目的**: 确保仓库处于正确的 commit 状态，保证上下文一致性。

**操作**:

- 使用 `git reset --hard` 将仓库重置到指定的 `error_commit_id`
- 检查工作树状态
- 验证 commit ID 有效性

**输出**: 成功/失败状态

### Phase 1: 初步诊断 (Triage Agent)

**目的**: 判断错误是否简单到无需查阅仓库代码即可修复。

**输入**:
- `error_test_code_log`: 失败的测试代码
- `error_message`: 错误消息
- `error_log`: 完整错误日志

**决策逻辑**:
- 调用 LLM 分析错误复杂度
- 简单错误（如语法错误、明显的逻辑漏洞）→ **熔断器触发**，跳过后续阶段
- 复杂错误（需要理解生产代码）→ 进入 Phase 2

**输出**:
- 状态: `SKIPPED_AT_PHASE_1` 或 `PROCEED`
- 原因: 决策理由
- 上下文: None（如果跳过）

### Phase 2: 一级检索 - 迭代式元数据/SQL 检索

**目的**: 通过快速的 SQL 查询获取代码元数据，避免昂贵的向量嵌入。

**操作流程**:
1. 构建轻量级索引（仅 SQLite，跳过向量嵌入）
2. 迭代式查询（最多 3 轮）：
   - **生成 SQL**: Agent 根据错误信息生成 SQL 查询
   - **安全验证**: 
     - 必须包含 `WHERE is_test = 0`（严禁检索测试代码）
     - 必须包含 `LIMIT` 子句（防止 Token 爆炸）
     - 仅允许 SELECT 查询
   - **执行查询**: 在 SQLite 数据库中执行
   - **评估结果**: Agent 判断元数据是否足以定位修复方案
     - **Case A (充足)**: 终止循环，**熔断器触发**
     - **Case B (重试)**: 修改 SQL，进入下一轮
     - **Case C (继续)**: 循环结束仍未满足，进入 Phase 3

**数据库表结构** (`methods` 表):
- `class_name`: 完全限定类名
- `method_name`: 方法名
- `signature`: 完整方法签名
- `javadoc`: 方法文档
- **`body`: 方法体源码**（完整实现代码）
- `is_test`: 是否为测试代码
- 以及其他元数据字段

**🔥 关键特性 - 完整内容评估**:
- **Agent 可以看到完整的方法体**：当 SQL 查询包含 `body` 字段时，LLM 在评估阶段会看到完整的方法实现代码
- **智能查询策略**：
  - 浅层探索：`SELECT class_name, method_name, signature` (快速浏览，适合探索阶段)
  - **深度分析**：`SELECT body, signature, javadoc, class_name` (看到完整代码，适合定位 bug)
- **展示策略**：取前 5 条结果，最多展示 10K tokens，确保 LLM 能充分分析代码细节

**输出**:
- 状态: `STOPPED_AT_PHASE_2` 或 `PROCEED`
- 原因: 决策理由
- 上下文: 筛选后的有用信息，**包含完整方法体**（如果 SQL 查询包含 body 字段）

### Phase 3: 二级检索 - 增量式语义/RAG 检索

**目的**: 使用向量嵌入进行语义相似度搜索，找到最相关的生产代码。

**核心创新 - 模块级增量 Embedding**:
- **不再一次性向量化整个仓库**，而是按模块逐步 embedding
- 每轮次 Agent 自主决策：是 embedding 新模块，还是优化查询？
- 成本优化：只 embedding 必要的模块，避免浪费
- **复用 Phase 2 的 SQLite**：不重新 parse 文件，直接从数据库读取后 embedding

**操作流程（最多 3 轮）**:

每一轮包含两个阶段：

**阶段 1: 决策 - Embedding 还是查询优化？**
- Agent 分析当前状态：
  - 已嵌入哪些模块？
  - 上一轮检索结果如何？
  - 错误信息暗示哪个模块可能相关？
- **决策 A (EMBED_MODULE)**：
  - 选择一个新模块进行 embedding
  - 仅向量化该模块的代码（增量添加，不清空已有）
  - 适用场景：上轮结果不相关、需要探索新模块
- **决策 B (REFINE_QUERY)**：
  - 不 embedding 新模块，仅优化搜索查询
  - 在已嵌入的模块中重新搜索
  - 适用场景：上轮结果接近但不够精确

**阶段 2: 检索与评估**
- **生成查询**: 构建语义搜索查询
- **决定目标**: methods、fields 或 both
- **执行检索**: 
  - `include_tests=False`（严禁检索测试代码）
  - `top_k=8`（默认返回 8 条结果）
  - 仅在已 embedding 的模块中搜索
- **评估结果**: 判断检索到的代码是否有用
  - **USEFUL（有用）**: 提取代码，返回成功 ✓
  - **RETRY（重试）**: 进入下一轮，agent 重新决策
  - **NOT_USEFUL（无用）**: 
    - 如果还有轮次剩余 → 进入下一轮，尝试不同策略
    - 如果已是最后一轮 → 流水线失败退出

**成本优势**:
- 典型场景：仅 embedding 1-2 个相关模块（而非全部 10+ 个）
- 成本节省：60-80%（相比全量 embedding）
- 时间节省：更快的迭代和反馈

**输出**:
- 状态: `SUCCESS_PHASE_3` 或 `FAILED_AT_PHASE_3`
- 原因: 决策理由
- 上下文: 筛选后的 RAG 源信息（如果有用）
- 元数据: 已嵌入的模块列表、每轮决策历史

## 使用方法

### 前置要求

1. **环境变量**:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   export OPENAI_BASE_URL="https://api.openai.com/v1"  # 可选
   ```

2. **依赖安装**:
   ```bash
   pip install -r requirements.txt
   ```

### 基础用法

```bash
# 使用默认文件夹里的 config.py 运行流水线
python run_pipeline.py

# 使用自定义配置文件
python run_pipeline.py --config-file my_config.py

# 启用详细日志
python run_pipeline.py --verbose
```

### 高级选项

```bash
# 跳过索引重建（用于快速迭代测试）
python run_pipeline.py --skip-rebuild

# 跳过 Git 重置（用于测试）
python run_pipeline.py --skip-phase0

# 自定义输出目录
python run_pipeline.py --output-dir ./my_results

# 自定义迭代轮数
python run_pipeline.py --max-sql-rounds 5 --max-rag-rounds 5

# 自定义 RAG 检索数量
python run_pipeline.py --rag-top-k 10
```

### 配置文件示例

在项目中创建 `config.py`：

```python
from dataclasses import dataclass
import os

@dataclass
class Config:
    # 仓库信息
    repo_path: str = "/path/to/your/java/project"
    error_commit_id: str = "abc123def"
    
    # 错误信息
    error_test_code_log: str = """
    @Test
    public void testSomething() {
        // 失败的测试代码
    }
    """
    
    error_message: str = "expected:<X> but was:<Y>"
    
    error_log: str = """
    [完整的错误日志]
    """
    
    # LLM 配置
    agent_model: str = "gpt-4o-2024-11-20"
    
    def __post_init__(self):
        # 验证环境变量
        if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError("OPENAI_API_KEY is not set")
```

## 输出格式

流水线执行完成后，会在 `pipeline_results/` 目录生成 JSON 文件：

```
pipeline_results/
└── pipeline_20241222_143025_912d8c48.json
```

### JSON 输出结构

```json
{
  "timestamp": "2024-12-22T14:30:25",
  "final_status": "SUCCESS_PHASE_3",
  "final_reason": "找到相关的生产代码实现",
  "final_context": "筛选后的代码片段...",
  
  "phase0": {
    "phase_name": "Phase 0: Git Reset",
    "status": "success",
    "reason": "Successfully reset to commit 912d8c48c",
    "duration_seconds": 2.5
  },
  
  "phase1": {
    "phase_name": "Phase 1: Triage",
    "status": "proceed",
    "reason": "错误涉及类型不匹配，需要查看生产代码实现",
    "duration_seconds": 3.2
  },
  
  "phase2": {
    "phase_name": "Phase 2: SQL Search",
    "status": "proceed",
    "reason": "SQL 查询找到了方法签名，但需要完整实现",
    "rounds": 3,
    "duration_seconds": 15.8
  },
  
  "phase3": {
    "phase_name": "Phase 3: RAG Search",
    "status": "success",
    "reason": "找到相关方法实现，可以定位问题",
    "context": "相关代码片段...",
    "rounds": 2,
    "duration_seconds": 45.3
  },
  
  "total_duration_seconds": 66.8,
  "config": { /* 配置快照 */ }
}
```

### 状态说明

- `SKIPPED_AT_PHASE_1`: Phase 1 判断错误简单，无需检索
- `STOPPED_AT_PHASE_2`: Phase 2 SQL 检索已找到充足信息
- `SUCCESS_PHASE_3`: Phase 3 RAG 检索成功找到有用代码
- `FAILED_AT_PHASE_3`: Phase 3 完成但未找到有用信息
- `ERROR`: 流水线执行出错




## 文件结构

```
rag_demo/
├── pipeline/                    # 流水线主包
│   ├── __init__.py
│   ├── models.py               # 数据结构定义
│   ├── agent.py                # LLM Agent 封装
│   ├── git_handler.py          # Git 操作
│   ├── index_builder.py        # 索引构建封装（indexer外做的Wrapper）
│   ├── pipeline.py             # 主流水线编排器
│   └── phases/                 # 各阶段实现
│       ├── __init__.py
│       ├── phase1_triage.py    # Phase 1 实现
│       ├── phase2_sql.py       # Phase 2 实现
│       └── phase3_rag.py       # Phase 3 实现
├── index_data/                 # 用于存放sql表和chromadb库
├── indexer/                    # 1.0版本的索引器 （V1.0）
├── build_index.py              # 1.0版本的索引构建入口（V1.0）
├── test_query.py               # 1.0版本的检索器入口（V1.0）
├── README.md                   # 1.0版本的使用教程（V1.0）
├── run_pipeline.py             # CLI 入口
├── config.py                   # 配置文件
├── README_PIPELINE.md          # V2.0的使用教程
└── pipeline_results/           # 输出目录
```


**Q: 可以并行运行多个流水线吗？**

A: 可以，但需要注意：
- 使用不同的 `output_dir`
- Git 操作可能冲突（建议使用不同仓库副本）
- 注意 API 速率限制








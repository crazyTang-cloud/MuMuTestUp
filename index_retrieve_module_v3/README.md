# Java 项目 AutoFix-RAG 索引器 (AutoFix-RAG Indexer for Java)

这是一个专为 Java 项目设计的高级代码索引器，旨在支持基于 RAG（检索增强生成）的测试自动修复。该工具使用 `tree-sitter` 解析 Java 代码，将方法元数据存储在 SQLite 中，并利用 OpenAI 创建语义嵌入（Embeddings）以实现高效检索。

## ✨ 功能特性

  - **Maven 多模块支持**: 自动检测并处理 Maven 多模块项目结构。
  - **专业级解析**: 采用 `tree-sitter-java` 进行精准的、生产级的 Java 语法解析。
  - **方法与字段双索引**: 同时索引 Java 方法和类字段（成员变量），支持独立查询。
  - **双重存储架构**: SQLite 用于冷存储（完整的方法体/字段信息）+ ChromaDB 用于热存储（向量嵌入）。
  - **语义搜索**: 利用 OpenAI Embeddings 实现基于代码含义而非仅基于文本匹配的相关性查找。
  - **智能测试检测**: 自动区分生产代码与测试代码。
  - **可视化进度追踪**: 提供清晰的日志记录和进度条显示。
  - **健壮的错误处理**: 单个文件的解析失败不会中断整个索引过程。

## 🏗️ 架构设计

```
Java 项目 → 文件遍历器 → Java 解析器 → 双重存储系统
                                      ├─→ SQLite (元数据 + 方法体源码)
                                      └─→ ChromaDB (向量嵌入)
```

### 数据模型

**SQLite** (`assets.db`):

  - **methods 表**: 存储完整的方法信息：方法签名、Javadoc、方法体源码、起止行号。
  - **fields 表**: 存储类字段信息：字段名、类型、修饰符、初始值、Javadoc。
  - 支持通过类名、文件路径或行号进行精确检索。
  - 针对快速查询进行了索引优化。

**ChromaDB**:

  - **java_methods collection**: 存储方法的语义嵌入（`类名 + 方法签名 + Javadoc`）。
  - **java_fields collection**: 存储字段的语义嵌入（`类名 + 字段声明 + Javadoc`）。
  - 支持相似性搜索，用于查找逻辑相关的代码。
  - 元数据包含 `sqlite_id`，用于回溯关联到完整的源码信息。

## 🚀 安装指南

### 先决条件

  - Python 3.10 或更高版本
  - OpenAI API Key（用于生成向量嵌入）
  - 包含源代码的 Java 项目

### 设置步骤

1.  **安装依赖**:

    ```bash
    pip install -r requirements.txt
    ```

## 📖 使用方法

### 基础用法

对一个 Java 项目建立索引 + 优化策略下的embedding：

```bash
python build_index.py --project-root "D:\identification_update\update\repository\flink"
```

### 高级选项

```bash
# 详细日志模式 (显示调试信息)
python build_index.py --project-root /path/to/java/project --verbose

# 追加到现有索引 (不重建/不清空旧数据)
python build_index.py --project-root /path/to/java/project --no-rebuild

# 仅索引某个模块
python build_index.py --project-root "D:\identification_update\update\repository\ta4j" --module ta4j

# 仅创建sqlite, 忽略chromaDB 的嵌入过程
python build_index.py --project-root "C:\Users\zyc\Desktop\files\agent\dromara-hutool" --skip-vector

# 使用环境变量中定义的项目根目录
export JAVA_PROJECT_ROOT=/path/to/java/project
python build_index.py
```

### 输出产物

索引器会在当前工作目录下创建一个 `index_data/` 目录：

```
index_data/
├── assets.db          # SQLite 数据库 (包含 methods 和 fields 两张表)
└── chroma_db/         # ChromaDB 向量库 (包含 java_methods 和 java_fields 两个集合)
```


## 🛠️ 工作原理

### 1\. 文件发现 (`file_walker.py`)

  - 遍历项目目录。
  - 通过检测 `pom.xml` 文件识别 Maven 模块。
  - 过滤掉构建目录（如 `target/`, `.git/` 等）。
  - 区分测试文件与生产代码文件。

### 2\. 代码解析 (`java_parser.py`)

使用 `tree-sitter-java` 提取以下信息：

  - **包声明**: 用于构建全限定类名。
  - **类信息**: 包括嵌套类和内部类。
  - **方法信息**:
    - 方法签名：完整的签名，包含修饰符、返回类型、参数列表
    - Javadoc 注释：方法的文档说明
    - 方法体：完整的源代码实现
  - **字段信息**:
    - 字段名称和类型
    - 修饰符（public, private, static, final 等）
    - 初始值（如果有）
    - Javadoc 注释
  - **行号**: 用于精确定位。

### 3\. SQLite 存储 (`storage.py`)

存储所有方法和字段信息，并建立以下索引：

**methods 表**:
```sql
CREATE TABLE methods (
    id TEXT PRIMARY KEY,
    module_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    class_name TEXT NOT NULL,
    method_name TEXT NOT NULL,
    signature TEXT NOT NULL,
    javadoc TEXT,
    body TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    is_test BOOLEAN NOT NULL,
    imports TEXT
);
```

**fields 表**:
```sql
CREATE TABLE fields (
    id TEXT PRIMARY KEY,
    module_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    class_name TEXT NOT NULL,
    field_name TEXT NOT NULL,
    field_type TEXT NOT NULL,
    modifiers TEXT,
    initializer TEXT,
    javadoc TEXT,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    is_test BOOLEAN NOT NULL,
    imports TEXT
);
```

### 4\. 向量嵌入 (`vector_store.py`)

每个向量记录包含：
ID（唯一标识）:
   "user-service:src/main/java/com/app/User.java#a1b2c3d4e5f6"
Document（被嵌入的文本）:
   Class: com.app.User
   Method: public boolean validate(String password)
   Doc: /** Validates user password ... */
Embedding（向量，由 OpenAI 生成，1536 维）:
   [0.0234, -0.0567, 0.0891, ..., 0.0123]  // 1536 个浮点数
Metadata（元数据，用于过滤和回溯）:
   {
     "sqlite_id": "user-service:src/main/java/com/app/User.java#a1b2c3d4e5f6",
     "class_name": "com.app.User",
     "module_name": "user-service",
     "method_name": "validate",
     "file_path": "src/main/java/com/app/User.java",
     "is_test": "False"
   }


## ⚙️ 配置说明

### IndexerConfig 选项

你可以通过修改 `indexer/config.py` 来自定义索引器：

  - `ignored_dirs`: 要跳过的目录（默认：`target`, `.git`, `.idea` 等）。
  - `sqlite_db_name`: SQLite 数据库文件名。
  - `chroma_collection_name`: ChromaDB 集合名称。
  - `java_file_pattern`: Java 文件匹配模式。

### 测试方法的嵌入策略

**设计理念**：所有方法（包括测试方法）都会被嵌入到向量库中，但**默认检索时会过滤掉测试方法**。

**原因**：
- 主要用例：修复失败的测试 → 需要查找最新的**生产代码**
- 测试方法在向量检索结果中会产生噪音，降低相关性
- 所有测试方法仍保存在 SQLite 中供分析使用

**使用方式**：

```bash
# 查询方法（默认）
python test_query.py --query "eu.verdelhan.ta4j.AnalysisCriterion whether is better" --top-k 5

# 查询字段
python test_query.py --query "eu.verdelhan.ta4j.TADecimal first" --target fields --top-k 5

# 同时查询方法和字段
python test_query.py --query "user authentication" --target both --top-k 5

# 包含测试代码在查询结果中
python test_query.py --query "validation logic" --include-tests
```

### 查询目标选项

`--target` 参数可以指定查询目标：
- `methods`（默认）：仅查询方法
- `fields`：仅查询字段
- `both`：同时查询方法和字段 


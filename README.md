# MuMuTestUp

**MuMuTestUp** is the official implementation of the paper:

> ***MuMuTestUp*: Mutation-based Multi-Agent Test Case Update**

MuMuTestUp is a multi-agent LLM framework for automatically updating Java test cases in response to source code changes. It leverages mutation testing (PITest) and coverage analysis (JaCoCo) to iteratively guide LLM agents toward high-quality, effective test updates.

---

## Quick Start

### Entry Points

| Script | Description |
|--------|-------------|
| `run_complete_beam.py` | **Sequential runner** — processes dataset samples one by one |
| `run_complete_beam_multi.py` | **Concurrent runner** — processes multiple samples in parallel using `ProcessPoolExecutor` |

```bash
# Sequential (single sample or full dataset)
python run_complete_beam.py

# Concurrent (recommended for large-scale evaluation)
python run_complete_beam_multi.py
```

---

## Environment Setup

**Python version:** 3.10

Install all dependencies before running:

```bash
pip install -r requirements.txt
```

---

## Dataset

Dataset files are located under `dataset/`, organized by organization and project:

```
dataset/
├── apache/druid/data.json
├── dromara/hutool/data.json
├── neo4j/neo4j-java-driver/data.json
├── Wikidata/Wikidata-Toolkit/data.json
├── wmixvideo/nfe/data.json
├── damianszczepanik/cucumber-reporting/data.json
├── mock-server/mockserver/data.json
├── pac4j/pac4j/data.json
├── j256/ormlite-core/data.json
└── sniffy/sniffy/data.json
```

Each `data.json` contains a list of samples. Each sample includes fields such as:
`project`, `bSource`, `aSource`, `bPath`, `aPath`, `bCommit`, `aCommit`, `hunks`, `focal_method`, `src_java_version`, `tgt_java_version`, `input`, etc.

### Cloning Repositories

**Before running any sample**, the corresponding project repository must be cloned into the `repos/` directory. The expected path format is `repos/<org>/<project>/`:

```bash
git clone https://github.com/dromara/hutool                  repos/dromara/hutool
git clone https://github.com/apache/druid                    repos/apache/druid
git clone https://github.com/neo4j/neo4j-java-driver         repos/neo4j/neo4j-java-driver
git clone https://github.com/Wikidata/Wikidata-Toolkit        repos/Wikidata/Wikidata-Toolkit
git clone https://github.com/wmixvideo/nfe                   repos/wmixvideo/nfe
git clone https://github.com/damianszczepanik/cucumber-reporting repos/damianszczepanik/cucumber-reporting
git clone https://github.com/mock-server/mockserver          repos/mock-server/mockserver
git clone https://github.com/pac4j/pac4j                     repos/pac4j/pac4j
git clone https://github.com/j256/ormlite-core               repos/j256/ormlite-core
git clone https://github.com/sniffy/sniffy                   repos/sniffy/sniffy
```

---

## Configuration (`config.py`)

All configuration lives in `config.py`. The key dataclasses are `LLMConfig`, `JavaConfig`, and `FrameworkConfig`, all composed under the top-level `Config`.

---

### `LLMConfig` — LLM Provider Settings

MuMuTestUp supports **local Ollama LLMs** and **external OpenAI-compatible LLMs**.

#### Using a Local Ollama LLM

Set `provider = "ollama"` and configure the Ollama server address:

```python
llm = LLMConfig(
    provider="ollama",
    ollama_host="http://localhost",   # Ollama server host
    ollama_port=11434,                # Ollama server port
    model="qwen2.5-coder:7b",        # Model name as shown in `ollama list`
    temperature=0.0,
    timeout=3000,
)
```

#### Using an External LLM (OpenAI-compatible API)

Set `provider = "openai"` and fill in **`api_key`** and **`api_url`**:

```python
llm = LLMConfig(
    provider="openai",
    api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxx",             # Your API key
    api_url="https://api.openai.com/v1/chat/completions",  # Full endpoint URL
    model="gpt-4o",
    temperature=0.0,
    timeout=3000,
    max_retries=4,
    retry_delay=1.0,
)
```

- **`api_key`**: Authentication token for the external LLM service.
- **`api_url`**: Full URL of the chat completions endpoint. Supports any OpenAI-compatible API (OpenAI, DeepSeek, Qwen, self-hosted vLLM, etc.).

You can also configure via **environment variables** (override defaults at runtime):

| Variable | Description |
|----------|-------------|
| `LLM_PROVIDER` | `"ollama"` or `"openai"` |
| `LLM_API_URL` | External API endpoint |
| `LLM_API_KEY` | API key |
| `LLM_MODEL` | Model name |
| `LLM_TEMPERATURE` | Sampling temperature (float) |
| `LLM_TIMEOUT` | Request timeout in seconds |
| `OLLAMA_HOST` | Ollama host (default: `http://localhost`) |
| `OLLAMA_PORT` | Ollama port (default: `11434`) |

---

### `JavaConfig` — Java & Maven Execution Settings

MuMuTestUp compiles and tests Java projects using Maven, supporting JDK 8, 11, 17, and 21.

| Parameter | Description |
|-----------|-------------|
| `java_homes` | Dict mapping Java version string → JDK home path. Must cover all versions used by the dataset (8, 11, 17, 21). |
| `maven_home` | Absolute path to your Maven installation (e.g., `apache-maven-3.9.6`). |
| `maven_repo_base` | Base directory for per-project Maven local repositories. Each project gets its own subdirectory to avoid dependency conflicts. |
| `repos_dir` | Absolute path to the root directory where all dataset repositories are cloned. Must match the `git clone` targets above. |
| `logs_base_dir` | Directory where per-sample execution logs are written. Ablation-mode suffixes are appended automatically (e.g., `logs_wo_mut`). |
| `reports_base_dir` | Directory where JaCoCo coverage and PITest mutation XML reports are stored. Ablation suffixes applied automatically. |
| `github_tokens` | List of GitHub personal access tokens. Used when the framework fetches repository metadata to avoid API rate limits. Optional but recommended. |
| `test_timeout` | Maximum seconds allowed for one full test execution cycle (compile + run + JaCoCo + PITest). Increase for large projects. |

Example:

```python
java = JavaConfig(
    java_homes={
        "8":  "/data/david/java/jdk1.8.0_391",
        "11": "/data/david/java/jdk-11.0.22",
        "17": "/data/david/java/jdk-17.0.10",
        "21": "/data/david/java/jdk-21.0.1",
    },
    maven_home="/data/david/maven/apache-maven-3.9.6",
    maven_repo_base="/data/david/maven_repo",
    repos_dir="/data/david/project/mumutestup/repos",
    logs_base_dir="/data/david/project/mumutestup/logs",
    reports_base_dir="/data/david/project/mumutestup/reports",
    github_tokens=["ghp_your_token_here"],
    test_timeout=6000,
)
```

---

### Full Config Example

```python
from config import Config, LLMConfig, JavaConfig

config = Config(
    llm=LLMConfig(
        provider="openai",
        api_key="sk-xxxx",
        api_url="https://api.openai.com/v1/chat/completions",
        model="gpt-4o",
        temperature=0.0,
        max_retries=4,
    ),
    java=JavaConfig(
        java_homes={
            "8":  "/data/david/java/jdk1.8.0_391",
            "11": "/data/david/java/jdk-11.0.22",
            "17": "/data/david/java/jdk-17.0.10",
            "21": "/data/david/java/jdk-21.0.1",
        },
        maven_home="/data/david/maven/apache-maven-3.9.6",
        maven_repo_base="/data/david/maven_repo",
        repos_dir="/data/david/project/mumutestup/repos",
        logs_base_dir="/data/david/project/mumutestup/logs",
        reports_base_dir="/data/david/project/mumutestup/reports",
        github_tokens=[],
        test_timeout=6000,
    ),
)
```

---

## Project Structure

```
mumutestup/
├── agents/                    # LLM agents (coverage, mutation, retrieval, error, etc.)
├── config.py                  # All configuration (LLMConfig, JavaConfig, FrameworkConfig)
├── dataset/                   # Benchmark dataset organized by org/project/data.json
├── dataset_loader.py          # Dataset loading and preprocessing utilities
├── index_retrieve_module_v3/  # Retrieval module (SQLite + ChromaDB RAG)
├── java_test_executor.py      # Maven / JaCoCo / PITest execution engine
├── llm/                       # LLM client wrappers (Ollama + OpenAI-compatible)
├── models/                    # Shared data models and result types
├── orchestrator/              # Multi-agent orchestration and iteration logic
├── repos/                     # Cloned project repositories (git clone targets)
├── requirements.txt           # Python 3.10 dependencies
├── run_complete_beam.py       # Sequential execution entry point
├── run_complete_beam_multi.py # Concurrent execution entry point
└── utils/                     # Logging helpers and utility functions
```

---

## Citation

If you use MuMuTestUp in your research, please cite:

```bibtex
@article{mumutestup2025,
  title   = {MuMuTestUp: Mutation-based Multi-Agent Test Case Update},
  year    = {2025}
}
```

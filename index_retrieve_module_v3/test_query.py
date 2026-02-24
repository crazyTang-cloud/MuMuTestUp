#!/usr/bin/env python3
"""
Quick test script for verifying build_index results using a few queries.

功能：
- 检查 SQLite 和 Chroma 是否已生成
- 打印基础统计信息（方法数量、模块数量等）
- 跑几条示例 query，展示 RAG 检索的“长相”

用法（在项目根目录执行）：
    python test_query.py --query "eu.verdelhan.ta4j.Tick how to get begin time of the state" 

可选参数：
    --query "your search text"    使用自定义 query 进行语义检索
    --top-k 5                     控制返回条数（默认 5）
    --include-tests               检索时包含测试方法（默认只看生产代码）
"""
import argparse
import os
from pathlib import Path

from indexer.storage import SQLiteStorage
from indexer.vector_store import VectorStore


def print_header(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")


def check_sqlite(db_path: Path) -> None:
    print_header("检查 SQLite 索引 (assets.db)")

    if not db_path.exists():
        print(f"❌ 未找到 SQLite 数据库: {db_path}")
        print("   请先运行 build_index.py 生成索引。")
        return

    with SQLiteStorage(db_path) as storage:
        stats = storage.get_statistics()

        print("📊 SQLite 统计信息:")
        print(f"  总方法数        : {stats['total_methods']}")
        print(f"  - 生产代码方法数: {stats['production_methods']}")
        print(f"  - 测试代码方法数: {stats['test_methods']}")
        print(f"  总字段数        : {stats['total_fields']}")
        print(f"  - 生产代码字段数: {stats['production_fields']}")
        print(f"  - 测试代码字段数: {stats['test_fields']}")
        print(f"  模块数          : {stats['modules']}")
        print(f"  文件数          : {stats['files']}")
        print(f"  类数            : {stats['classes']}")


def check_chroma(chroma_path: Path, collection_name: str) -> VectorStore | None:
    print_header("检查 Chroma 向量索引 (chroma_db)")

    if not chroma_path.exists():
        print(f"❌ 未找到 Chroma 目录: {chroma_path}")
        print("   请先运行 build_index.py 生成索引。")
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ 未设置 OPENAI_API_KEY 环境变量，无法进行向量检索。")
        return None

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    vs = VectorStore(
        chroma_path=chroma_path,
        collection_name=collection_name,
        openai_api_key=api_key,
        openai_base_url=base_url,
    )

    stats = vs.get_statistics()
    print("📊 Chroma 统计信息:")
    print(f"  集合名称  : {stats['collection_name']}")
    print(f"  向量总数  : {stats['total_vectors']}")

    return vs


def run_sample_query(
    vs: VectorStore,
    db_path: Path,
    query: str,
    top_k: int,
    include_tests: bool,
) -> None:
    print_header("运行语义检索 Query")
    print(f"🔍 Query: {query}")
    print(f"   top_k       = {top_k}")
    print(f"   include_tests = {include_tests}")

    results = vs.search(
        query=query,
        top_k=top_k,
        include_tests=include_tests,
    )

    if not results:
        print("⚠️ 未检索到结果，请尝试换一个 query 或确认索引是否正常。")
        return

    print(f"\n共返回 {len(results)} 条结果（按相似度排序）：")

    # 同时打开 SQLite，用 sqlite_id 回查方法体，验证“指针”是否正常
    with SQLiteStorage(db_path) as storage:
        for i, r in enumerate(results, 1):
            print(f"\n—— 结果 {i} ——")
            print(f"score      : {r['score']:.4f}")
            print(f"module     : {r['module_name']}")
            print(f"class      : {r['class_name']}")
            print(f"method     : {r['method_name']}")
            print(f"file_path  : {r['file_path']}")
            print(f"sqlite_id  : {r['sqlite_id']}")

            # 通过 sqlite_id 回到 SQLite，检查方法体是否存在
            if r["sqlite_id"]:
                m = storage.get_method_by_id(r["sqlite_id"])
                if m:
                    body_preview = (m["body"] or "").strip().splitlines()
                    if body_preview:
                        # 只展示前几行，防止代码太长
                        body_preview = body_preview[:]
                        print("body:")
                        for line in body_preview:
                            print("    " + line)
                    else:
                        print("body       : <empty>")
                else:
                    print("body       : ⚠️ 在 SQLite 中未找到该 sqlite_id 对应的方法")

            print("content(用于向量化的文本片段):")
            for line in r["content"].splitlines():
                print("    " + line)


def run_field_query(
    vs: VectorStore,
    db_path: Path,
    query: str,
    top_k: int,
    include_tests: bool,
) -> None:
    print_header("运行字段语义检索 Query")
    print(f"🔍 Query: {query}")
    print(f"   top_k       = {top_k}")
    print(f"   include_tests = {include_tests}")

    results = vs.search(
        query=query,
        top_k=top_k,
        include_tests=include_tests,
    )

    if not results:
        print("⚠️ 未检索到结果，请尝试换一个 query 或确认索引是否正常。")
        return

    print(f"\n共返回 {len(results)} 条结果（按相似度排序）：")

    # 同时打开 SQLite，用 sqlite_id 回查字段详情
    with SQLiteStorage(db_path) as storage:
        for i, r in enumerate(results, 1):
            print(f"\n—— 结果 {i} ——")
            print(f"score      : {r['score']:.4f}")
            print(f"module     : {r['module_name']}")
            print(f"class      : {r['class_name']}")
            print(f"field_name : {r.get('field_name', 'N/A')}")
            print(f"field_type : {r.get('field_type', 'N/A')}")
            print(f"file_path  : {r['file_path']}")
            print(f"sqlite_id  : {r['sqlite_id']}")

            # 通过 sqlite_id 回到 SQLite，检查字段详情
            if r["sqlite_id"]:
                f = storage.get_field_by_id(r["sqlite_id"])
                if f:
                    print(f"modifiers  : {f.get('modifiers', '')}")
                    print(f"initializer: {f.get('initializer', '<none>')}")
                    if f.get("javadoc"):
                        print("javadoc:")
                        for line in f["javadoc"].splitlines():
                            print("    " + line)
                else:
                    print("⚠️ 在 SQLite 中未找到该 sqlite_id 对应的字段")

            print("content(用于向量化的文本片段):")
            for line in r["content"].splitlines():
                print("    " + line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test script for querying the built Java code index.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="user authentication and validation",
        help="Semantic search query text.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return.",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include test methods/fields in search results (default: only production).",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["methods", "fields", "both"],
        default="methods",
        help="Search target: 'methods' (default), 'fields', or 'both'.",
    )
    return parser.parse_args()
    

def main() -> None:
    args = parse_args()

    project_root = Path.cwd()
    index_data_dir = project_root / "index_data"
    db_path = index_data_dir / "assets.db"
    chroma_path = index_data_dir / "chroma_db"

    print_header("AutoFix-RAG Index 快速自检")
    print(f"项目根目录     : {project_root}")
    print(f"SQLite 路径    : {db_path}")
    print(f"Chroma 路径    : {chroma_path}")
    print(f"查询目标       : {args.target}")

    # 1) 检查 SQLite
    check_sqlite(db_path)

    # 2) 根据 target 参数查询不同的内容
    if args.target == "methods":
        # 检查方法的 Chroma collection
        vs = check_chroma(chroma_path, collection_name="java_methods")
        if vs is None:
            return
        
        # 运行方法查询
        run_sample_query(
            vs=vs,
            db_path=db_path,
            query=args.query,
            top_k=args.top_k,
            include_tests=args.include_tests,
        )
    
    elif args.target == "fields":
        # 检查字段的 Chroma collection
        vs = check_chroma(chroma_path, collection_name="java_fields")
        if vs is None:
            return
        
        # 运行字段查询
        run_field_query(
            vs=vs,
            db_path=db_path,
            query=args.query,
            top_k=args.top_k,
            include_tests=args.include_tests,
        )
    
    elif args.target == "both":
        # 同时查询方法和字段
        print("\n" + "=" * 70)
        print("【查询方法 (Methods)】")
        print("=" * 70)
        vs_methods = check_chroma(chroma_path, collection_name="java_methods")
        if vs_methods:
            run_sample_query(
                vs=vs_methods,
                db_path=db_path,
                query=args.query,
                top_k=args.top_k,
                include_tests=args.include_tests,
            )
        
        print("\n" + "=" * 70)
        print("【查询字段 (Fields)】")
        print("=" * 70)
        vs_fields = check_chroma(chroma_path, collection_name="java_fields")
        if vs_fields:
            run_field_query(
                vs=vs_fields,
                db_path=db_path,
                query=args.query,
                top_k=args.top_k,
                include_tests=args.include_tests,
            )


if __name__ == "__main__":
    main()


# python test_query.py --query "eu.verdelhan.ta4j.Tick how to get begin time of the state" 
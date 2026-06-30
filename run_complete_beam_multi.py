"""
Complete BEAM Framework Example with Real Java Test Execution

This example demonstrates the full BEAM framework workflow using real
JaCoCo and PITest execution instead of mock execution.
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import glob
import json
from datetime import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from orchestrator import Orchestrator
from java_test_executor import JavaTestExecutor, ensure_repository, detect_java_version
from dataset_loader import DatasetLoader
from config import config
from utils import logger, get_sample_logger
from models import TestResultInfo, TestResultStatus
import subprocess


def get_file_from_commit(repo_path: Path, commit: str, file_path: str) -> str:
    """
    Get file content from a specific commit.
    
    Args:
        repo_path: Path to the repository
        commit: Commit hash
        file_path: Relative path to the file in the repository
        
    Returns:
        File content as string
    """
    try:
        result = subprocess.run(
            ['git', 'show', f'{commit}:{file_path}'],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get file {file_path} from commit {commit}: {e.stderr}")
        raise


def run_beam_with_real_execution(sample_index: int = 0, dataset_path: str = None, maven_repo_dir: str = None):
    """
    Run BEAM framework with real test execution.
    
    Args:
        sample_index: Index of the sample in the dataset to process
        dataset_path: Path to the dataset JSON file (optional)
        maven_repo_dir: Path to Maven local repository (optional, None uses default ~/.m2/repository)
    """
    # 1. Load dataset
    if dataset_path is None:
        dataset_path = "/data/david/project/mumutestup/dataset/dromara/hutool/data.json"
    
    logger.info(f"Loading dataset from: {dataset_path}")
    
    loader = DatasetLoader(dataset_path)
    sample = loader.get_sample(sample_index)
    beam_input = loader.prepare_for_beam(sample)
    
    logger.info(f"Processing sample {sample_index}: {beam_input['test_name']}")
    logger.info(f"Project: {beam_input['project']}")
    logger.info(f"Focal method changed: {beam_input['focal_method_changed']}")
    
    # 2. Initialize sample logger for this sample (using sample ID for naming)
    sample_logger = get_sample_logger()
    # 获取样本ID: 优先使用数据集中的ID字段,否则使用索引
    # 数据集ID格式: "dromara/hutool:13", 提取后为 "13"
    sample_id = sample.get('ID', f"{beam_input['project']}:{sample_index}")
    sample_logger.initialize(
        dataset_path=dataset_path,
        sample_name=beam_input['test_name'],
        sample_index=sample_index,
        sample_id=sample_id  # 使用样本ID命名日志，如 "dromara/hutool:13"
    )
    
    logger.info(f"Sample log file: {sample_logger.log_file}")
    
    # Log dataset sample information
    sample_logger.log_header("DATASET SAMPLE INFORMATION")
    sample_logger.log_info(f"Dataset: {dataset_path}")
    sample_logger.log_info(f"Sample ID: {sample_id}")
    sample_logger.log_info(f"Sample Index: {sample_index}")
    sample_logger.log_info(f"Test Name: {beam_input['test_name']}")
    sample_logger.log_info(f"Project: {beam_input['project']}")
    sample_logger.log_info(f"Focal Method Changed: {beam_input['focal_method_changed']}")
    
    # Log original and expected test
    sample_logger.log_subheader("ORIGINAL TEST CODE (Before Change)")
    if sample_logger.logger:
        for i, line in enumerate(beam_input['original_test'].split('\n'), 1):
            sample_logger.logger.info(f"  {i:4d} | {line}")
    
    sample_logger.log_subheader("EXPECTED TEST CODE (Ground Truth)")
    if sample_logger.logger:
        for i, line in enumerate(beam_input['expected_test'].split('\n'), 1):
            sample_logger.logger.info(f"  {i:4d} | {line}")
    
    # Log diff hunks if any
    if beam_input['diff_hunks']:
        sample_logger.log_subheader("DIFF HUNKS")
        for i, hunk in enumerate(beam_input['diff_hunks']):
            sample_logger.log_info(f"Hunk {i + 1}: {hunk.file_path}")
            sample_logger.log_info(f"  Type: {hunk.hunk_type or 'unknown'}")
            sample_logger.log_info(f"  Frequency: {hunk.frequency}")
            sample_logger.log_info(f"  Context:\n{hunk.context}")
    
    sample_logger.log_separator()
    
    # 3. Ensure repository exists
    # 从dataset路径中提取项目名称
    project_name = get_project_name_from_path(dataset_path)
    logger.info(f"Ensuring repository: {project_name}")
    
    try:
        repo_path = ensure_repository(project_name)
        logger.info(f"Repository ready at: {repo_path}")
        sample_logger.log_info(f"Repository: {repo_path}")
    except Exception as e:
        logger.error(f"Failed to clone repository: {e}")
        logger.error("Please check your GitHub token in config.py")
        sample_logger.log_error(f"Failed to clone repository: {e}")
        sample_logger.close()
        return None
    
    # 4. Checkout to aCommit (the fixed version)
    aCommit = sample['aCommit']
    bCommit = sample['bCommit']
    logger.info(f"Checking out to aCommit: {aCommit}")
    sample_logger.log_info(f"Checking out to aCommit: {aCommit}")
    try:
        result = subprocess.run(
            ['git', 'checkout', '-f', aCommit],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Successfully checked out to {aCommit}")
        sample_logger.log_info(f"Successfully checked out to {aCommit}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to checkout to {aCommit}: {e.stderr}")
        sample_logger.log_error(f"Failed to checkout to {aCommit}: {e.stderr}")
        sample_logger.close()
        return None
    
    # 4.5. Keep the aCommit test file (don't replace with bCommit version)
    # The aCommit test file already has the correct exception signatures
    # We will only update the specific test method content
    aPath = sample['aPath']
    test_file = repo_path / aPath
    
    logger.info(f"Using test file from aCommit: {aPath}")
    sample_logger.log_info(f"Using test file from aCommit: {aPath}")
    
    # Backup original aPath file for restoration after execution
    original_aPath_content = None
    try:
        if test_file.exists():
            original_aPath_content = test_file.read_text(encoding='utf-8')
            logger.info(f"Original {aPath} backed up in memory")
            sample_logger.log_info(f"Original {aPath} backed up in memory")
        else:
            logger.error(f"{aPath} does not exist in aCommit")
            sample_logger.log_error(f"{aPath} does not exist in aCommit")
            sample_logger.close()
            return None
    except Exception as e:
        logger.error(f"Failed to backup original file: {e}")
        sample_logger.log_error(f"Failed to backup original file: {e}")
        sample_logger.close()
        return None
    
    # 5. Detect Java version
    java_version = detect_java_version(repo_path)
    logger.info(f"Detected Java version: {java_version}")
    sample_logger.log_info(f"Java Version: {java_version}")
    
    # 6. Create executor
    try:
        java_executor = JavaTestExecutor(
            str(repo_path), 
            java_version=java_version,
            maven_repo_dir=maven_repo_dir,
            project_name=project_name,  # Use extracted project name
            sample_id=sample_id  # Pass sample_id for organizing reports
        )
    except ValueError as e:
        logger.error(f"Failed to create executor: {e}")
        logger.error("Please update Java paths in config.py")
        sample_logger.log_error(f"Failed to create executor: {e}")
        sample_logger.close()
        return None
    
    # 7. Extract test information
    test_name = beam_input['test_name']
    parts = test_name.replace('()', '').split('.')
    test_method = parts[-1]
    test_class = '.'.join(parts[:-1])
    test_rel_path = sample['aPath']  # Use aPath since we're on aCommit
    
    logger.info(f"Test class: {test_class}")
    logger.info(f"Test method: {test_method}")
    logger.info(f"Test file (aPath): {test_rel_path}")
    
    sample_logger.log_info(f"Test Class: {test_class}")
    sample_logger.log_info(f"Test Method: {test_method}")
    sample_logger.log_info(f"Test File (aPath): {test_rel_path}")
    sample_logger.log_separator()
    
    # 8. Create test executor function for Orchestrator
    # Use a mutable object to track current iteration
    iteration_tracker = {'current': 0}
    
    def test_executor(test_code: str) -> TestResultInfo:
        """
        Test executor function that wraps JavaTestExecutor for use with Orchestrator.
        
        Args:
            test_code: The test code to execute (will replace the test method in aCommit)
            
        Returns:
            TestResultInfo with execution results
        """
        # Increment iteration counter
        iteration_tracker['current'] += 1
        
        # Get new_imports from test_case (set by orchestrator)
        new_imports = getattr(beam_input['test_case'], 'new_imports', [])
        
        # Get test_imports from beam_input (from input['test_import'])
        # NOTE: test_imports are now ignored in execute_test to avoid outdated imports
        # Only new_imports will be merged with aCommit file's imports
        test_imports = beam_input.get('test_imports', [])
        
        # Get class_fields and non_test_methods from test_case
        class_fields = beam_input['test_case'].class_fields
        non_test_methods = beam_input['test_case'].non_test_methods
        
        # Get original test code for simple string replacement (from bCommit test file in aCommit)
        # This is more reliable than regex matching for complex nested code
        original_test_code = beam_input['test_case'].original_code if hasattr(beam_input['test_case'], 'original_code') else None
        
        return java_executor.execute_test(
            test_code=test_code,
            test_class=test_class,
            test_method=test_method,
            focal_method_info=beam_input['focal_method_info'],
            test_rel_path=test_rel_path,
            iteration=iteration_tracker['current'],
            new_imports=new_imports,
            test_imports=test_imports,
            class_fields=class_fields,
            non_test_methods=non_test_methods,
            original_test_code=original_test_code
        )
    
    # 9. Create Orchestrator and run BEAM framework
    logger.info("=" * 80)
    logger.info("STARTING BEAM FRAMEWORK WITH REAL EXECUTION")
    logger.info("=" * 80)
    
    # Initialize orchestrator with repository information for retrieval system
    orchestrator = Orchestrator(
        repo_path=str(repo_path),
        project_name=project_name,
        commit_id=aCommit  # Use aCommit (the fixed version) for retrieval
    )
    
    try:
        result = orchestrator.run(
            test_case=beam_input['test_case'],
            focal_method_info=beam_input['focal_method_info'],
            test_executor=test_executor,
            max_iterations=config.framework.max_iterations,
            diff_hunks=beam_input['diff_hunks'],
            prioritized_hunks=beam_input.get('prioritized_hunks', None)
        )
        
        # 9. Display results
        logger.info("=" * 80)
        logger.info("BEAM FRAMEWORK RESULTS")
        logger.info("=" * 80)
        logger.info(f"Best iteration: {result.iteration}")
        logger.info(f"Final status: {result.test_result.status.name}")
        logger.info(f"Coverage: {result.test_result.test_case.coverage_info.coverage_percentage:.2f}%")
        logger.info(f"Mutation kill rate: {result.test_result.test_case.mutation_info.kill_percentage:.2f}%")
        
        # 10. Compare with expected result
        logger.info("\n" + "=" * 80)
        logger.info("COMPARISON WITH GROUND TRUTH")
        logger.info("=" * 80)
        logger.info("Original test (before):")
        logger.info(beam_input['original_test'][:200] + "...")
        logger.info("\nExpected test (after - ground truth):")
        logger.info(beam_input['expected_test'][:200] + "...")
        logger.info("\nBEAM generated test:")
        logger.info(result.updated_test_code[:200] + "..." if result.updated_test_code else "No code generated")
        
        # Log comparison to sample logger
        sample_logger.log_header("COMPARISON WITH GROUND TRUTH")
        sample_logger.log_subheader("BEAM Generated Test")
        if result.updated_test_code and sample_logger.logger:
            for i, line in enumerate(result.updated_test_code.split('\n'), 1):
                sample_logger.logger.info(f"  {i:4d} | {line}")
        
        sample_logger.log_info(f"Log files saved to: {sample_logger.log_dir}")
        
    except Exception as e:
        logger.error(f"BEAM framework execution failed: {e}")
        import traceback
        traceback.print_exc()
        sample_logger.log_error(f"Framework execution failed: {e}")
        sample_logger.log_error(traceback.format_exc())
        result = None
    
    finally:
        # Always restore original aPath file after execution
        logger.info("\n" + "=" * 80)
        logger.info("RESTORING ORIGINAL TEST FILE")
        logger.info("=" * 80)
        sample_logger.log_header("RESTORING ORIGINAL TEST FILE")
        
        if original_aPath_content is not None:
            try:
                test_file.write_text(original_aPath_content, encoding='utf-8')
                logger.info(f"Successfully restored original {aPath}")
                sample_logger.log_info(f"Successfully restored original {aPath}")
            except Exception as e:
                logger.error(f"Failed to restore original file: {e}")
                sample_logger.log_error(f"Failed to restore original file: {e}")
        else:
            logger.warning("No original content to restore")
            sample_logger.log_warning("No original content to restore")
        
        sample_logger.close()
    
    return result


def find_all_data_json_files(dataset_root: str = "/data/david/project/mumutestup/dataset") -> List[str]:
    """
    查找dataset目录下所有的data.json文件。
    
    Args:
        dataset_root: dataset根目录路径
        
    Returns:
        所有data.json文件的路径列表
    """
    dataset_path = Path(dataset_root)
    data_json_files = []
    
    # 递归查找所有data.json文件
    for data_json in dataset_path.rglob("data.json"):
        data_json_files.append(str(data_json))
    
    return sorted(data_json_files)


def get_project_name_from_path(dataset_path: str) -> str:
    """
    从dataset路径中提取项目名称。
    例如: /data/david/project/mumutestup/dataset/dromara/hutool/data.json -> dromara/hutool
    
    Args:
        dataset_path: data.json文件的路径
        
    Returns:
        项目名称 (格式: org/repo)
    """
    path = Path(dataset_path)
    # 获取data.json的父目录(repo)和祖父目录(org)
    repo = path.parent.name
    org = path.parent.parent.name
    return f"{org}/{repo}"


def process_single_project(project_name: str, dataset_root: str = "/data/david/project/mumutestup/dataset", 
                          maven_repo_dir: str = None, start_index: int = 0, count: Optional[int] = None,
                          run_ablation: bool = False) -> Dict[str, Any]:
    """
    处理指定项目的所有样本或指定范围的样本。
    
    Args:
        project_name: 项目名称，格式为 "org/repo"，例如 "dromara/hutool"
        dataset_root: dataset根目录路径
        maven_repo_dir: Maven本地仓库路径（可选）
        start_index: 起始样本索引
        count: 处理的样本数量（None表示处理所有样本）
        run_ablation: 是否运行消融实验
        
    Returns:
        Dict包含处理结果统计信息
    """
    # 构建data.json路径
    dataset_path = Path(dataset_root) / project_name.replace("/", "/") / "data.json"
    
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return
    
    logger.info("=" * 80)
    logger.info(f"PROCESSING PROJECT: {project_name}")
    logger.info(f"Dataset: {dataset_path}")
    logger.info("=" * 80)
    
    # 加载数据集以确定样本总数
    loader = DatasetLoader(str(dataset_path))
    total_samples = len(loader)
    
    # 确定要处理的样本范围
    if count is None:
        count = total_samples - start_index
    
    end_index = min(start_index + count, total_samples)
    
    logger.info(f"Total samples in dataset: {total_samples}")
    logger.info(f"Processing samples: {start_index} to {end_index - 1} (total: {end_index - start_index})")
    logger.info("=" * 80)
    
    results = []
    successes = 0
    failures = 0
    
    for i in range(start_index, end_index):
        logger.info(f"\n\n{'=' * 80}")
        logger.info(f"PROCESSING PROJECT: {project_name}, SAMPLE {i}/{total_samples - 1}")
        logger.info(f"{'=' * 80}\n")
        
        try:
            if run_ablation:
                # Run with all ablation configurations
                ablation_results = run_with_ablation_study(run_beam_with_real_execution, i, str(dataset_path), maven_repo_dir)
                # Check if any configuration succeeded
                if any(r is not None for r in ablation_results.values()):
                    results.append((i, ablation_results))
                    successes += 1
                else:
                    failures += 1
            else:
                result = run_beam_with_real_execution(i, str(dataset_path), maven_repo_dir)
                if result:
                    results.append((i, result))
                    successes += 1
                else:
                    failures += 1
        except Exception as e:
            logger.error(f"Failed to process sample {i}: {e}")
            import traceback
            traceback.print_exc()
            failures += 1
    
    # 汇总
    logger.info("\n" + "=" * 80)
    logger.info(f"PROJECT {project_name} PROCESSING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total processed: {end_index - start_index}")
    logger.info(f"Successes: {successes}")
    logger.info(f"Failures: {failures}")
    
    if results:
        logger.info("\nSuccessful samples:")
        for idx, result in results:
            logger.info(f"  Sample {idx}:")
            if run_ablation and isinstance(result, dict):
                # Ablation results
                for config_name, config_result in result.items():
                    if config_result:
                        logger.info(f"    Config {config_name}:")
                        logger.info(f"      Best iteration: {config_result.iteration}")
                        logger.info(f"      Coverage: {config_result.test_result.test_case.coverage_info.coverage_percentage:.2f}%")
                        logger.info(f"      Mutation: {config_result.test_result.test_case.mutation_info.kill_percentage:.2f}%")
            else:
                # Normal result
                logger.info(f"    Best iteration: {result.iteration}")
                logger.info(f"    Coverage: {result.test_result.test_case.coverage_info.coverage_percentage:.2f}%")
                logger.info(f"    Mutation: {result.test_result.test_case.mutation_info.kill_percentage:.2f}%")
    
    # 返回统计信息
    return {
        "project_name": project_name,
        "total_processed": end_index - start_index,
        "successes": successes,
        "failures": failures,
        "results": results
    }


def _process_project_worker(args):
    """
    工作进程函数，用于并行处理单个项目。
    
    Args:
        args: 元组包含(project_name, dataset_root, maven_repo_dir, samples_per_project, run_ablation)
        
    Returns:
        Dict包含处理结果或异常信息
    """
    project_name, dataset_root, maven_repo_dir, samples_per_project, run_ablation = args
    
    try:
        # 获取dataset路径
        dataset_path = str(Path(dataset_root) / project_name.replace("/", "/") / "data.json")
        
        # 加载数据集以确定样本总数
        loader = DatasetLoader(dataset_path)
        total_samples = len(loader)
        
        # 确定要处理的样本数量
        count = samples_per_project if samples_per_project is not None else total_samples
        count = min(count, total_samples)
        
        logger.info(f"[{project_name}] Total samples: {total_samples}, will process: {count}")
        
        # 处理该项目的样本
        result = process_single_project(
            project_name=project_name,
            dataset_root=dataset_root,
            maven_repo_dir=maven_repo_dir,
            start_index=0,
            count=count,
            run_ablation=run_ablation
        )
        
        return {
            "success": True,
            "project_name": project_name,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"[{project_name}] Failed to process: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "project_name": project_name,
            "error": str(e)
        }


def process_all_projects(dataset_root: str = "/data/david/project/mumutestup/dataset", 
                        maven_repo_dir: str = None, samples_per_project: Optional[int] = None,
                        run_ablation: bool = False, parallel: bool = True, max_workers: Optional[int] = None):
    """
    遍历dataset下所有项目，为每个项目的所有样本进行测试用例更新。
    支持并行处理多个项目。
    
    Args:
        dataset_root: dataset根目录路径
        maven_repo_dir: Maven本地仓库路径（可选）
        samples_per_project: 每个项目处理的样本数量（None表示处理所有样本）
        run_ablation: 是否运行消融实验
        parallel: 是否并行处理项目（默认True）
        max_workers: 最大并行工作进程数（None表示使用CPU核心数）
    """
    # 获取CPU核心数
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    logger.info("=" * 80)
    logger.info("PROCESSING ALL PROJECTS IN DATASET")
    logger.info(f"Dataset root: {dataset_root}")
    logger.info(f"Parallel mode: {'ENABLED' if parallel else 'DISABLED'}")
    if parallel:
        logger.info(f"Max parallel workers: {max_workers} (CPU cores: {multiprocessing.cpu_count()})")
    logger.info("=" * 80)
    
    # 查找所有data.json文件
    data_json_files = find_all_data_json_files(dataset_root)
    
    if not data_json_files:
        logger.error(f"No data.json files found in {dataset_root}")
        return
    
    # 提取项目名称列表
    project_names = [get_project_name_from_path(path) for path in data_json_files]
    
    logger.info(f"Found {len(project_names)} projects:")
    for project_name in project_names:
        logger.info(f"  - {project_name}")
    
    logger.info("\n" + "=" * 80)
    
    # 统计信息
    total_projects = len(project_names)
    processed_projects = 0
    failed_projects = []
    project_results = {}
    
    if parallel:
        # 并行处理项目
        logger.info(f"Starting parallel processing with {max_workers} workers...")
        logger.info("=" * 80)
        
        # 准备参数列表
        worker_args = [
            (project_name, dataset_root, maven_repo_dir, samples_per_project, run_ablation)
            for project_name in project_names
        ]
        
        # 使用ProcessPoolExecutor进行并行处理
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_project = {
                executor.submit(_process_project_worker, args): args[0]
                for args in worker_args
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_project):
                project_name = future_to_project[future]
                try:
                    result = future.result()
                    
                    if result["success"]:
                        processed_projects += 1
                        project_results[project_name] = result["result"]
                        logger.info(f"✓ [{project_name}] Completed successfully")
                    else:
                        failed_projects.append(project_name)
                        logger.error(f"✗ [{project_name}] Failed: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    failed_projects.append(project_name)
                    logger.error(f"✗ [{project_name}] Exception in worker: {e}")
                    import traceback
                    traceback.print_exc()
    else:
        # 串行处理项目（原有逻辑）
        for idx, project_name in enumerate(project_names, 1):
            logger.info("\n\n" + "=" * 80)
            logger.info(f"PROJECT {idx}/{total_projects}: {project_name}")
            logger.info("=" * 80)
            
            try:
                dataset_path = str(Path(dataset_root) / project_name.replace("/", "/") / "data.json")
                
                # 加载数据集以确定样本总数
                loader = DatasetLoader(dataset_path)
                total_samples = len(loader)
                
                # 确定要处理的样本数量
                count = samples_per_project if samples_per_project is not None else total_samples
                count = min(count, total_samples)
                
                logger.info(f"Total samples in project: {total_samples}")
                logger.info(f"Will process: {count} samples")
                
                # 处理该项目的样本
                result = process_single_project(
                    project_name=project_name,
                    dataset_root=dataset_root,
                    maven_repo_dir=maven_repo_dir,
                    start_index=0,
                    count=count,
                    run_ablation=run_ablation
                )
                
                processed_projects += 1
                project_results[project_name] = result
                
            except Exception as e:
                logger.error(f"Failed to process project {project_name}: {e}")
                import traceback
                traceback.print_exc()
                failed_projects.append(project_name)
    
    # 最终汇总
    logger.info("\n\n" + "=" * 80)
    logger.info("ALL PROJECTS PROCESSING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total projects: {total_projects}")
    logger.info(f"Successfully processed: {processed_projects}")
    logger.info(f"Failed: {len(failed_projects)}")
    
    if failed_projects:
        logger.info("\nFailed projects:")
        for project in failed_projects:
            logger.info(f"  - {project}")
    
    if project_results:
        logger.info("\nProject-wise summary:")
        for project_name, result in project_results.items():
            logger.info(f"  {project_name}:")
            logger.info(f"    Total processed: {result['total_processed']}")
            logger.info(f"    Successes: {result['successes']}")
            logger.info(f"    Failures: {result['failures']}")


def batch_process_samples(start_index: int = 0, count: int = 5, dataset_path: str = None, maven_repo_dir: str = None):
    """
    Process multiple samples in batch.
    
    Args:
        start_index: Starting index in the dataset
        count: Number of samples to process
        dataset_path: Path to the dataset JSON file (optional)
        maven_repo_dir: Path to Maven local repository (optional)
    """
    logger.info("=" * 80)
    logger.info(f"BATCH PROCESSING: {count} samples starting from index {start_index}")
    logger.info("=" * 80)
    
    results = []
    successes = 0
    failures = 0
    
    for i in range(start_index, start_index + count):
        logger.info(f"\n\n{'=' * 80}")
        logger.info(f"PROCESSING SAMPLE {i}")
        logger.info(f"{'=' * 80}\n")
        
        try:
            result = run_beam_with_real_execution(i, dataset_path, maven_repo_dir)
            if result:
                results.append((i, result))
                successes += 1
            else:
                failures += 1
        except Exception as e:
            logger.error(f"Failed to process sample {i}: {e}")
            failures += 1
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total processed: {count}")
    logger.info(f"Successes: {successes}")
    logger.info(f"Failures: {failures}")
    
    if results:
        logger.info("\nSuccessful samples:")
        for idx, result in results:
            logger.info(f"  Sample {idx}:")
            logger.info(f"    Best iteration: {result.iteration}")
            logger.info(f"    Coverage: {result.test_result.test_case.coverage_info.coverage_percentage:.2f}%")
            logger.info(f"    Mutation: {result.test_result.test_case.mutation_info.kill_percentage:.2f}%")


def calculate_score(result) -> float:
    """
    Calculate score for a test result.
    
    Scoring rules (respects ablation settings):
    - Compilation failed: -1000
    - Compilation passed but test failed: -100
    - Test passed: 1000 + (line_cov + branch_cov + mutation) * 10
    
    Ablation rules:
    - all_ablation_disable_error: Only use compilation/execution status (1000 if pass)
    - all_ablation_disable_coverage: Don't include coverage in score
    - all_ablation_disable_mutation: Don't include mutation in score
    
    Args:
        result: Test result object
        
    Returns:
        Score as float
    """
    if result is None:
        return -1000.0
    
    # Check compilation status
    if hasattr(result, 'test_result'):
        test_result = result.test_result
        
        # Compilation failed
        if test_result.status == TestResultStatus.COMPILE_ERROR:
            return -1000.0
        
        # Test failed (runtime error or assertion failure)
        if test_result.status == TestResultStatus.RUN_FAIL:
            return -100.0
        
        # All other statuses (PASS, COVERAGE_LOSS, MUTATION_LOSS, etc.) 
        # are considered as "compilation and execution succeeded"
        score = 1000.0
        
        # If all_ablation_disable_error is True, don't use coverage or mutation in scoring
        if config.framework.all_ablation_disable_error:
            return score
        
        # Calculate score based on actual metrics (respecting ablation settings)
        coverage_info = test_result.test_case.coverage_info
        mutation_info = test_result.test_case.mutation_info
        
        # Add coverage to score (unless disabled)
        if not config.framework.all_ablation_disable_coverage:
            # Get line coverage
            line_cov = coverage_info.line_coverage_percentage if coverage_info else 0.0
            score += line_cov * 10.0
            
            # Get branch coverage (only if branches exist)
            if coverage_info and coverage_info.total_branches > 0:
                branch_cov = coverage_info.branch_coverage_percentage
                score += branch_cov * 10.0
        
        # Add mutation to score (unless disabled)
        if not config.framework.all_ablation_disable_mutation:
            mutation_score = mutation_info.kill_percentage if mutation_info else 0.0
            score += mutation_score * 10.0
        
        return score
    
    # Unknown status
    return -1000.0


def should_rerun_baseline(baseline_result, ablation_results: Dict[str, Any]) -> tuple[bool, list]:
    """
    Check if baseline should be rerun based on ablation results.
    
    Args:
        baseline_result: Result from baseline configuration
        ablation_results: Dictionary of results from other ablation configs
        
    Returns:
        Tuple of (should_rerun: bool, reasons: list of str)
    """
    if not config.framework.ablation_rerun_baseline:
        return False, []
    
    if baseline_result is None:
        return False, ["Baseline result is None"]
    
    reasons = []
    
    # Get baseline metrics
    baseline_score = calculate_score(baseline_result)
    baseline_compiled = False
    baseline_passed = False
    baseline_line_cov = 0.0
    baseline_branch_cov = 0.0
    baseline_mutation = 0.0
    
    if hasattr(baseline_result, 'test_result'):
        baseline_compiled = baseline_result.test_result.status != TestResultStatus.COMPILE_ERROR
        baseline_passed = baseline_result.test_result.status not in [TestResultStatus.COMPILE_ERROR, TestResultStatus.RUN_FAIL]
        
        if baseline_passed:
            cov_info = baseline_result.test_result.test_case.coverage_info
            mut_info = baseline_result.test_result.test_case.mutation_info
            
            baseline_line_cov = cov_info.line_coverage_percentage if cov_info else 0.0
            if cov_info and cov_info.total_branches > 0:
                baseline_branch_cov = cov_info.branch_coverage_percentage
            baseline_mutation = mut_info.kill_percentage if mut_info else 0.0
    
    # Check each ablation config (skip baseline itself)
    for config_name, result in ablation_results.items():
        if config_name == "baseline" or result is None:
            continue
        
        # Get ablation metrics
        ablation_score = calculate_score(result)
        ablation_compiled = False
        ablation_passed = False
        ablation_line_cov = 0.0
        ablation_branch_cov = 0.0
        ablation_mutation = 0.0
        
        if hasattr(result, 'test_result'):
            ablation_compiled = result.test_result.status != TestResultStatus.COMPILE_ERROR
            ablation_passed = result.test_result.status not in [TestResultStatus.COMPILE_ERROR, TestResultStatus.RUN_FAIL]
            
            if ablation_passed:
                cov_info = result.test_result.test_case.coverage_info
                mut_info = result.test_result.test_case.mutation_info
                
                ablation_line_cov = cov_info.line_coverage_percentage if cov_info else 0.0
                if cov_info and cov_info.total_branches > 0:
                    ablation_branch_cov = cov_info.branch_coverage_percentage
                ablation_mutation = mut_info.kill_percentage if mut_info else 0.0
        
        # Check conditions for rerun
        if not baseline_compiled and ablation_compiled:
            reasons.append(f"{config_name}: compiled (baseline didn't)")
        
        if not baseline_passed and ablation_passed:
            reasons.append(f"{config_name}: test passed (baseline didn't)")
        
        if baseline_passed and ablation_passed:
            if ablation_line_cov > baseline_line_cov:
                reasons.append(f"{config_name}: line coverage {ablation_line_cov:.2f}% > {baseline_line_cov:.2f}%")
            
            # Only compare branch coverage if both have branches
            if baseline_branch_cov > 0 or ablation_branch_cov > 0:
                if ablation_branch_cov > baseline_branch_cov:
                    reasons.append(f"{config_name}: branch coverage {ablation_branch_cov:.2f}% > {baseline_branch_cov:.2f}%")
            
            if ablation_mutation > baseline_mutation:
                reasons.append(f"{config_name}: mutation {ablation_mutation:.2f}% > {baseline_mutation:.2f}%")
    
    return len(reasons) > 0, reasons


def get_ablation_configs():
    """
    Get all ablation study configurations.
    
    Returns:
        List of tuples: (config_name, config_dict, description)
    """
    configs = [
        # # 1. Baseline - no ablation
        # (
        #     "baseline",
        #     {
        #         'ablation_disable_mutation': False,
        #         'ablation_disable_coverage': False,
        #         'ablation_disable_retrieval': False,
        #         'ablation_disable_error': False,
        #         'use_target_prioritized_changes': False,
        #         'all_ablation_disable_mutation': False,
        #         'all_ablation_disable_coverage': False,
        #         'all_ablation_disable_error': False
        #     },
        #     "Baseline (no ablation)"
        # ),
        # # 2. Use prioritized changes
        # (
        #     "w_prior",
        #     {
        #         'ablation_disable_mutation': False,
        #         'ablation_disable_coverage': False,
        #         'ablation_disable_retrieval': False,
        #         'ablation_disable_error': False,
        #         'use_target_prioritized_changes': True,
        #         'all_ablation_disable_mutation': False,
        #         'all_ablation_disable_coverage': False,
        #         'all_ablation_disable_error': False
        #     },
        #     "With prioritized changes"
        # ),
        
        # # 3. Completely disable mutation (no computation, no agent, no scoring, no threshold)
        # (
        #     "all_wo_mut",
        #     {
        #         'ablation_disable_mutation': False,
        #         'ablation_disable_coverage': False,
        #         'ablation_disable_retrieval': False,
        #         'ablation_disable_error': False,
        #         'use_target_prioritized_changes': True,
        #         'all_ablation_disable_mutation': True,
        #         'all_ablation_disable_coverage': False,
        #         'all_ablation_disable_error': False
        #     },
        #     "Completely without mutation (no computation)"
        # ),
        # # 4. Completely disable coverage (no computation, no agents, no scoring, no threshold)
        # (
        #     "all_wo_cov",
        #     {
        #         'ablation_disable_mutation': False,
        #         'ablation_disable_coverage': False,
        #         'ablation_disable_retrieval': False,
        #         'ablation_disable_error': False,
        #         'use_target_prioritized_changes': True,
        #         'all_ablation_disable_mutation': True,
        #         'all_ablation_disable_coverage': True,
        #         'all_ablation_disable_error': False
        #     },
        #     "Completely without coverage (no computation)"
        # ),
        # # 5. Completely disable retrieval (no SQLite, no ChromaDB)
        # (
        #     "all_wo_rag",
        #     {
        #         'ablation_disable_mutation': False,
        #         'ablation_disable_coverage': False,
        #         'ablation_disable_retrieval': True,
        #         'ablation_disable_error': False,
        #         'use_target_prioritized_changes': True,
        #         'all_ablation_disable_mutation': True,
        #         'all_ablation_disable_coverage': True,
        #         'all_ablation_disable_error': False,
        #         'ablation_sqlite_single_round': False
        #     },
        #     "Completely without retrieval"
        # ),
        # 5.5. SQLite single-round retrieval only (no iteration, no ChromaDB)
        # (
        #     "sqlite_1round",
        #     {
        #         'ablation_disable_mutation': False,
        #         'ablation_disable_coverage': False,
        #         'ablation_disable_retrieval': False,
        #         'ablation_disable_error': False,
        #         'use_target_prioritized_changes': True,
        #         'all_ablation_disable_mutation': True,
        #         'all_ablation_disable_coverage': True,
        #         'all_ablation_disable_error': False,
        #         'ablation_sqlite_single_round': True,
        #         'ablation_use_lsp': False
        #     },
        #     "SQLite single-round only (no iteration, no ChromaDB)"
        # ),
        (
            "all_wo_branch",
            {
                'ablation_disable_mutation': False,
                'ablation_disable_coverage': False,
                'ablation_disable_retrieval': False,
                'ablation_disable_error': False,
                'use_target_prioritized_changes': True,
                'all_ablation_disable_mutation': False,
                'all_ablation_disable_coverage': False,
                'all_ablation_disable_error': False,
                'ablation_sqlite_single_round': False,
                'ablation_sql_only_3rounds': False,
                'ablation_rag_only_3rounds': True,
                'ablation_sql_1round_rag_3rounds': False,
                'ablation_use_lsp': False,
                'ablation_coverage_lines_only': True
            },
            "Completely without branch coverage"
        ),
        (
            "all_wo_mutation",
            {
                'ablation_disable_mutation': False,
                'ablation_disable_coverage': False,
                'ablation_disable_retrieval': False,
                'ablation_disable_error': False,
                'use_target_prioritized_changes': True,
                'all_ablation_disable_mutation': True,
                'all_ablation_disable_coverage': False,
                'all_ablation_disable_error': False,
                'ablation_sqlite_single_round': False,
                'ablation_sql_only_3rounds': False,
                'ablation_rag_only_3rounds': True,
                'ablation_sql_1round_rag_3rounds': False,
                'ablation_use_lsp': False,
                'ablation_coverage_lines_only': False
            },
            "Completely without mutation"
        ),
        # (
        #     "all_wo_rag",
        #     {
        #         'ablation_disable_mutation': False,
        #         'ablation_disable_coverage': False,
        #         'ablation_disable_retrieval': False,
        #         'ablation_disable_error': False,
        #         'use_target_prioritized_changes': True,
        #         'all_ablation_disable_mutation': False,
        #         'all_ablation_disable_coverage': False,
        #         'all_ablation_disable_error': False,
        #         'ablation_sql_only_3rounds': True,
        #         'ablation_rag_only_3rounds': False,
        #         'ablation_sql_1round_rag_3rounds': False,
        #         'ablation_use_lsp': False,
        #         'ablation_coverage_lines_only': False
        #     },
        #     "Completely without rag retrieval"
        # ),
        # (
        #     "all_wo_sql",
        #     {
        #         'ablation_disable_mutation': False,
        #         'ablation_disable_coverage': False,
        #         'ablation_disable_retrieval': False,
        #         'ablation_disable_error': False,
        #         'use_target_prioritized_changes': True,
        #         'all_ablation_disable_mutation': False,
        #         'all_ablation_disable_coverage': False,
        #         'all_ablation_disable_error': False,
        #         'ablation_rag_only_3rounds': True,
        #         'ablation_sql_1round_rag_3rounds': False,
        #         'ablation_sql_only_3rounds': False,
        #         'ablation_use_lsp': False,
        #         'ablation_coverage_lines_only': False
        #     },
        #     "Completely without SQL retrieval"
        # ),
        # (
        #     "all_w_1sql_3rag",
        #     {
        #         'ablation_disable_mutation': False,
        #         'ablation_disable_coverage': False,
        #         'ablation_disable_retrieval': False,
        #         'ablation_disable_error': False,
        #         'use_target_prioritized_changes': True,
        #         'all_ablation_disable_mutation': False,
        #         'all_ablation_disable_coverage': False,
        #         'all_ablation_disable_error': False,
        #         'ablation_sql_1round_rag_3rounds': True,
        #         'ablation_rag_only_3rounds': False,
        #         'ablation_sql_only_3rounds': False,
        #         'ablation_use_lsp': False,
        #         'ablation_coverage_lines_only': False
        #     },
        #     "1 round SQL, then 3 rounds RAG retrieval"
        # ),
        # (
        #     "all_wo_mut_branch",
        #     {
        #         'ablation_disable_mutation': False,
        #         'ablation_disable_coverage': False,
        #         'ablation_disable_retrieval': False,
        #         'ablation_disable_error': False,
        #         'use_target_prioritized_changes': True,
        #         'all_ablation_disable_mutation': True,
        #         'all_ablation_disable_coverage': False,
        #         'all_ablation_disable_error': False,
        #         'ablation_sqlite_single_round': False,
        #         'ablation_use_lsp': False,
        #         'ablation_coverage_lines_only': True
        #     },
        #     "Completely without mutation and branch coverage"
        # ),
        # (
        #     "all_wo_mut_branch_use_lsp",
        #     {
        #         'ablation_disable_mutation': False,
        #         'ablation_disable_coverage': False,
        #         'ablation_disable_retrieval': False,
        #         'ablation_disable_error': False,
        #         'use_target_prioritized_changes': True,
        #         'all_ablation_disable_mutation': True,
        #         'all_ablation_disable_coverage': False,
        #         'all_ablation_disable_error': False,
        #         'ablation_sqlite_single_round': False,
        #         'ablation_use_lsp': True,
        #         'ablation_coverage_lines_only': True
        #     },
        #     "Completely without mutation and branch coverage and use LSP for retrieval"
        # ),
        # 5.6. Use LSP for retrieval (no SQLite, no ChromaDB)
        # (
        #     "use_lsp",
        #     {
        #         'ablation_disable_mutation': False,
        #         'ablation_disable_coverage': False,
        #         'ablation_disable_retrieval': False,
        #         'ablation_disable_error': False,
        #         'use_target_prioritized_changes': True,
        #         'all_ablation_disable_mutation': True,
        #         'all_ablation_disable_coverage': True,
        #         'all_ablation_disable_error': False,
        #         'ablation_sqlite_single_round': False,
        #         'ablation_use_lsp': True
        #     },
        #     "Use LSP (Language Server Protocol) for retrieval"
        # ),
        # # 5. Completely disable error feedback (no iteration, one-shot only)
        # (
        #     "all_wo_error",
        #     {
        #         'ablation_disable_mutation': False,
        #         'ablation_disable_coverage': False,
        #         'ablation_disable_retrieval': False,
        #         'ablation_disable_error': False,
        #         'use_target_prioritized_changes': True,
        #         'all_ablation_disable_mutation': False,
        #         'all_ablation_disable_coverage': False,
        #         'all_ablation_disable_error': True
        #     },
        #     "Completely without error feedback (one-shot)"
        # ),
        
        # # 3. Disable mutation
        # (
        #     "wo_mut",
        #     {
        #         'ablation_disable_mutation': True,
        #         'ablation_disable_coverage': False,
        #         'ablation_disable_retrieval': False,
        #         'ablation_disable_error': False,
        #         'use_target_prioritized_changes': False
        #     },
        #     "Without mutation analysis"
        # ),
        # # 4. Disable mutation + coverage
        # (
        #     "wo_mut_cov",
        #     {
        #         'ablation_disable_mutation': True,
        #         'ablation_disable_coverage': True,
        #         'ablation_disable_retrieval': False,
        #         'ablation_disable_error': False,
        #         'use_target_prioritized_changes': False
        #     },
        #     "Without mutation + coverage"
        # ),
        # # 5. Disable mutation + coverage + retrieval
        # (
        #     "wo_mut_cov_ret",
        #     {
        #         'ablation_disable_mutation': True,
        #         'ablation_disable_coverage': True,
        #         'ablation_disable_retrieval': True,
        #         'ablation_disable_error': False,
        #         'use_target_prioritized_changes': False
        #     },
        #     "Without mutation + coverage + retrieval"
        # ),
        # # 6. Disable all (mutation + coverage + retrieval + error)
        # (
        #     "wo_all",
        #     {
        #         'ablation_disable_mutation': True,
        #         'ablation_disable_coverage': True,
        #         'ablation_disable_retrieval': True,
        #         'ablation_disable_error': True,
        #         'use_target_prioritized_changes': False
        #     },
        #     "Without mutation + coverage + retrieval + error"
        # ),
    ]
    return configs


def set_ablation_config(config_dict):
    """
    Set ablation configuration in global config.
    
    Args:
        config_dict: Dictionary with ablation settings
    """
    for key, value in config_dict.items():
        setattr(config.framework, key, value)


def log_ablation_comparison(sample_id: str, extracted_results: Dict[str, Dict], 
                           baseline_rerun: bool = False, 
                           rerun_reasons: list = None,
                           baseline_first_score: float = None,
                           baseline_second_score: float = None,
                           project_name: str = None):
    """
    Log ablation study comparison results to a separate file.
    
    Args:
        sample_id: Sample identifier (cleaned for filename)
        extracted_results: Dictionary of extracted data from all configurations
        baseline_rerun: Whether baseline was rerun
        rerun_reasons: Reasons for rerunning baseline
        baseline_first_score: Score from first baseline run
        baseline_second_score: Score from second baseline run
        project_name: Project name in format "org/repo" (e.g., "dromara/hutool")
    """
    # Create ablation comparison log directory with project structure
    base_log_dir = Path("/data/david/project/mumutestup/ablation_comparison_logs")
    
    # If project_name provided, create org/repo subdirectories
    if project_name:
        log_dir = base_log_dir / project_name
    else:
        log_dir = base_log_dir
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file for this sample
    # sample_id should already be cleaned (no slashes or colons)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{sample_id}_{timestamp}.json"
    
    # Prepare comparison data
    comparison_data = {
        "sample_id": sample_id,
        "timestamp": timestamp,
        "baseline_rerun": baseline_rerun,
        "rerun_reasons": rerun_reasons if rerun_reasons else [],
        "baseline_first_score": baseline_first_score,
        "baseline_second_score": baseline_second_score,
        "configurations": {}
    }
    
    # Add extracted results for each configuration
    for config_name, config_data in extracted_results.items():
        comparison_data["configurations"][config_name] = config_data
    
    # Save to JSON file
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Ablation comparison log saved to: {log_file}")


def run_with_ablation_study(run_func, *args, **kwargs):
    """
    Run a function with all ablation configurations.
    Supports baseline rerun if any ablation outperforms baseline.
    
    Args:
        run_func: Function to run (e.g., run_beam_with_real_execution)
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        Dictionary mapping config names to results
    """
    ablation_configs = get_ablation_configs()
    results = {}
    
    logger.info("=" * 80)
    logger.info("ABLATION STUDY MODE")
    logger.info(f"Will run {len(ablation_configs)} configurations for each sample")
    if config.framework.ablation_rerun_baseline:
        logger.info("Baseline rerun: ENABLED (will rerun if ablation outperforms)")
    else:
        logger.info("Baseline rerun: DISABLED")
    logger.info("=" * 80)
    
    # First pass: run all configurations
    # Store extracted data immediately to avoid object mutation issues
    extracted_results = {}
    
    for idx, (config_name, config_dict, description) in enumerate(ablation_configs, 1):
        logger.info("\n" + "=" * 80)
        logger.info(f"ABLATION {idx}/{len(ablation_configs)}: {description}")
        logger.info(f"Configuration: {config_name}")
        logger.info("=" * 80)
        logger.info(f"Settings: {config_dict}")
        logger.info("=" * 80)
        
        # Set the ablation configuration
        set_ablation_config(config_dict)
        
        # Get the suffix for logging
        suffix = config.framework.get_ablation_suffix()
        logger.info(f"Logs will be saved to: {config.java.get_logs_dir(config.framework)}")
        logger.info(f"Reports will be saved to: {config.java.get_reports_dir(config.framework)}")
        logger.info("")
        
        try:
            # Run the function with this configuration
            result = run_func(*args, **kwargs)
            results[config_name] = result
            
            # Extract data immediately before it can be mutated
            if result and hasattr(result, 'test_result'):
                test_result = result.test_result
                extracted_data = {
                    "status": test_result.status.name if hasattr(test_result.status, 'name') else str(test_result.status),
                    "score": result.score if hasattr(result, 'score') else calculate_score(result),
                    "iteration": result.iteration if hasattr(result, 'iteration') else 0
                }
                
                # Extract metrics if available
                if test_result.status not in [TestResultStatus.COMPILE_ERROR, TestResultStatus.RUN_FAIL]:
                    cov_info = test_result.test_case.coverage_info
                    mut_info = test_result.test_case.mutation_info
                    
                    # Coverage information (detailed)
                    if cov_info:
                        extracted_data["coverage"] = {
                            "line_coverage_percentage": cov_info.line_coverage_percentage,
                            "branch_coverage_percentage": cov_info.branch_coverage_percentage if cov_info.total_branches > 0 else None,
                            "covered_lines_count": cov_info.covered_lines_count,
                            "total_lines": cov_info.total_lines,
                            "covered_branches_count": cov_info.covered_branches_count,
                            "total_branches": cov_info.total_branches,
                            "overall_coverage_percentage": cov_info.coverage_percentage
                        }
                    else:
                        extracted_data["coverage"] = None
                    
                    # Mutation information (detailed)
                    if mut_info:
                        mutation_data = {
                            "kill_percentage": mut_info.kill_percentage,
                            "total_mutations": mut_info.total_mutations,
                            "killed_mutations_count": len(mut_info.killed_mutations)
                        }
                        
                        # If detailed mutations available, add more info
                        if mut_info.detailed_mutations:
                            detailed = mut_info.detailed_mutations
                            mutation_data["killed_count"] = len(detailed.killed_mutations)
                            mutation_data["survived_count"] = len(detailed.survived_mutations)
                            mutation_data["no_coverage_count"] = len(detailed.no_coverage_mutations)
                            mutation_data["total_count"] = len(detailed.mutations)
                        
                        extracted_data["mutation"] = mutation_data
                    else:
                        extracted_data["mutation"] = None
                
                extracted_results[config_name] = extracted_data
                logger.info(f"✓ Ablation {idx}/{len(ablation_configs)} completed: {extracted_data['status']} (score: {extracted_data['score']:.2f})")
            else:
                score = calculate_score(result)
                extracted_results[config_name] = {
                    "status": "result_is_none" if result is None else "no_test_result",
                    "score": score
                }
                logger.info(f"✗ Ablation {idx}/{len(ablation_configs)} failed (score: {score:.2f})")
        except Exception as e:
            logger.error(f"✗ Ablation {idx}/{len(ablation_configs)} encountered error: {e}")
            import traceback
            traceback.print_exc()
            results[config_name] = None
            extracted_results[config_name] = {
                "status": "exception",
                "score": -1000.0,
                "error": str(e)
            }
    
    # Check if baseline should be rerun
    baseline_result = results.get("baseline")
    baseline_first_score = calculate_score(baseline_result)
    baseline_second_score = None
    baseline_rerun = False
    rerun_reasons = []
    
    # Only check for rerun if baseline exists
    if baseline_result is None:
        logger.info("\n" + "=" * 80)
        logger.info("BASELINE NOT IN CONFIGURATION - SKIPPING RERUN CHECK")
        logger.info("=" * 80)
        should_rerun = False
        reasons = []
    else:
        should_rerun, reasons = should_rerun_baseline(baseline_result, results)
    
    if should_rerun:
        baseline_rerun = True
        rerun_reasons = reasons
        
        logger.info("\n" + "=" * 80)
        logger.info("BASELINE RERUN TRIGGERED")
        logger.info("=" * 80)
        logger.info(f"Baseline first run score: {baseline_first_score:.2f}")
        logger.info("Reasons for rerun:")
        for reason in reasons:
            logger.info(f"  - {reason}")
        logger.info("=" * 80)
        
        # Find baseline config
        baseline_config = None
        for config_name, config_dict, description in ablation_configs:
            if config_name == "baseline":
                baseline_config = config_dict
                break
        
        if baseline_config:
            # Set baseline configuration
            set_ablation_config(baseline_config)
            
            logger.info("\nRerunning baseline configuration...")
            logger.info(f"Logs will be saved to: {config.java.get_logs_dir(config.framework)}")
            logger.info(f"Reports will be saved to: {config.java.get_reports_dir(config.framework)}")
            logger.info("")
            
            try:
                # Rerun baseline
                result_rerun = run_func(*args, **kwargs)
                baseline_second_score = calculate_score(result_rerun)
                
                logger.info(f"Baseline rerun score: {baseline_second_score:.2f}")
                
                # Choose the better result and extract data
                if baseline_second_score > baseline_first_score:
                    logger.info(f"✓ Using rerun result (better score: {baseline_second_score:.2f} > {baseline_first_score:.2f})")
                    results["baseline"] = result_rerun
                    
                    # Extract rerun data
                    if result_rerun and hasattr(result_rerun, 'test_result'):
                        test_result = result_rerun.test_result
                        extracted_data = {
                            "status": test_result.status.name if hasattr(test_result.status, 'name') else str(test_result.status),
                            "score": result_rerun.score if hasattr(result_rerun, 'score') else baseline_second_score,
                            "iteration": result_rerun.iteration if hasattr(result_rerun, 'iteration') else 0
                        }
                        
                        if test_result.status not in [TestResultStatus.COMPILE_ERROR, TestResultStatus.RUN_FAIL]:
                            cov_info = test_result.test_case.coverage_info
                            mut_info = test_result.test_case.mutation_info
                            
                            # Coverage information (detailed)
                            if cov_info:
                                extracted_data["coverage"] = {
                                    "line_coverage_percentage": cov_info.line_coverage_percentage,
                                    "branch_coverage_percentage": cov_info.branch_coverage_percentage if cov_info.total_branches > 0 else None,
                                    "covered_lines_count": cov_info.covered_lines_count,
                                    "total_lines": cov_info.total_lines,
                                    "covered_branches_count": cov_info.covered_branches_count,
                                    "total_branches": cov_info.total_branches,
                                    "overall_coverage_percentage": cov_info.coverage_percentage
                                }
                            else:
                                extracted_data["coverage"] = None
                            
                            # Mutation information (detailed)
                            if mut_info:
                                mutation_data = {
                                    "kill_percentage": mut_info.kill_percentage,
                                    "total_mutations": mut_info.total_mutations,
                                    "killed_mutations_count": len(mut_info.killed_mutations)
                                }
                                
                                # If detailed mutations available, add more info
                                if mut_info.detailed_mutations:
                                    detailed = mut_info.detailed_mutations
                                    mutation_data["killed_count"] = len(detailed.killed_mutations)
                                    mutation_data["survived_count"] = len(detailed.survived_mutations)
                                    mutation_data["no_coverage_count"] = len(detailed.no_coverage_mutations)
                                    mutation_data["total_count"] = len(detailed.mutations)
                                
                                extracted_data["mutation"] = mutation_data
                            else:
                                extracted_data["mutation"] = None
                        
                        extracted_results["baseline"] = extracted_data
                else:
                    logger.info(f"✓ Keeping original result (better score: {baseline_first_score:.2f} >= {baseline_second_score:.2f})")
                
            except Exception as e:
                logger.error(f"✗ Baseline rerun encountered error: {e}")
                import traceback
                traceback.print_exc()
                logger.info("Keeping original baseline result")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("ABLATION STUDY SUMMARY")
    logger.info("=" * 80)
    for idx, (config_name, _, description) in enumerate(ablation_configs, 1):
        result = results.get(config_name)
        score = calculate_score(result)
        status = "✓ Success" if result else "✗ Failed"
        
        rerun_marker = ""
        if config_name == "baseline" and baseline_rerun:
            if baseline_second_score and baseline_second_score > baseline_first_score:
                rerun_marker = f" [RERUN: {baseline_first_score:.2f} → {baseline_second_score:.2f}]"
            else:
                rerun_marker = f" [RERUN: kept original {baseline_first_score:.2f}]"
        
        logger.info(f"{idx}. {description:50s} {status:12s} Score: {score:8.2f}{rerun_marker}")
    logger.info("=" * 80)
    
    # Log comparison to separate file
    # Extract real sample_id from dataset
    sample_index = args[0] if len(args) > 0 else 0
    dataset_path = args[1] if len(args) > 1 else kwargs.get('dataset_path', None)
    
    # Default dataset path if not provided
    if dataset_path is None:
        dataset_path = "/data/david/project/mumutestup/dataset/dromara/hutool/data.json"
    
    # Load dataset to get real sample ID and project name
    try:
        from dataset_loader import DatasetLoader
        loader = DatasetLoader(dataset_path)
        sample = loader.get_sample(sample_index)
        beam_input = loader.prepare_for_beam(sample)
        
        # Get sample ID: prefer ID field from dataset, otherwise use project:index
        sample_id = sample.get('ID', f"{beam_input['project']}:{sample_index}")
        
        # Get project name (e.g., "dromara/hutool")
        project_name = beam_input['project']
        
        # Clean sample_id for filename (replace slashes and colons with underscores)
        sample_id_clean = sample_id.replace('/', '_').replace(':', '_')
    except Exception as e:
        logger.warning(f"Failed to load dataset for sample ID: {e}")
        # Fallback to simple naming
        sample_id_clean = f"sample_{sample_index}"
        project_name = "unknown/unknown"
    
    log_ablation_comparison(
        sample_id=sample_id_clean,
        extracted_results=extracted_results,
        baseline_rerun=baseline_rerun,
        rerun_reasons=rerun_reasons,
        baseline_first_score=baseline_first_score,
        baseline_second_score=baseline_second_score,
        project_name=project_name
    )
    
    return results


def main():
    """Main entry point."""
    logger.info("=" * 80)
    logger.info("BEAM Framework - Complete Example with Real Execution")
    logger.info("=" * 80)
    
    # Configuration check
    logger.info("\nConfiguration:")
    logger.info(f"  Maven: {config.java.maven_home}")
    logger.info(f"  Java versions: {list(config.java.java_homes.keys())}")
    logger.info(f"  Repos directory: {config.java.repos_dir}")
    logger.info(f"  GitHub tokens: {len(config.java.github_tokens)} configured")
    logger.info(f"  Test timeout: {config.java.test_timeout}s")
    logger.info(f"  Max iterations: {config.framework.max_iterations}")
    logger.info(f"  Coverage threshold: {config.framework.coverage_threshold * 100}%")
    logger.info(f"  Mutation threshold: {config.framework.mutation_threshold * 100}%")
    
    # Important notes
    logger.info("\n" + "=" * 80)
    logger.info("WORKFLOW:")
    logger.info("=" * 80)
    logger.info("1. Load test case from dataset")
    logger.info("2. Initialize sample logger for detailed logging")
    logger.info("3. Clone/verify repository")
    logger.info("4. Detect Java version")
    logger.info("5. Initialize JavaTestExecutor")
    logger.info("6. Run BEAM framework:")
    logger.info("   - Iteration 1: RootCauseAnalysis → TestUpdate → Execute")
    logger.info("   - Iterations 2+: Execute → Analyze → Coordinate → TestUpdate")
    logger.info("7. Each execution runs:")
    logger.info("   - Phase 1: JaCoCo (coverage + error detection)")
    logger.info("   - Phase 2: PITest (mutation testing, only if Phase 1 succeeds)")
    logger.info("8. Return best result across all iterations")
    logger.info("9. Detailed logs saved to: logs/<project>/<sample>.log")
    logger.info("=" * 80)
    
    # Ask about ablation study
    print("\n" + "=" * 80)
    print("ABLATION STUDY MODE")
    print("=" * 80)
    ablation_choice = input("Do you want to run ablation experiments? (yes/no) [no]: ").strip().lower()
    run_ablation = ablation_choice in ['yes', 'y']
    
    if run_ablation:
        configs = get_ablation_configs()
        print(f"\nAblation study will run {len(configs)} configurations:")
        for idx, (name, _, desc) in enumerate(configs, 1):
            print(f"  {idx}. {desc}")
        print("\nEach sample will be processed with ALL configurations before moving to the next sample.")
        print("Results will be saved to separate directories based on the configuration.")
        print("")
    
    print("\nChoose an option:")
    print("1. Process single sample (interactive)")
    print("2. Process sample 0 (default)")
    print("3. Batch process samples 0-4")
    print("4. Process with custom dataset path")
    print("5. Process with custom Maven repository")
    print("6. Process all samples in a specific project")
    print("7. Process all samples in ALL projects")
    print("8. Process all samples in multiple specific projects")
    
    choice = input("\nEnter choice (1-8) [2]: ").strip() or "2"
    
    maven_repo_dir = None
    
    if choice == "1":
        sample_index = int(input("Enter sample index: "))
        if run_ablation:
            run_with_ablation_study(run_beam_with_real_execution, sample_index)
        else:
            run_beam_with_real_execution(sample_index)
    elif choice == "2":
        if run_ablation:
            run_with_ablation_study(run_beam_with_real_execution, 0)
        else:
            run_beam_with_real_execution(0)
    elif choice == "3":
        if run_ablation:
            # For batch processing with ablation, process each sample with all configs
            for i in range(0, 5):
                logger.info(f"\n\n{'=' * 80}")
                logger.info(f"PROCESSING SAMPLE {i} WITH ALL ABLATION CONFIGURATIONS")
                logger.info(f"{'=' * 80}\n")
                run_with_ablation_study(run_beam_with_real_execution, i)
        else:
            batch_process_samples(0, 5)
    elif choice == "4":
        dataset_path = input("Enter dataset path: ").strip()
        sample_index = int(input("Enter sample index [0]: ").strip() or "0")
        if run_ablation:
            run_with_ablation_study(run_beam_with_real_execution, sample_index, dataset_path)
        else:
            run_beam_with_real_execution(sample_index, dataset_path)
    elif choice == "5":
        maven_repo_dir = input("Enter Maven repository path: ").strip()
        sample_index = int(input("Enter sample index [0]: ").strip() or "0")
        dataset_path = input("Enter dataset path (press Enter to use default): ").strip() or None
        if run_ablation:
            run_with_ablation_study(run_beam_with_real_execution, sample_index, dataset_path, maven_repo_dir)
        else:
            run_beam_with_real_execution(sample_index, dataset_path, maven_repo_dir)
    elif choice == "6":
        # 处理指定项目的所有样本
        print("\n可用的项目:")
        dataset_root = input("Enter dataset root path [/data/david/project/mumutestup/dataset]: ").strip() or "/data/david/project/mumutestup/dataset"
        data_json_files = find_all_data_json_files(dataset_root)
        
        if not data_json_files:
            logger.error(f"No projects found in {dataset_root}")
            sys.exit(1)
        
        for idx, path in enumerate(data_json_files, 1):
            project_name = get_project_name_from_path(path)
            print(f"  {idx}. {project_name}")
        
        project_choice = input(f"\nEnter project number (1-{len(data_json_files)}): ").strip()
        try:
            project_idx = int(project_choice) - 1
            if 0 <= project_idx < len(data_json_files):
                project_name = get_project_name_from_path(data_json_files[project_idx])
                
                # 询问处理范围
                start_idx = int(input("Enter start index [0]: ").strip() or "0")
                count_input = input("Enter number of samples to process (press Enter for all): ").strip()
                count = int(count_input) if count_input else None
                
                maven_repo = input("Enter Maven repository path (press Enter to use default): ").strip() or None
                
                process_single_project(
                    project_name=project_name,
                    dataset_root=dataset_root,
                    maven_repo_dir=maven_repo,
                    start_index=start_idx,
                    count=count,
                    run_ablation=run_ablation
                )
            else:
                logger.error("Invalid project number")
                sys.exit(1)
        except ValueError:
            logger.error("Invalid input")
            sys.exit(1)
    elif choice == "7":
        # 处理所有项目
        dataset_root = input("Enter dataset root path [/data/david/project/mumutestup/dataset]: ").strip() or "/data/david/project/mumutestup/dataset"
        samples_input = input("Enter max samples per project (press Enter for all samples): ").strip()
        samples_per_project = int(samples_input) if samples_input else None
        maven_repo = input("Enter Maven repository path (press Enter to use default): ").strip() or None
        
        # 询问是否并行处理
        parallel_input = input(f"Enable parallel processing? (yes/no) [yes]: ").strip().lower() or "yes"
        parallel = parallel_input in ['yes', 'y']
        
        max_workers = None
        if parallel:
            cpu_cores = multiprocessing.cpu_count()
            workers_input = input(f"Enter max parallel workers (press Enter for CPU cores={cpu_cores}): ").strip()
            max_workers = int(workers_input) if workers_input else None
        
        confirm = input(f"\nThis will process ALL projects in {dataset_root}. Continue? (yes/no): ").strip().lower()
        if confirm in ['yes', 'y']:
            process_all_projects(
                dataset_root=dataset_root,
                maven_repo_dir=maven_repo,
                samples_per_project=samples_per_project,
                run_ablation=run_ablation,
                parallel=parallel,
                max_workers=max_workers
            )
        else:
            logger.info("Operation cancelled by user")
    elif choice == "8":
        # 处理多个特定项目
        print("\n可用的项目:")
        dataset_root = input("Enter dataset root path [/data/david/project/mumutestup/dataset]: ").strip() or "/data/david/project/mumutestup/dataset"
        data_json_files = find_all_data_json_files(dataset_root)
        
        if not data_json_files:
            logger.error(f"No projects found in {dataset_root}")
            sys.exit(1)
        
        for idx, path in enumerate(data_json_files, 1):
            project_name = get_project_name_from_path(path)
            print(f"  {idx}. {project_name}")
        
        project_choice = input(f"\nEnter project numbers (e.g., 1,3,5 or 1 3 5) (1-{len(data_json_files)}): ").strip()
        try:
            # 解析输入，支持逗号或空格分隔
            project_indices = []
            for part in project_choice.replace(',', ' ').split():
                idx = int(part.strip()) - 1
                if 0 <= idx < len(data_json_files):
                    project_indices.append(idx)
                else:
                    logger.warning(f"Invalid project number: {part}, skipping")
            
            if not project_indices:
                logger.error("No valid project numbers provided")
                sys.exit(1)
            
            # 去重并排序
            project_indices = sorted(set(project_indices))
            selected_projects = [get_project_name_from_path(data_json_files[idx]) for idx in project_indices]
            
            logger.info(f"\nSelected {len(selected_projects)} project(s):")
            for project_name in selected_projects:
                logger.info(f"  - {project_name}")
            
            # 询问处理范围（对所有选中的项目使用相同的设置）
            start_idx = int(input("\nEnter start index [0]: ").strip() or "0")
            count_input = input("Enter number of samples to process per project (press Enter for all): ").strip()
            count = int(count_input) if count_input else None
            
            maven_repo = input("Enter Maven repository path (press Enter to use default): ").strip() or None
            
            # 询问是否并行处理
            parallel_input = input(f"Enable parallel processing? (yes/no) [yes]: ").strip().lower() or "yes"
            parallel = parallel_input in ['yes', 'y']
            
            max_workers = None
            if parallel:
                cpu_cores = multiprocessing.cpu_count()
                workers_input = input(f"Enter max parallel workers (press Enter for CPU cores={cpu_cores}): ").strip()
                max_workers = int(workers_input) if workers_input else None
            
            # 处理每个选中的项目
            total_projects = len(selected_projects)
            processed_projects = 0
            failed_projects = []
            project_results = {}
            
            if parallel:
                # 并行处理
                if max_workers is None:
                    max_workers = multiprocessing.cpu_count()
                
                logger.info("\n" + "=" * 80)
                logger.info(f"Starting parallel processing with {max_workers} workers...")
                logger.info("=" * 80)
                
                # 准备参数列表
                worker_args = [
                    (project_name, dataset_root, maven_repo, start_idx, count, run_ablation)
                    for project_name in selected_projects
                ]
                
                # 包装器函数用于process_single_project
                def _worker_wrapper(args):
                    pname, droot, mrepo, sidx, cnt, rabl = args
                    try:
                        result = process_single_project(
                            project_name=pname,
                            dataset_root=droot,
                            maven_repo_dir=mrepo,
                            start_index=sidx,
                            count=cnt,
                            run_ablation=rabl
                        )
                        return {"success": True, "project_name": pname, "result": result}
                    except Exception as e:
                        logger.error(f"[{pname}] Failed: {e}")
                        import traceback
                        traceback.print_exc()
                        return {"success": False, "project_name": pname, "error": str(e)}
                
                # 使用ProcessPoolExecutor进行并行处理
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future_to_project = {
                        executor.submit(_worker_wrapper, args): args[0]
                        for args in worker_args
                    }
                    
                    for future in as_completed(future_to_project):
                        project_name = future_to_project[future]
                        try:
                            result = future.result()
                            if result["success"]:
                                processed_projects += 1
                                project_results[project_name] = result["result"]
                                logger.info(f"✓ [{project_name}] Completed successfully")
                            else:
                                failed_projects.append(project_name)
                                logger.error(f"✗ [{project_name}] Failed: {result.get('error', 'Unknown error')}")
                        except Exception as e:
                            failed_projects.append(project_name)
                            logger.error(f"✗ [{project_name}] Exception: {e}")
            else:
                # 串行处理
                for idx, project_name in enumerate(selected_projects, 1):
                    logger.info("\n\n" + "=" * 80)
                    logger.info(f"PROJECT {idx}/{total_projects}: {project_name}")
                    logger.info("=" * 80)
                    
                    try:
                        result = process_single_project(
                            project_name=project_name,
                            dataset_root=dataset_root,
                            maven_repo_dir=maven_repo,
                            start_index=start_idx,
                            count=count,
                            run_ablation=run_ablation
                        )
                        processed_projects += 1
                        project_results[project_name] = result
                    except Exception as e:
                        logger.error(f"Failed to process project {project_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        failed_projects.append(project_name)
            
            # 汇总
            logger.info("\n\n" + "=" * 80)
            logger.info("MULTIPLE PROJECTS PROCESSING SUMMARY")
            logger.info("=" * 80)
            logger.info(f"Total projects: {total_projects}")
            logger.info(f"Successfully processed: {processed_projects}")
            logger.info(f"Failed: {len(failed_projects)}")
            
            if failed_projects:
                logger.info("\nFailed projects:")
                for project in failed_projects:
                    logger.info(f"  - {project}")
            
            if project_results:
                logger.info("\nProject-wise summary:")
                for project_name, result in project_results.items():
                    logger.info(f"  {project_name}:")
                    logger.info(f"    Total processed: {result['total_processed']}")
                    logger.info(f"    Successes: {result['successes']}")
                    logger.info(f"    Failures: {result['failures']}")
            
        except ValueError as e:
            logger.error(f"Invalid input: {e}")
            sys.exit(1)
    else:
        logger.error("Invalid choice")
        sys.exit(1)


if __name__ == '__main__':
    main()

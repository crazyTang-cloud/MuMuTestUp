"""
Dataset Loader for BEAM Framework

Converts data.json format to BEAM framework format
"""

import json
import re
from typing import List, Dict, Any, Tuple
from models import TestCase, FocalMethodInfo, DiffHunk, CoverageInfo, MutationInfo


class DatasetLoader:
    """加载和处理data.json数据集"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict]:
        """加载JSON数据"""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)
    
    def get_sample(self, index: int) -> Dict[str, Any]:
        """获取单个原始样本"""
        if index < 0 or index >= len(self.data):
            raise IndexError(f"Index {index} out of range [0, {len(self.data)})")
        return self.data[index]
    
    def prepare_for_beam(self, sample: Dict) -> Dict[str, Any]:
        """
        将数据集样本转换为BEAM框架输入
        
        Args:
            sample: 原始数据样本
            
        Returns:
            {
                'test_case': TestCase,
                'focal_method_info': FocalMethodInfo,
                'diff_hunks': List[DiffHunk],
                'focal_method_changed': bool,
                'original_test': str,  # bSource.code (变更前)
                'expected_test': str,  # aSource.code (变更后，ground truth)
                'test_name': str,
                'project': str,
                'test_imports': List[str]  # 测试类的import语句列表
            }
        """
        input_data = sample['input']
        
        # 6. 提取测试类的import语句
        # test_import 存储在 input 字段中
        test_imports = input_data.get('test_import', [])
        
        # 7. 提取测试类的类变量和非测试方法
        class_fields = input_data.get('class_fields', [])
        non_test_methods = input_data.get('non_test_methods', [])
        
        # 1. 转换test_case
        # 注意：这里使用input['test_code']（即bSource的代码）作为AI的输入
        # 这样AI可以看到原始的测试代码，并基于diff_hunks进行更新
        # 在实际执行时，会在aCommit的测试文件基础上进行替换
        original_test_code = self._clean_test_code(input_data['test_code'])
        
        test_case = TestCase(
            name=input_data['test_name'],
            code=original_test_code,
            focal_method=self._extract_focal_method_name(input_data['test_name']),
            # 初始化为空，需要通过test_executor获取实际数据
            coverage_info=CoverageInfo(),
            mutation_info=MutationInfo(),
            test_imports=test_imports,  # 设置测试类的import语句
            class_fields=class_fields,  # 设置测试类的类变量
            non_test_methods=non_test_methods,  # 设置测试类的非测试方法
            original_code=original_test_code  # bCommit版本的测试代码
        )
        
        # 2. 转换focal_method_info
        # 注意：input_data['focal_method'] 是 hunks 列表
        # 需要从顶层 sample['focal_method'] 获取方法信息
        focal_data = sample.get('focal_method', {})
        
        # 提取 focal method 代码和行号信息
        a_focal = focal_data.get('a_focal_method_code', {})
        b_focal = focal_data.get('b_focal_method_code', {})
        
        # 处理 a_focal_method_code 可能是字符串或字典的情况
        if isinstance(a_focal, dict):
            current_code = a_focal.get('code', '')
            start_line = a_focal.get('startLine', 0)
            end_line = a_focal.get('endLine', 0)
        else:
            current_code = a_focal
            start_line = 0
            end_line = 0
        
        if isinstance(b_focal, dict):
            original_code = b_focal.get('code', '')
        else:
            original_code = b_focal
        
        focal_method_info = FocalMethodInfo(
            name=focal_data.get('focal_method_name', self._extract_focal_method_name(input_data['test_name'])),
            current_code=current_code,
            original_code=original_code,
            changed_lines=[],  # 可以从diff计算，暂时为空
            start_line=start_line,
            end_line=end_line,
            class_name=focal_data.get('focal_class_name', ''),
            source_file_path=focal_data.get('a_focal_method_path', '')
        )
        
        # 3. 转换四种类型的hunks为DiffHunk列表
        diff_hunks = self._convert_typed_hunks(input_data)
        
        # 3.5 转换prioritized_changes (如果存在)
        prioritized_hunks = self._convert_prioritized_changes(input_data)
        
        # 4. 获取focal_method_changed标志
        # focal_method_changed 在 input['focal_method_info'] 中
        focal_method_info_dict = input_data.get('focal_method_info', {})
        focal_method_changed = focal_method_info_dict.get('focal_method_changed', False)
        
        # 5. 获取样本ID
        sample_id = sample.get('ID', f"unknown:{input_data['test_name']}")
        
        return {
            'test_case': test_case,
            'focal_method_info': focal_method_info,
            'diff_hunks': diff_hunks,
            'prioritized_hunks': prioritized_hunks,  # 预先优先级排序的hunks（消融实验用）
            'focal_method_changed': focal_method_changed,
            'original_test': sample['bSource']['code'],
            'expected_test': sample['aSource']['code'],
            'test_name': input_data['test_name'],
            'project': input_data.get('project', 'unknown'),
            'sample_id': sample_id,  # 样本唯一ID，如 "dromara/hutool:13"
            'test_imports': test_imports  # 测试类的import语句列表
        }
    
    def _clean_test_code(self, test_code: str) -> str:
        """
        清理测试代码，移除标记
        
        移除[BREAKAGE_START]和[BREAKAGE_END]标记
        """
        cleaned = test_code.replace('[BREAKAGE_START]', '')
        cleaned = cleaned.replace('[BREAKAGE_END]', '')
        return cleaned.strip()
    
    def _extract_focal_method_name(self, test_name: str) -> str:
        """
        从测试名称提取焦点方法名
        
        例如: "cn.hutool.db.sql.ConditionTest.parseTest()" -> "parse"
        
        策略:
        1. 取最后一个部分（方法名）
        2. 移除括号
        3. 移除Test后缀
        """
        # 取最后一个点后的部分
        parts = test_name.split('.')
        method_with_parens = parts[-1]  # "parseTest()"
        
        # 移除括号
        method = method_with_parens.replace('()', '').replace('(', '').replace(')', '')
        
        # 移除Test后缀
        if method.endswith('Test'):
            method = method[:-4]  # 移除最后4个字符 "Test"
        
        return method
    
    def _convert_typed_hunks(self, input_data: Dict) -> List[DiffHunk]:
        """
        转换四种类型的hunks为DiffHunk列表
        
        Args:
            input_data: 输入数据，包含:
                - test_method: 与测试方法调用相同方法的hunks
                - focal_method: 与focal method调用相同方法的hunks
                - focal_file: focal method所在文件的hunks
                - high_frequency: 高频出现的hunks
                - prioritized_changes (可选): 对比实验用的预先优先级排序的hunks
        
        Returns:
            DiffHunk列表，每个hunk都有hunk_type标记
        """
        diff_hunks = []
        
        # 处理test_method类型的hunks
        test_method_hunks = input_data.get('test_method', [])
        for i, hunk_data in enumerate(test_method_hunks):
            diff_hunk = self._create_diff_hunk(
                hunk_data, 
                hunk_type="test_method",
                hunk_id=f"test_method_{i}"
            )
            diff_hunks.append(diff_hunk)
        
        # 处理focal_method类型的hunks
        focal_method_hunks = input_data.get('focal_method', [])
        for i, hunk_data in enumerate(focal_method_hunks):
            diff_hunk = self._create_diff_hunk(
                hunk_data,
                hunk_type="focal_method",
                hunk_id=f"focal_method_{i}"
            )
            diff_hunks.append(diff_hunk)
        
        # 处理focal_file类型的hunks
        focal_file_hunks = input_data.get('focal_file', [])
        for i, hunk_data in enumerate(focal_file_hunks):
            diff_hunk = self._create_diff_hunk(
                hunk_data,
                hunk_type="focal_file",
                hunk_id=f"focal_file_{i}"
            )
            diff_hunks.append(diff_hunk)
        
        # 处理high_frequency类型的hunks
        high_frequency_hunks = input_data.get('high_frequency', [])
        for i, hunk_data in enumerate(high_frequency_hunks):
            diff_hunk = self._create_diff_hunk(
                hunk_data,
                hunk_type="high_frequency",
                hunk_id=f"high_frequency_{i}"
            )
            diff_hunks.append(diff_hunk)
        
        return diff_hunks
    
    def _convert_prioritized_changes(self, input_data: Dict) -> List[DiffHunk]:
        """
        转换prioritized_changes为DiffHunk列表（用于消融实验）
        
        prioritized_changes 格式:
        ["[<HUNK>][<DEL>]old code[<ADD>]new code[</HUNK>]", ...]
        
        需要去除 [<HUNK>] 和 [</HUNK>] 标记
        
        Args:
            input_data: 输入数据，包含prioritized_changes字段
        
        Returns:
            DiffHunk列表，每个hunk的hunk_type为"prioritized"
        """
        diff_hunks = []
        
        prioritized_changes = input_data.get('prioritized_changes', [])
        for i, hunk_str in enumerate(prioritized_changes):
            # 去除 [<HUNK>] 和 [</HUNK>] 标记
            cleaned_hunk = hunk_str.replace('[<HUNK>]', '').replace('[</HUNK>]', '').strip()
            
            # 创建DiffHunk对象
            diff_hunk = DiffHunk(
                hunk_id=f"prioritized_{i}",
                file_path="unknown",  # prioritized_changes 不包含文件路径信息
                old_lines=[],
                new_lines=[],
                context=cleaned_hunk,
                hunk_type="prioritized",
                frequency=len(prioritized_changes) - i  # 按顺序递减频率，保持优先级
            )
            diff_hunks.append(diff_hunk)
        
        return diff_hunks
    
    def _create_diff_hunk(self, hunk_data: Dict, hunk_type: str, hunk_id: str) -> DiffHunk:
        """
        从hunk数据创建DiffHunk对象
        
        Args:
            hunk_data: hunk数据，包含:
                - diff: 变更内容
                - file: 文件路径
                - frequency: 频率（可选，仅用于high_frequency类型）
            hunk_type: hunk类型
            hunk_id: hunk ID
        
        Returns:
            DiffHunk对象
        """
        # 获取frequency（如果有的话）
        frequency = hunk_data.get('frequency', 1)
        
        return DiffHunk(
            hunk_id=hunk_id,
            file_path=hunk_data.get('file', 'unknown'),
            old_lines=[],  # 可以从diff解析，暂时为空
            new_lines=[],  # 可以从diff解析，暂时为空
            context=hunk_data.get('diff', ''),
            hunk_type=hunk_type,
            frequency=frequency
        )
    
    def _convert_related_changes(self, changes: List[Dict]) -> List[DiffHunk]:
        """
        转换related_changes为DiffHunk列表（保留用于向后兼容）
        
        Args:
            changes: related_changes列表，每项包含:
                - diff: 变更内容
                - file: 文件路径
        
        Returns:
            DiffHunk列表
        """
        diff_hunks = []
        
        for i, change in enumerate(changes):
            diff_hunk = DiffHunk(
                hunk_id=f"related_change_{i}",
                file_path=change.get('file', 'unknown'),
                old_lines=[],  # 可以从diff解析，暂时为空
                new_lines=[],  # 可以从diff解析，暂时为空
                context=change.get('diff', ''),
                hunk_type=None,  # 旧格式没有类型
                frequency=1
            )
            diff_hunks.append(diff_hunk)
        
        return diff_hunks
    
    def batch_prepare(self, start_index: int = 0, 
                     batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        批量准备数据
        
        Args:
            start_index: 起始索引
            batch_size: 批次大小
            
        Returns:
            准备好的BEAM输入列表
        """
        end_index = min(start_index + batch_size, len(self.data))
        batch = []
        
        for i in range(start_index, end_index):
            sample = self.get_sample(i)
            beam_input = self.prepare_for_beam(sample)
            batch.append(beam_input)
        
        return batch
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据集统计信息
        
        Returns:
            统计信息字典
        """
        total = len(self.data)
        focal_changed_count = 0
        focal_unchanged_count = 0
        
        for sample in self.data:
            if sample['input']['focal_method']['focal_method_changed']:
                focal_changed_count += 1
            else:
                focal_unchanged_count += 1
        
        return {
            'total_samples': total,
            'focal_method_changed': focal_changed_count,
            'focal_method_unchanged': focal_unchanged_count,
            'percentage_changed': focal_changed_count / total * 100 if total > 0 else 0,
            'percentage_unchanged': focal_unchanged_count / total * 100 if total > 0 else 0
        }


# 使用示例
if __name__ == '__main__':
    # 加载数据集
    loader = DatasetLoader('D:/project/python/beam/dataset/data.json')
    
    # 打印统计信息
    stats = loader.get_statistics()
    print("数据集统计:")
    print(f"  总样本数: {stats['total_samples']}")
    print(f"  Focal method变更: {stats['focal_method_changed']} ({stats['percentage_changed']:.1f}%)")
    print(f"  Focal method未变更: {stats['focal_method_unchanged']} ({stats['percentage_unchanged']:.1f}%)")
    
    # 获取第一个样本
    print("\n第一个样本:")
    sample = loader.get_sample(0)
    beam_input = loader.prepare_for_beam(sample)
    
    print(f"  测试名称: {beam_input['test_name']}")
    print(f"  项目: {beam_input['project']}")
    print(f"  Focal method变更: {beam_input['focal_method_changed']}")
    print(f"  相关变更数: {len(beam_input['diff_hunks'])}")
    print(f"  测试代码长度: {len(beam_input['test_case'].code)} 字符")

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

class TestResultStatus(str, Enum):
    """Test execution result status"""
    COMPILE_ERROR = "compile_error"
    RUN_FAIL = "run_fail"
    PASS = "pass"
    COVERAGE_LOSS = "coverage_loss"
    MUTATION_LOSS = "mutation_loss"
    COVERAGE_AND_MUTATION_LOSS = "coverage_and_mutation_loss"

@dataclass
class DiffHunk:
    """A single diff hunk from code changes"""
    hunk_id: str  # e.g., "@@ -10,5 +10,8 @@"
    file_path: str
    old_lines: List[str]
    new_lines: List[str]
    context: str  # The actual diff content
    hunk_type: Optional[str] = None  # Type: "test_method", "focal_method", "focal_file", "high_frequency"
    frequency: int = 1  # Frequency of this hunk appearing in the codebase (only for high_frequency type)
    
    def __str__(self) -> str:
        type_str = f", type={self.hunk_type}" if self.hunk_type else ""
        freq_str = f", freq={self.frequency}" if self.hunk_type == "high_frequency" else ""
        return f"DiffHunk({self.file_path}:{self.hunk_id}{type_str}{freq_str})"

@dataclass
class LineCoverage:
    """Detailed coverage info for a single line"""
    line_number: int
    is_covered: bool
    covered_instructions: int = 0
    missed_instructions: int = 0
    covered_branches: int = 0
    missed_branches: int = 0
    
    @property
    def has_branch(self) -> bool:
        return self.covered_branches > 0 or self.missed_branches > 0
    
    @property
    def branch_coverage_str(self) -> str:
        if not self.has_branch:
            return ""
        total = self.covered_branches + self.missed_branches
        return f"{self.covered_branches}/{total} branches"

@dataclass
class DetailedCoverageInfo:
    """Detailed coverage information for focal method"""
    line_coverages: List[LineCoverage] = field(default_factory=list)
    
    @property
    def covered_lines(self) -> List[int]:
        return [lc.line_number for lc in self.line_coverages if lc.is_covered]
    
    @property
    def uncovered_lines(self) -> List[int]:
        return [lc.line_number for lc in self.line_coverages if not lc.is_covered]
    
    @property
    def covered_branch_lines(self) -> List[int]:
        return [lc.line_number for lc in self.line_coverages 
                if lc.has_branch and lc.covered_branches > 0]
    
    @property
    def uncovered_branch_lines(self) -> List[int]:
        return [lc.line_number for lc in self.line_coverages 
                if lc.has_branch and lc.missed_branches > 0]
    
    def get_line_coverage(self, line_number: int) -> Optional[LineCoverage]:
        for lc in self.line_coverages:
            if lc.line_number == line_number:
                return lc
        return None

@dataclass
class CoverageInfo:
    """Coverage information for a test case"""
    covered_lines: List[int] = field(default_factory=list)
    covered_branches: List[str] = field(default_factory=list)
    total_lines: int = 0
    total_branches: int = 0
    coverage_percentage: float = 0.0  # 保留用于向后兼容，默认等于line_coverage_percentage
    # 新增：单独的行覆盖率和分支覆盖率
    line_coverage_percentage: float = 0.0
    branch_coverage_percentage: Optional[float] = None  # None表示没有分支
    covered_lines_count: int = 0  # 覆盖的行数
    covered_branches_count: int = 0  # 覆盖的分支数
    # 新增：详细的 focal method 覆盖信息
    detailed_coverage: Optional[DetailedCoverageInfo] = None

@dataclass
class MutationDetail:
    """Detailed info for a single mutation"""
    mutation_id: str
    line_number: int
    mutator: str  # e.g., "org.pitest.mutationtest.engine.gregor.mutators.VoidMethodCallMutator"
    description: str  # e.g., "removed call to java/util/concurrent/locks/Lock::lock"
    status: str  # "KILLED", "SURVIVED", "NO_COVERAGE"
    killing_test: str = ""  # The test that killed this mutation (if any)
    
    @property
    def mutator_simple_name(self) -> str:
        """Get simple name of the mutator"""
        return self.mutator.split('.')[-1] if self.mutator else ""
    
    @property
    def is_killed(self) -> bool:
        return self.status == "KILLED"
    
    @property
    def is_survived(self) -> bool:
        return self.status == "SURVIVED"
    
    @property
    def is_no_coverage(self) -> bool:
        return self.status == "NO_COVERAGE"

@dataclass
class DetailedMutationInfo:
    """Detailed mutation information for focal method"""
    mutations: List[MutationDetail] = field(default_factory=list)
    
    @property
    def killed_mutations(self) -> List[MutationDetail]:
        return [m for m in self.mutations if m.is_killed]
    
    @property
    def survived_mutations(self) -> List[MutationDetail]:
        return [m for m in self.mutations if m.is_survived]
    
    @property
    def no_coverage_mutations(self) -> List[MutationDetail]:
        return [m for m in self.mutations if m.is_no_coverage]
    
    def get_mutations_at_line(self, line_number: int) -> List[MutationDetail]:
        return [m for m in self.mutations if m.line_number == line_number]

@dataclass
class MutationInfo:
    """Mutation information for a test case"""
    killed_mutations: List[str] = field(default_factory=list)
    total_mutations: int = 0
    kill_percentage: float = 0.0
    # 新增：详细的 focal method 变异信息
    detailed_mutations: Optional[DetailedMutationInfo] = None

@dataclass
class TestCase:
    """Test case representation"""
    name: str
    code: str
    focal_method: str
    coverage_info: CoverageInfo = field(default_factory=CoverageInfo)
    mutation_info: MutationInfo = field(default_factory=MutationInfo)
    test_imports: List[str] = field(default_factory=list)  # 测试类的import语句列表
    new_imports: List[str] = field(default_factory=list)  # 新增的import语句列表（由LLM生成）
    class_fields: List[str] = field(default_factory=list)  # 测试类的类变量（字段）列表
    non_test_methods: List[Dict[str, Any]] = field(default_factory=list)  # 测试类中的非@Test方法
    original_code: str = ""  # 原始测试代码 (bCommit版本，用于过滤已知symbols)
    
@dataclass
class FocalMethodInfo:
    """Information about focal method"""
    name: str
    current_code: str
    original_code: str = ""  # Optional: for reference only
    changed_lines: List[int] = field(default_factory=list)  # Optional: for reference only
    # 新增：focal method 在源文件中的行范围
    start_line: int = 0  # focal method 起始行号（源文件中）
    end_line: int = 0  # focal method 结束行号（源文件中）
    class_name: str = ""  # 完整类名 e.g., "cn.hutool.db.sql.Condition"
    source_file_path: str = ""  # 源文件相对路径

@dataclass
class TestResultInfo:
    """Test execution result information"""
    status: TestResultStatus
    test_case: TestCase
    focal_method_info: FocalMethodInfo
    error_message: Optional[str] = None
    raw_error_output: Optional[str] = None  # 完整的原始错误输出（Maven输出）
    coverage_loss_details: Dict[str, Any] = field(default_factory=dict)
    mutation_loss_details: Dict[str, Any] = field(default_factory=dict)
    test_imports: List[str] = field(default_factory=list)  # 测试类的import语句列表
    test_method_start_line: int = 0  # 测试方法在文件中的起始行号（用于计算相对位置）

@dataclass
class TestUpdateResult:
    """Result from test update agent, containing both test code and new imports"""
    test_code: str  # The updated test method code
    new_imports: List[str] = field(default_factory=list)  # New import statements to add

@dataclass
class RetrievedMethod:
    """A method retrieved from the retrieval system"""
    class_name: str
    method_name: str
    signature: str
    body: Optional[str] = None
    javadoc: Optional[str] = None
    file_path: Optional[str] = None
    relevance_score: float = 0.0

@dataclass
class RetrievedField:
    """A field retrieved from the retrieval system"""
    class_name: str
    field_name: str
    field_type: str
    value: Optional[str] = None
    javadoc: Optional[str] = None
    file_path: Optional[str] = None
    relevance_score: float = 0.0

@dataclass
class RetrievalResult:
    """Result from retrieval system"""
    retrieved_methods: List[RetrievedMethod] = field(default_factory=list)
    retrieved_fields: List[RetrievedField] = field(default_factory=list)
    retrieval_successful: bool = False
    retrieval_reasoning: str = ""
    sql_queries_used: List[str] = field(default_factory=list)
    rag_queries_used: List[str] = field(default_factory=list)
    # 新增：按symbol分组的检索结果和失败的symbols
    retrieved_items: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)  # {symbol: [items]}
    failed_symbols: List[str] = field(default_factory=list)  # 检索失败的symbols

@dataclass
class AnalysisResult:
    """Analysis result from an agent"""
    agent_name: str
    analysis_type: str  # 'error', 'coverage', 'mutation'
    recommendations: Dict[str, Any] = field(default_factory=dict)
    strategies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # 新增：错误分析的结构化字段
    error_type: str = ""  # 'assertion_failure', 'compilation_error', 'project_symbol_missing', etc.
    known_symbols: List[str] = field(default_factory=list)  # 已知符号（需要import）
    unknown_symbols: List[str] = field(default_factory=list)  # 未知符号（项目特定）
    error_locations: List[Dict[str, Any]] = field(default_factory=list)  # 错误位置
    root_cause: str = ""  # LLM分析的根因
    explanation: str = ""  # LLM分析的详细解释
    retrieval_result: Optional[RetrievalResult] = None  # 检索结果
    annotated_test_code: str = ""  # 带错误注释的测试代码（用于显示给LLM）

@dataclass
class UpdateInstruction:
    """Instruction for test update agent"""
    instruction_type: str  # 'fix_compilation_error', 'fix_assertion_failure', 'improve_coverage', 'improve_mutation'
    details: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""

@dataclass
class UpdatePlan:
    """Plan for updating test case"""
    root_causes: List[str]  # List of root causes identified
    update_strategies: List[str]  # Strategies to address root causes
    affected_code_areas: List[str]  # Areas in focal method that need new tests
    assertion_improvements: List[str]  # Assertions that need strengthening
    new_test_cases_needed: List[str]  # Descriptions of new test cases to add
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InputPreprocessResult:
    """Result of input preprocessing for test updates (filtered hunks and test context)"""
    filtered_hunks: List[DiffHunk] = field(default_factory=list)  # Filtered relevant hunks
    retrieval_result: Optional[RetrievalResult] = None  # Retrieved information
    reasoning: str = ""  # Detailed reasoning behind the analysis
    needs_retrieval: bool = False  # Whether retrieval was needed
    filtered_class_fields: Optional[List[str]] = None  # Filtered relevant class fields (None = not filtered, use all)
    filtered_non_test_methods: Optional[List[Dict[str, Any]]] = None  # Filtered relevant non-test methods (None = not filtered, use all)

@dataclass
class IterationResult:
    """Result of a single iteration"""
    iteration: int
    test_result: TestResultInfo
    analysis_results: List[AnalysisResult] = field(default_factory=list)
    update_instructions: List[UpdateInstruction] = field(default_factory=list)
    updated_test_code: Optional[str] = None
    new_imports: List[str] = field(default_factory=list)  # 该次迭代新增的import语句
    score: float = 0.0  # For ranking best result

@dataclass
class FrameworkState:
    """Framework state tracking"""
    current_iteration: int = 0
    test_case: Optional[TestCase] = None
    focal_method_info: Optional[FocalMethodInfo] = None
    iteration_results: List[IterationResult] = field(default_factory=list)
    best_result: Optional[IterationResult] = None
    best_score: float = 0.0

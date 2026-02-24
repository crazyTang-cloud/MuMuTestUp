"""
DEPRECATED: This config file is for standalone use of index_retrieve_module_v3.

When using this module as part of the BEAM framework, configuration should
be managed through the main config.py in the project root directory.

The BEAM framework automatically reads settings from the main config.py,
including API keys, model settings, and temperature values.
"""
from dataclasses import dataclass
import os

@dataclass
class Config:
    repo_path: str = r"C:\Users\zyc\Desktop\files\agent\dromara-hutool"
    error_commit_id: str = "912d8c48c"

    # 报错的测试函数代码
    error_test_code_log: str ="""
    @Test public void parseTest() {
    // Test with valid expression
    Condition age = Condition.parse("age", "< 10");
    Assert.assertEquals("age < ?", age.toString());
    Assert.assertSame(BigDecimal.class, age.getValue().getClass());

    // Test with an empty string for the expression
    try {
        Condition invalidAge = Condition.parse("age", "");
        Assert.fail("Expected an exception to be thrown");
    } catch (IllegalArgumentException e) {
        Assert.assertTrue(e.getMessage().contains("Invalid expression"));
    }

    // Test with a null value for the expression
    try {
        Condition nullAge = Condition.parse("age", null);
        Assert.fail("Expected an exception to be thrown");
    } catch (IllegalArgumentException e) {
        Assert.assertTrue(e.getMessage().contains("Invalid expression"));
    }

    // Test with invalid expressions
    try {
        Condition invalidExpression = Condition.parse("age", "<");
        Assert.fail("Expected an exception to be thrown");
    } catch (IllegalArgumentException e) {
        Assert.assertTrue(e.getMessage().contains("Invalid expression"));
    }
}
"""
    #报错的错误信息（简短版）
    error_message: str ="""expected same:<class java.math.BigDecimal> was not:<class java.lang.Long>"""

    #报错的错误日志
    error_log: str ="""
[INFO] Scanning for projects...
[INFO] ------------------------------------------------------------------------
[INFO] Reactor Build Order:
[INFO] 
[INFO] hutool                                                                     [pom]
[INFO] hutool-core                                                                [jar]
[INFO] hutool-log                                                                 [jar]
[INFO] hutool-setting                                                             [jar]
[INFO] hutool-db                                                                  [jar]
[INFO] 
[INFO] ----------------------< cn.hutool:hutool-parent >-----------------------
[INFO] Building hutool 5.7.10-SNAPSHOT                                    [1/5]
[INFO]   from pom.xml
[INFO] --------------------------------[ pom ]---------------------------------
[INFO] 
[INFO] --- jacoco:0.8.8:prepare-agent (default-cli) @ hutool-parent ---
[INFO] argLine set to -javaagent:/data/david/project/beam/repos/dromara_hutool/.m2/repository/org/jacoco/org.jacoco.agent/0.8.8/org.jacoco.agent-0.8.8-runtime.jar=destfile=/data/david/project/beam/repos/dromara/hutool/target/jacoco.exec
[INFO] 
[INFO] --- jacoco:0.8.8:report (default-cli) @ hutool-parent ---
[INFO] Skipping JaCoCo execution due to missing execution data file.
[INFO] 
[INFO] -----------------------< cn.hutool:hutool-core >------------------------
[INFO] Building hutool-core 5.7.10-SNAPSHOT                               [2/5]
[INFO]   from hutool-core/pom.xml
[INFO] --------------------------------[ jar ]---------------------------------
[INFO] 
[INFO] --- jacoco:0.8.8:prepare-agent (default-cli) @ hutool-core ---
[INFO] argLine set to -javaagent:/data/david/project/beam/repos/dromara_hutool/.m2/repository/org/jacoco/org.jacoco.agent/0.8.8/org.jacoco.agent-0.8.8-runtime.jar=destfile=/data/david/project/beam/repos/dromara/hutool/hutool-core/target/jacoco.exec
[INFO] 
[INFO] --- resources:3.3.1:resources (default-resources) @ hutool-core ---
[INFO] skip non existing resourceDirectory /data/david/project/beam/repos/dromara/hutool/hutool-core/src/main/resources
[INFO] 
[INFO] --- compiler:3.8.1:compile (default-compile) @ hutool-core ---
[INFO] Changes detected - recompiling the module!
[INFO] Compiling 542 source files to /data/david/project/beam/repos/dromara/hutool/hutool-core/target/classes
[INFO] /data/david/project/beam/repos/dromara/hutool/hutool-core/src/main/java/cn/hutool/core/map/CustomKeyMap.java: Some input files use unchecked or unsafe operations.
[INFO] /data/david/project/beam/repos/dromara/hutool/hutool-core/src/main/java/cn/hutool/core/map/CustomKeyMap.java: Recompile with -Xlint:unchecked for details.
[INFO] 
[INFO] --- resources:3.3.1:testResources (default-testResources) @ hutool-core ---
[INFO] Copying 14 resources from src/test/resources to target/test-classes
[INFO] 
[INFO] --- compiler:3.8.1:testCompile (default-testCompile) @ hutool-core ---
[INFO] Changes detected - recompiling the module!
[INFO] Compiling 174 source files to /data/david/project/beam/repos/dromara/hutool/hutool-core/target/test-classes
[INFO] /data/david/project/beam/repos/dromara/hutool/hutool-core/src/test/java/cn/hutool/core/date/DateUtilTest.java: /data/david/project/beam/repos/dromara/hutool/hutool-core/src/test/java/cn/hutool/core/date/DateUtilTest.java uses or overrides a deprecated API.
[INFO] /data/david/project/beam/repos/dromara/hutool/hutool-core/src/test/java/cn/hutool/core/date/DateUtilTest.java: Recompile with -Xlint:deprecation for details.
[INFO] 
[INFO] --- surefire:3.2.2:test (default-test) @ hutool-core ---
[INFO] 
[INFO] --- jacoco:0.8.8:report (default-cli) @ hutool-core ---
[INFO] Skipping JaCoCo execution due to missing execution data file.
[INFO] 
[INFO] ------------------------< cn.hutool:hutool-log >------------------------
[INFO] Building hutool-log 5.7.10-SNAPSHOT                                [3/5]
[INFO]   from hutool-log/pom.xml
[INFO] --------------------------------[ jar ]---------------------------------
[INFO] 
[INFO] --- jacoco:0.8.8:prepare-agent (default-cli) @ hutool-log ---
[INFO] argLine set to -javaagent:/data/david/project/beam/repos/dromara_hutool/.m2/repository/org/jacoco/org.jacoco.agent/0.8.8/org.jacoco.agent-0.8.8-runtime.jar=destfile=/data/david/project/beam/repos/dromara/hutool/hutool-log/target/jacoco.exec
[INFO] 
[INFO] --- resources:3.3.1:resources (default-resources) @ hutool-log ---
[INFO] Copying 1 resource from src/main/resources to target/classes
[INFO] 
[INFO] --- compiler:3.8.1:compile (default-compile) @ hutool-log ---
[INFO] Changes detected - recompiling the module!
[INFO] Compiling 44 source files to /data/david/project/beam/repos/dromara/hutool/hutool-log/target/classes
[INFO] 
[INFO] --- resources:3.3.1:testResources (default-testResources) @ hutool-log ---
[INFO] Copying 7 resources from src/test/resources to target/test-classes
[INFO] 
[INFO] --- compiler:3.8.1:testCompile (default-testCompile) @ hutool-log ---
[INFO] Changes detected - recompiling the m... truncated log
    """

    agent_model: str = "gpt-4o-2024-11-20"

    def __post_init__(self):
        if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError("OPENAI_API_KEY is not set")
        if os.getenv("OPENAI_BASE_URL") is None:
            raise ValueError("OPENAI_BASE_URL is not set")


    
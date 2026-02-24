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
    repo_path: str = r"D:\identification_update\update\repository\commons-configuration"
    error_commit_id: str = "3b98bf"

    # 报错的测试函数代码
    error_test_code_log: str ="""
    public void testRegisterLookup()
    {
        int cnt = interpolator.prefixSet().size();
        interpolator.registerLookup(TEST_PREFIX, new DefaultLookup());
        assertTrue("New lookup not registered", interpolator.prefixSet()
                .contains(TEST_PREFIX));
        assertEquals("Wrong number of registered lookups", cnt + 1,
                interpolator.prefixSet().size());
        ConfigurationInterpolator int2 = new ConfigurationInterpolator();
        assertFalse("Local registration has global impact", int2.prefixSet()
                .contains(TEST_PREFIX));
    }
"""
    #报错的错误信息（简短版）
    error_message: str =r"""
D:\identification_update\update\repository\commons-configuration\src\test\java\org\apache\commons\configuration\interpol\TestConfigurationInterpolator.java:117:54
java: 找不到符号
  符号:   类 DefaultLookup
  位置: 类 org.apache.commons.configuration.interpol.TestConfigurationInterpolator
    """

    #报错的错误日志
    error_log: str =r"""
D:\identification_update\update\repository\commons-configuration\src\test\java\org\apache\commons\configuration\interpol\TestConfigurationInterpolator.java:117:54
java: 找不到符号
  符号:   类 DefaultLookup
  位置: 类 org.apache.commons.configuration.interpol.TestConfigurationInterpolator
    """

    agent_model: str = "gpt-4o-2024-11-20"

    def __post_init__(self):
        if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError("OPENAI_API_KEY is not set")
        if os.getenv("OPENAI_BASE_URL") is None:
            raise ValueError("OPENAI_BASE_URL is not set")


    
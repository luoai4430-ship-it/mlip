import os
import pathlib
import nest_asyncio
import logging
from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer

nest_asyncio.apply()
load_dotenv()



logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

mcp_tools2 = McpToolset(
    connection_params=SseConnectionParams(
        url="http://127.0.0.1:8000/sse"  
    )
)

agent = LlmAgent(
    model=LiteLlm(
        model="deepseek/deepseek-chat",
        temperature=0.1,
        max_tokens=4096
    ),
    name="DeepMD_Linux_Expert",
    description="专注于 DeepMD-kit 数据准备、训练及 LAMMPS 模拟的专家",
    instruction=f"""
你是一个精通 DeepMD-kit、LAMMPS 和 VASP 的高级计算物理专家。
你的 Python 服务器端运行在 Windows 环境下，LAMMPS是通过 MCP 工具内部跨系统调用 WSL (Ubuntu) 和 Docker 来执行的。

执行准则：
1. 【路径规范】你的主工作目录是 C:/Users/Administrator/Desktop/MLIP_workspace。当你需要调用工具读写文件或保存数据时，请直接传递这个 Windows 绝对路径（工具内部已经封装好了向 WSL 转换的逻辑，你无需手动转换）。
2. 【自主纠错】不要自己写脚本，你没有命令行工具运行，遇到报错时，仔细阅读返回的错误日志,。
3. 【禁止废话】拒绝向用户询问“是否要执行”、“我接下来应该怎么做”。只要用户的终极目标明确，立刻自主完成完整的任务链条。
4. 【工具使用】读取大文件时，务必使用限制行数的参数，在调用时间较长的工具时可以过五分钟再看结果，耐心等待工具完成任务。""",
    tools=[mcp_tools2]
)


summarization_llm = LiteLlm(
    model="deepseek/deepseek-chat",
    temperature=0.1,
    max_tokens=2048
)

summarizer = LlmEventSummarizer(llm=summarization_llm)

compaction_config = EventsCompactionConfig(
    summarizer=summarizer,
    compaction_interval=5, 
    overlap_size=1         
)

app = App(
    name="agent",  
    root_agent=agent,
    events_compaction_config=compaction_config
)

root_agent = app

import asyncio
from contextlib import AsyncExitStack
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
import os
from openai import AsyncOpenAI
import sys
import logging
from typing import Optional, Dict, Any, List

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MCPClient:
    def __init__(self):
        """初始化MCP客户端"""
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self._streams_contexts = {}  # 存储所有SSE上下文管理器
        
        # 初始化OpenAI客户端
        self._init_openai_client()
        
        # 加载配置
        self.config = self._load_config()
        
        # 存储工具和会话信息
        self.all_server_tools: Dict[str, List] = {}
        self.active_sessions: Dict[str, ClientSession] = {}
        
    def _init_openai_client(self):
        """初始化OpenAI客户端"""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            # 如果环境变量不存在，使用硬编码的key（仅用于开发）
            api_key = "XXXXXXXX"
            logger.warning("使用硬编码API密钥，建议设置OPENROUTER_API_KEY环境变量")
        
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        # self.AI_model = "deepseek/deepseek-chat"
        self.AI_model = "qwen/qwen-plus"
        
    def _load_config(self) -> Dict:
        """加载MCP服务器配置"""
        config_path = os.path.join(os.path.dirname(__file__), "mcp_server_config.json")
        try:
            with open(config_path, "r", encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"成功加载配置文件: {config_path}")
                return config
        except FileNotFoundError:
            logger.error(f"配置文件未找到: {config_path}")
            return {"mcpServers": {}}
        except json.JSONDecodeError as e:
            logger.error(f"配置文件JSON格式错误: {str(e)}")
            return {"mcpServers": {}}
        except Exception as e:
            logger.error(f"加载配置文件时发生未知错误: {str(e)}")
            return {"mcpServers": {}}
    
    def list_available_servers(self) -> Dict[str, Dict[str, Any]]:
        """列出所有可用的服务器"""
        return self.config.get("mcpServers", {})

    async def _connect_stdio_server(self, server_name: str, server_config: Dict) -> Optional[ClientSession]:
        """连接到stdio类型的服务器"""
        try:
            command = server_config.get("command", "python")
            args = server_config.get("args", [])
            env = server_config.get("env")
            
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=env
            )
            
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            stdio, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
            
            logger.info(f"成功连接到stdio服务器: {server_name}")
            return session
            
        except Exception as e:
            logger.error(f"连接stdio服务器 '{server_name}' 失败: {str(e)}")
            return None

    async def _connect_sse_server(self, server_name: str, server_config: Dict) -> Optional[ClientSession]:
        """连接到SSE类型的服务器"""
        try:
            url = server_config.get("url")
            if not url:
                logger.warning(f"服务器 '{server_name}' 缺少URL配置")
                return None
            
            # 直接使用exit_stack管理SSE连接，避免手动管理
            streams = await self.exit_stack.enter_async_context(sse_client(url=url))
            session = await self.exit_stack.enter_async_context(ClientSession(*streams))
            
            logger.info(f"成功连接到SSE服务器: {server_name}")
            return session
            
        except Exception as e:
            logger.error(f"连接SSE服务器 '{server_name}' 失败: {str(e)}")
            return None

    async def connect_to_all_servers(self) -> int:
        """连接到所有可用的MCP服务器"""
        servers = self.list_available_servers()
        if not servers:
            raise ValueError("未找到有效的MCP服务器配置")
        
        connected_servers = 0
        
        for server_name, server_config in servers.items():
            # 检查服务器是否被禁用
            if server_config.get("disabled", False):
                logger.info(f"服务器 '{server_name}' 已被禁用，跳过连接")
                continue
            
            transport_type = server_config.get("transportType", "stdio")
            session = None
            
            # 根据传输类型连接服务器
            if transport_type == "stdio":
                session = await self._connect_stdio_server(server_name, server_config)
            elif transport_type == "sse":
                session = await self._connect_sse_server(server_name, server_config)
            else:
                logger.warning(f"服务器 '{server_name}' 使用不支持的传输类型: {transport_type}")
                continue
            
            if session is None:
                continue
            
            try:
                # 初始化会话
                await session.initialize()
                
                # 保存会话到活跃会话字典
                self.active_sessions[server_name] = session
                
                # 列出可用工具
                response = await session.list_tools()
                tools = response.tools
                
                # 保存工具到类变量
                self.all_server_tools[server_name] = tools
                
                tool_names = [tool.name for tool in tools]
                logger.info(f"服务器 '{server_name}' 连接成功，可用工具: {tool_names}")
                connected_servers += 1
                
            except Exception as e:
                logger.error(f"初始化服务器 '{server_name}' 会话时出错: {str(e)}")
                # 从活跃会话中移除失败的会话
                if server_name in self.active_sessions:
                    del self.active_sessions[server_name]
        
        # 统计信息
        total_tools = sum(len(tools) for tools in self.all_server_tools.values())
        logger.info(f"成功连接到 {connected_servers} 个服务器，共获取 {total_tools} 个工具")
        
        return connected_servers
    
    async def switch_active_session(self, server_name: str):
        """切换当前活跃的会话"""
        if server_name not in self.active_sessions:
            available_servers = ", ".join(self.active_sessions.keys())
            raise ValueError(f"未找到名为 '{server_name}' 的活跃会话。可用服务器: {available_servers}")
        
        self.session = self.active_sessions[server_name]
        logger.info(f"已切换到服务器 '{server_name}' 的会话")

    def _prepare_tools_for_openai(self) -> List[Dict]:
        """为OpenAI准备工具列表"""
        available_tools = []
        for server_name, tools in self.all_server_tools.items():
            for tool in tools:
                available_tools.append({
                    "type": "function",
                    "function": {
                        "name": f"{server_name}:{tool.name}",
                        "description": f"[{server_name}] {tool.description}",
                        "parameters": tool.inputSchema
                    }
                })
        return available_tools

    def _parse_tool_name(self, tool_call_name: str) -> tuple[str, str]:
        """解析工具名称，返回服务器名和工具名"""
        if ":" in tool_call_name:
            return tool_call_name.split(":", 1)
        else:
            # 如果没有服务器前缀，使用当前活跃会话
            default_server = next(iter(self.active_sessions.keys()))
            return default_server, tool_call_name

    async def _execute_tool_call(self, tool_call, messages: List[Dict]) -> str:
        """执行单个工具调用"""
        tool_call_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        
        server_name, tool_name = self._parse_tool_name(tool_call_name)
        
        # 获取对应服务器的会话
        if server_name not in self.active_sessions:
            error_msg = f"找不到服务器 '{server_name}'"
            logger.error(error_msg)
            return f"[错误: {error_msg}]"
        
        session = self.active_sessions[server_name]
        
        try:
            result = await session.call_tool(tool_name, tool_args)
            success_msg = f"调用工具 {server_name}:{tool_name}，参数: {tool_args}"
            logger.info(success_msg)
            
            # 将工具调用和结果添加到消息中
            messages.append({
                "role": "assistant",
                "tool_calls": [{
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call_name,
                        "arguments": json.dumps(tool_args)
                    }
                }]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result.content)
            })
            
            return f"[{success_msg}]"
            
        except Exception as e:
            error_message = f"调用工具 {server_name}:{tool_name} 失败: {str(e)}"
            logger.error(error_message)
            
            # 将错误信息添加到消息中
            messages.append({
                "role": "assistant",
                "tool_calls": [{
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call_name,
                        "arguments": json.dumps(tool_args)
                    }
                }]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": f"[错误: {error_message}]"
            })
            
            return f"[错误: {error_message}]"

    async def process_query(self, query: str) -> str:
        """处理用户查询"""
        if not self.active_sessions:
            return "错误: 没有可用的MCP服务器连接"
        
        # 准备消息和工具
        messages = [
            {
                "role": "system",
                "content": "你是一个智能助手，可以回答用户问题。对于简单的问题，直接回答即可。只有在需要特定信息或执行特定操作时，才使用提供的工具。请自主判断是否需要调用工具来回答问题。每个工具名称前都有服务器名称前缀，格式为'服务器名:工具名'。"
            },
            {
                "role": "user",
                "content": query
            }
        ]
        
        available_tools = self._prepare_tools_for_openai()
        final_text = []
        
        try:
            # 初始OpenAI API调用
            response = await self.client.chat.completions.create(
                model=self.AI_model,
                messages=messages,
                tools=available_tools,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            if message.content:
                final_text.append(message.content)
            
            # 处理工具调用循环
            max_iterations = 10  # 防止无限循环
            iteration = 0
            
            while message.tool_calls and iteration < max_iterations:
                iteration += 1
                logger.info(f"执行第 {iteration} 轮工具调用")
                
                # 处理每个工具调用
                for tool_call in message.tool_calls:
                    result_msg = await self._execute_tool_call(tool_call, messages)
                    final_text.append(result_msg)
                
                # 获取OpenAI的下一个响应
                response = await self.client.chat.completions.create(
                    model=self.AI_model,
                    messages=messages,
                    tools=available_tools,
                    tool_choice="auto"
                )
                
                message = response.choices[0].message
                if message.content:
                    final_text.append(message.content)
            
            if iteration >= max_iterations:
                final_text.append("[警告: 达到最大工具调用次数限制]")
            
        except Exception as e:
            error_msg = f"处理查询时发生错误: {str(e)}"
            logger.error(error_msg)
            final_text.append(f"[错误: {error_msg}]")
        
        return "\n".join(final_text)

    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("\nMCP Client Started!")
        print("输入您的查询或输入 'quit' 退出。")
        print("可用命令:")
        print("  - 'servers': 显示所有活跃服务器")
        print("  - 'tools': 显示所有可用工具")
        
        while True:
            try:
                query = input("\n查询: ").strip()
                
                if query.lower() == 'quit':
                    break
                elif query.lower() == 'servers':
                    self._show_active_servers()
                    continue
                elif query.lower() == 'tools':
                    self._show_available_tools()
                    continue
                
                response = await self.process_query(query)
                print(f"\n回复: {response}")
                
            except KeyboardInterrupt:
                print("\n\n收到中断信号，正在退出...")
                break
            except Exception as e:
                logger.error(f"聊天循环中发生错误: {str(e)}")
                print(f"\n错误: {str(e)}")
    
    def _show_active_servers(self):
        """显示活跃服务器信息"""
        if not self.active_sessions:
            print("没有活跃的服务器连接")
            return
        
        print("\n活跃服务器:")
        for server_name in self.active_sessions.keys():
            current = " (当前)" if self.session == self.active_sessions[server_name] else ""
            tool_count = len(self.all_server_tools.get(server_name, []))
            print(f"  - {server_name}{current}: {tool_count} 个工具")
    
    def _show_available_tools(self):
        """显示所有可用工具"""
        if not self.all_server_tools:
            print("没有可用的工具")
            return
        
        print("\n所有可用工具:")
        for server_name, tools in self.all_server_tools.items():
            print(f"\n服务器 '{server_name}':")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description}")
    
    async def cleanup(self):
        """清理资源"""
        logger.info("正在清理资源...")
        
        # 首先关闭所有SSE连接
        for server_name, context in self._streams_contexts.items():
            try:
                await context.__aexit__(None, None, None)
                logger.info(f"已关闭服务器 '{server_name}' 的SSE连接")
            except Exception as e:
                logger.error(f"关闭服务器 '{server_name}' 的SSE连接时出错: {str(e)}")
        
        # 清空SSE上下文管理器字典
        self._streams_contexts.clear()
        
        # 最后关闭exit_stack
        try:
            await self.exit_stack.aclose()
            logger.info("资源清理完成")
        except Exception as e:
            logger.error(f"关闭exit_stack时出错: {str(e)}")


async def main():
    """主函数"""
    client = MCPClient()
    
    try:
        # 显示可用的服务器
        servers = client.list_available_servers()
        if not servers:
            logger.error("未找到有效的MCP服务器配置，请检查mcp_server_config.json文件")
            sys.exit(1)
        
        print("可用的MCP服务器:")
        for i, (name, config) in enumerate(servers.items()):
            transport_type = config.get("transportType", "stdio")
            disabled = "（已禁用）" if config.get("disabled", False) else ""
            
            if transport_type == "stdio":
                command = config.get("command", "python")
                args_str = " ".join(config.get("args", []))
                print(f"{i+1}. {name}{disabled}: {command} {args_str}")
            else:
                url = config.get("url", "未指定URL")
                print(f"{i+1}. {name}{disabled}: {transport_type} - {url}")
        
        # 连接所有服务器
        connected_count = await client.connect_to_all_servers()
        
        if connected_count == 0:
            logger.error("没有成功连接到任何服务器")
            sys.exit(1)
        
        # 显示工具信息
        client._show_available_tools()
        
        # 启动聊天循环
        await client.chat_loop()
        
    except ValueError as e:
        logger.error(f"配置错误: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("收到中断信号")
    except Exception as e:
        logger.error(f"程序运行时发生未知错误: {str(e)}")
        sys.exit(1)
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

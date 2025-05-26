


          
# MCP Client 使用文档

## 项目简介
MCP Client 是一个基于Python的多服务器客户端工具，支持连接和管理多个MCP服务器。该工具提供了灵活的服务器连接管理和交互式命令行界面，集成了OpenAI API，可以智能处理用户查询并调用相应的工具。

## 环境要求

### 系统要求
- Python 3.x
- 异步IO支持

### 依赖包
- openai
- mcp
- httpx
- httpx-sse
- pydantic

## 配置说明

# 创建虚拟环境
```bash
python3 -m venv venv

# 激活虚拟环境
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 环境变量
- `OPENROUTER_API_KEY`: OpenRouter API密钥（必需）
  - 用于访问OpenAI API服务
  - 如未设置将使用默认测试密钥（不推荐用于生产环境）

  可以使用以下命令将`OPENROUTER_API_KEY`写到环境变量里
  ```bash
  echo 'export OPENROUTER_API_KEY="XXXXXXX"' >> venv/bin/activate
  ```` 



### 配置文件
文件名：`mcp_server_config.json`

支持两种服务器类型配置：

1. stdio类型服务器：
```json
{
  "serverName": {
    "command": "执行命令",
    "args": ["参数列表"],
    "env": {
      "环境变量名": "环境变量值"
    },
    "transportType": "stdio"
  }
}
```

2. SSE类型服务器：
```json
{
  "serverName": {
    "url": "服务器URL地址",
    "disabled": false,
    "transportType": "sse"
  }
}
```

## 主要功能

### 命令行指令
- `servers`: 显示所有当前活跃的服务器
- `tools`: 显示所有可用的工具列表
- `quit`: 退出程序

### 核心功能
1. 多服务器管理
   - 支持同时连接多个MCP服务器
   - 自动管理服务器连接生命周期
   - 支持在不同服务器间切换

2. 工具调用
   - 自动发现并加载服务器工具
   - 智能工具选择和调用
   - 支持跨服务器工具调用

3. 智能对话
   - 集成OpenAI API
   - 自动工具选择和参数解析
   - 支持多轮对话和上下文管理

## 使用示例
### 1、安装依赖
```bash
pip install -r requirements.txt
```
### 2. 启动客户端

```bash
# 启动程序
python client.py
```
### 3. 交互示例
```bash
# 查看服务器列表
> servers

# 查看可用工具
> tools

# 执行查询
> 查询: 请帮我查询数据库状态
```

## 错误处理
- 配置文件错误：自动返回默认配置
- 服务器连接失败：自动跳过并继续其他服务器连接
- 工具调用错误：详细的错误信息记录
- 资源清理：自动清理所有连接和资源

## 日志系统
- 使用Python标准logging模块
- 日志级别：INFO
- 格式：时间 - 级别 - 消息
- 自动记录关键操作和错误信息

## 开发者注意事项
1. 异步编程
   - 使用`asyncio`进行异步操作
   - 使用`AsyncExitStack`管理资源
   - 注意异步上下文管理

2. 错误处理
   - 所有关键操作都有异常处理
   - 使用日志记录详细错误信息
   - 保证资源正确释放

3. 配置管理
   - 检查配置文件格式
   - 验证必要的环境变量
   - 提供合理的默认值

4. 安全性
   - API密钥管理
   - 服务器连接安全
   - 错误信息安全处理

## 贡献指南
1. 代码风格
   - 遵循PEP 8规范
   - 使用类型注解
   - 添加适当的注释

2. 测试
   - 编写单元测试
   - 测试不同类型的服务器连接
   - 测试错误处理场景

3. 文档
   - 更新API文档
   - 添加新功能说明
   - 保持示例代码最新

## 常见问题解答
1. Q: 如何添加新的服务器？
   A: 在配置文件中添加新的服务器配置，支持stdio和sse两种类型。

2. Q: 如何处理连接超时？
   A: 系统会自动重试并记录日志，可以通过日志查看详细信息。

3. Q: 如何扩展新的工具？
   A: 在服务器端实现新的工具，客户端会自动发现并加载。

        
---
title: Model Context Protocol (MCP)
source: https://docs.langchain.com/oss/python/langchain/mcp
category: tools
url: https://docs.langchain.com/oss/python/langchain/mcp
---

# Model Context Protocol (MCP)

> Connect LangChain agents to MCP servers with the MCPAdapter, built on FastMCP.

Model Context Protocol (MCP) is an open protocol that standardizes how applications provide tools and context to language models. LangChain agents call tools defined on MCP servers through `MCPAdapter`, which discovers a server's tools and adapts them into LangChain tools you can pass straight to `create_agent`.

`MCPAdapter` is built on FastMCP, which handles transport inference, protocol negotiation, connection management, and authentication. This section covers the LangChain-specific layer and links out to the FastMCP client documentation for the connection details underneath.

The `langchain.mcp` namespace requires `langchain[mcp]>=1.4.0` and is in beta. Importing from it raises a `LangChainBetaWarning` once per process. The API may change.

If you used MCP before v1.4.0, see Migrate from `langchain-mcp-adapters`.

## Install

Install LangChain with the `mcp` extra, which pulls in FastMCP:

  ```bash pip
  pip install "langchain[mcp]"
  ```

  ```bash uv
  uv add "langchain[mcp]"
  ```

## Quickstart

Open an `MCPAdapter`, discover the server's tools with `list_tools()`, and build the agent inside the context. The tools hold the client, so the agent stays usable after the context exits:

```python
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter


async def main():
    async with MCPAdapter("https://example.com/mcp") as adapter:
        tools = await adapter.list_tools()
        agent = create_agent("claude-sonnet-5", tools)
        return await agent.ainvoke({"messages": [{"role": "user", "content": "..."}]})
```

## Transports

`MCPAdapter` infers the transport from the target you hand it, so the only thing that changes between an in-process server, a local script over stdio, and a remote URL is the target itself:

```python
from pathlib import Path

from langchain.mcp import MCPAdapter

# An in-process FastMCP server: no subprocess, no socket. Ideal for tests.
in_memory = MCPAdapter(server)  # a FastMCP instance

# A script path is launched over stdio, one subprocess per adapter.
stdio = MCPAdapter(Path("weather_server.py"))

# A string must be an http(s) URL, reached over streamable HTTP.
http = MCPAdapter("https://example.com/mcp")
```

A target can be any of the following:

* **An `http`/`https` URL** (`str`): reached over streamable HTTP.
* **A script path** (`Path`): launched as a subprocess over stdio.
* **A transport object** (`StreamableTransport`): a pre-configured transport object. See Client Transports.
* **An in-process `FastMCP` server**: connected in-memory, with no subprocess or socket.
* **An `MCPConfig` dict** (`{"mcpServers": {...}}`): several servers behind one adapter. See Connections.
* **A prebuilt `fastmcp.Client`**: for full control over transport, caching, and protocol negotiation.

A `str` target must be an `http` or `https` URL. FastMCP resolves a string by testing it as a filesystem path before testing it as a URL, so a string naming an existing `.py` or `.js` file would launch that file as a subprocess. Because strings are the form a target most often arrives in from configuration or from a model, `MCPAdapter` rejects strings that don't match the shape of a URL.

## Next steps

Connections
    Connection lifecycle, multiple servers, protocol eras, and caching.

Authentication
    Bearer tokens, OAuth 2.1, and per-user server auth.

Tools
    Load MCP tools into agents, control their execution, and handle their outputs.

***

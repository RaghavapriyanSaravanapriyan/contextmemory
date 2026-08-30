<div align="center">

<img width="1600" height="533" alt="image" src="https://github.com/user-attachments/assets/67ad9bce-94a9-43c3-97d4-f26d3d9231c3" />
                                                                                                                                
### Give your local model a memory that does not lie.

**A fast, temporal memory layer for Ollama, MCP agents, and AI applications.**

<br />

![Python](https://img.shields.io/badge/python-3.11%2B-8b9cff?style=flat-square)
![C++](https://img.shields.io/badge/core-C%2B%2B-65e6b0?style=flat-square)
![Ollama](https://img.shields.io/badge/ollama-local-ffb86b?style=flat-square)
![MCP](https://img.shields.io/badge/MCP-ready-c9a7ff?style=flat-square)

<br />

*Your model is brilliant, fast, and forgetful.*

*ContextMemory is the quiet brain beside it.*

</div>

---

## The Short Version

AI assistants forget between conversations. ContextMemory gives them somewhere
to keep the useful parts: who you are, what changed, what matters, and what is
still true.

It is local-first and deliberately unglamorous about your data. No cloud
memory service. No mystery database. No giant context dump. A C++ temporal
memory engine, a Python front desk, an Ollama chat, an MCP bridge, and a TUI
that lets you watch the brain work.

## Start Talking To A Model With Memory

```bash
uv sync
ollama pull qwen3:4b
uv run contextmemory chat --model qwen3:4b
```

That is the whole loop. ContextMemory will connect to Ollama, start it if
needed, wire the MCP tools into the model, and keep the memory journal on disk.

```text
ContextMemory chat | Ollama: qwen3:4b | MCP: connected

you> I live in Seattle and prefer Vim.
ollama> I will remember that.

you> Where do I live?
ollama> You live in Seattle.
```

Restart the command. Ask again. The fact is still there.

```bash
uv run contextmemory chat
```

Omit `--model` to use the first model installed in Ollama. Type `/exit` to
leave. This is a normal terminal chat, not a special UI. Ollama is the voice;
ContextMemory is the memory behind the voice.

## Give Any Agent The Same Brain

ContextMemory speaks MCP over stdio. Add this server to OpenCode, Claude Code,
Cursor, Cline, or any other MCP client:

```json
{
  "mcpServers": {
    "contextmemory": {
      "command": "uv",
      "args": [
        "run",
        "--project",
        "/absolute/path/to/contextmemory",
        "contextmemory",
        "mcp",
        "--container",
        "brain"
      ]
    }
  }
}
```

Or run the bridge directly:

```bash
contextmemory mcp --container brain
```

The brain exposes four small tools:

| Tool | Job |
| --- | --- |
| `memory(content)` | Keep a meaningful fact or conversation detail |
| `recall(query)` | Find memories relevant to a question |
| `context(topic)` | Return compact working context |
| `forget(id)` | Remove a memory when it should not survive |

MCP clients decide when to call tools. The built-in `contextmemory chat` host
does the wiring for you and also supports native Ollama tool calls.

## The Demo That Makes It Click

```bash
uv run contextmemory demo
```

Choose **Run Offline Demo**. In under a minute, the brain walks through a small
story:

```text
New York  ------ moved ------>  Seattle
Acme      ------ joined ----->  Globex

current truth       Seattle / Globex
historical truth    New York / Acme
unknown question    I don't have enough information
```

The dashboard turns that story into something you can see: Brain, Timeline,
Why, Models, Retrieval Live, Performance, Connections, and Health. Press `O`
to select a local Ollama model. Press `R` to replay the story.

For the live version:

```bash
uv run contextmemory demo --live --model qwen3:4b
```

Or use the repository launcher:

```bash
./run.sh
./run.sh --live --model qwen3:4b
```

## What Makes It Different

| Ordinary chat memory | ContextMemory |
| --- | --- |
| Keeps getting longer | Retrieves only what matters |
| Treats old and new facts alike | Tracks validity over time |
| Guesses when context is thin | Abstains when evidence is missing |
| Hides the retrieval path | Shows provenance, routing, and timing |
| Depends on a hosted service | Runs locally beside your model |

The important question is not “how much did we store?”

It is:

> Did we retrieve the right thing, at the right time, for the right reason?

## How The Brain Works

```text
conversation / tool trace
          |
       CAPTURE       immutable episode, cheap and lossless
          |
       EXTRACT      optional single-pass LLM distillation
          |
      RECONCILE     deduplicate, version, and project truth
          |
        RECALL      bounded temporal query plan
          |
         PACK       minimum-sufficient evidence under a budget
          |
       ANSWER       Ollama, a frontier model, or your own reader
```

Three layers, three jobs:

- **C++ ETMC core** handles capture, temporal validity, reconciliation,
  projections, search, and evidence packing.
- **Python layer** handles orchestration, Ollama, the public API, MCP, the TUI,
  and integrations.
- **Evaluation rig** measures write precision, evolution, forgetting, accuracy,
  and latency instead of waving at a leaderboard.

Episodes are immutable. Cells carry validity windows. When “I live in New York”
becomes “I moved to Seattle,” both truths remain available, but only Seattle is
current. When the memory has no answer, the honest answer wins.

## CLI Surface

```bash
# The normal terminal experience
contextmemory chat --model qwen3:4b

# The visual brain
contextmemory demo
contextmemory demo --live --model qwen3:4b

# An MCP server for external agents
contextmemory mcp --container brain

# Direct ingestion
contextmemory ingest --turn 'user:I moved to Seattle.'

# Measure the engine
contextmemory bench --system contextmemory
contextmemory dims --system contextmemory \
  --reader-api-base http://localhost:11434 \
  --reader-model qwen3:4b
```

## Python API

```python
from datetime import datetime

from contextmemory.api import MemoryClient
from contextmemory.eval.protocol import Session, Turn

brain = MemoryClient("user_123")

brain.session(Session(
    session_id="conversation-1",
    timestamp=datetime.now(),
    turns=[Turn(role="user", content="I moved to Seattle.")],
))

report = brain.recall("Where do I live?")
for hit in report.hits:
    print(hit.text)
```

Memory is scoped by container. The default is `brain`; use a different
`--container` for a user, repository, project, or agent.

## Install And Develop

Requirements: Python 3.11+, a C++ compiler, CMake, Ninja, and Ollama for live
model use.

```bash
uv sync
./scripts/verify.sh
```

The verification command runs the C++ checks, Python tests, and lint. The
deterministic suite does not need a model or network.

## Repository Map

```text
core/                 C++ ETMC memory engine
contextmemory/        Python API, Ollama, MCP, TUI, and evaluation
tests/                automated validation
benchmarks/           benchmark data and workloads
docs/                 architecture and research notes
reports/              historical runs and investigations
scripts/verify.sh     one-command verification
```

## The Point

More context is not the same thing as more memory.

Good memory is selective. It remembers what matters, notices when reality
changes, and knows when to stay quiet.

## License

Apache-2.0.

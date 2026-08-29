# ContextMemory

> Give your local model a memory that does not lie.

Your model is brilliant, fast, and forgetful. ContextMemory is the small,
durable brain that sits beside it: keeping facts, tracking change, finding the
right detail, and saying *I don't know* when the evidence is not there.

It is local-first, fast by construction, and pleasantly boring about your
data. No cloud memory service. No mystery database. Just a C++ temporal memory
engine with an Ollama chat, MCP tools, Python API, and a TUI you can watch
think.

## The One-Minute Start

Install the project, make sure Ollama is installed, and pull a model:

```bash
uv sync
ollama pull qwen3:4b
```

Start a normal terminal chat with ContextMemory already connected:

```bash
uv run contextmemory chat --model qwen3:4b
```

This is the whole product loop in one command:

1. Connects to the local Ollama server, or starts `ollama serve`.
2. Selects the requested model.
3. Starts the ContextMemory MCP tool host in-process.
4. Gives Ollama the `memory`, `recall`, `context`, and `forget` tools.
5. Persists memories locally for the next process.

Now chat exactly as usual. The difference is that the model has somewhere to
put the things worth keeping:

```text
ContextMemory chat | Ollama: qwen3:4b | MCP: connected

you> I live in Seattle and prefer Vim.
ollama> I will remember that.

you> Where do I live?
ollama> You live in Seattle.
```

Use `/exit` to quit. Omit `--model` to use the first installed model:

```bash
uv run contextmemory chat
```

The host routes durable statements through ContextMemory and retrieves memory
before user-history questions. Native Ollama tool calls are also supported.
Ollama is the voice; ContextMemory is the memory behind the voice.

## Give Any Agent A Memory

For OpenCode, Claude Code, Cursor, Cline, or another MCP client, register this
stdio server command:

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

If the project environment is already active, the equivalent command is:

```bash
contextmemory mcp --container brain
```

The MCP server exposes four tools:

- `memory(content)`: store a meaningful fact or conversation detail
- `recall(query)`: retrieve relevant memories
- `context(topic)`: retrieve compact working context
- `forget(id)`: request removal of a memory

MCP clients decide when to call tools. The direct `contextmemory chat` command
is the zero-configuration host when you want Ollama and memory in one terminal.

## Watch The Brain Work

Launch the interactive product shell:

```bash
uv run contextmemory demo
```

On the welcome screen, choose **Run Offline Demo**. The little deterministic
story is designed to make the idea click in under a minute:

- facts being stored
- current state retrieval
- a location and employer update
- historical truth after a contradiction
- abstention when memory does not contain an answer
- retrieval timing and token telemetry

The dashboard includes **RUN DEMO**, **ASK**, Brain, Timeline, Why, Models,
Retrieval Live, Performance, Connections, and Health. Press `O` to scan local
Ollama models and select one. Press `R` to replay the demo.

For a live model-backed TUI session:

```bash
uv run contextmemory demo --live --model qwen3:4b
```

The launcher also works from the repository root:

```bash
./run.sh                 # offline TUI
./run.sh --live          # live Ollama TUI
./run.sh --live --model qwen3:4b
```

## Under The Hood

```text
conversation or tool trace
        |
      capture       immutable episode
        |
      extract       one optional LLM pass into structured cells
        |
      reconcile     deterministic deduplication and versioning
        |
      recall        bounded query plan and search
        |
      pack          minimum sufficient evidence under a token budget
        |
      answer        Ollama or another compatible reader
```

The C++ ETMC core is the quiet machinery: capture, reconciliation, temporal
validity, projections, search, and evidence packing. The Python layer is the
front desk: API, Ollama integration, MCP bridge, TUI, and evaluation harness.

Memory journals are stored per container under the platform data directory.
The default container is `brain`; use `--container` to give each project or
user its own little mind.

## CLI

```bash
contextmemory chat --model qwen3:4b       # direct Ollama + memory chat
contextmemory demo                         # offline TUI
contextmemory demo --live --model qwen3:4b # live TUI
contextmemory mcp --container brain        # MCP stdio server
contextmemory ingest --turn 'user:fact'    # ingest a turn
contextmemory bench --system contextmemory # latency benchmark
contextmemory dims --system contextmemory \
  --reader-api-base http://localhost:11434 \
  --reader-model qwen3:4b
```

## Python API

```python
from contextmemory.api import MemoryClient
from contextmemory.eval.protocol import Session, Turn
from datetime import datetime

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

## Build, Test, Improve

Requirements: Python 3.11+, a C++ compiler, CMake, and Ninja. `uv sync`
installs the Python dependencies and builds the C++ extension.

Run the complete deterministic verification:

```bash
./scripts/verify.sh
```

This runs the C++ checks, Python tests, and lint. No model or network is
required for the deterministic suite.

## Repository Map

```text
core/                 C++ ETMC memory engine
contextmemory/        Python API, Ollama, MCP, TUI, and evaluation
tests/                automated tests
benchmarks/           benchmark data and workloads
docs/                 architecture and research notes
reports/              historical runs and investigations
scripts/verify.sh     repository verification
```

## License

Apache-2.0.

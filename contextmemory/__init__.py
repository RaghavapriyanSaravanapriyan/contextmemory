"""ContextMemory: a better persistent memory and context optimization layer."""

from .api import MemoryClient, from_reader_client

__all__ = ["MemoryClient", "from_reader_client"]
__version__ = "0.1.0"
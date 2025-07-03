# chuk_tool_registry/providers/memory.py
"""
Improved in-memory implementation of the asynchronous tool registry.

Key improvements:
1. Better type safety and error handling
2. Improved metadata management
3. Cleaner code organization
4. Enhanced validation and consistency checks
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict

# registry
from chuk_tool_registry.core.exceptions import ToolNotFoundError
from chuk_tool_registry.core.interface import ToolRegistryInterface
from chuk_tool_registry.core.metadata import ToolMetadata

logger = logging.getLogger(__name__)


class RegistryStatistics:
    """
    Statistics and metrics for the registry.
    """
    def __init__(self):
        self.tool_count = 0
        self.namespace_count = 0
        self.registrations_count = 0
        self.last_registration_time: Optional[float] = None
    
    def update_on_registration(self, namespace: str, tool_count: int, namespace_count: int):
        """Update statistics after a tool registration."""
        import time
        self.tool_count = tool_count
        self.namespace_count = namespace_count
        self.registrations_count += 1
        self.last_registration_time = time.time()


class InMemoryToolRegistry(ToolRegistryInterface):
    """
    Improved in-memory implementation of the async ToolRegistryInterface.

    Features:
    - Namespace-based organization
    - Thread-safe async operations
    - Rich metadata management
    - Tool validation
    - Statistics tracking
    - Comprehensive error handling
    """

    def __init__(self, *, enable_statistics: bool = True, validate_tools: bool = True):
        """
        Initialize the registry.
        
        Args:
            enable_statistics: Whether to track registry statistics
            validate_tools: Whether to validate tools during registration
        """
        # Core storage: {namespace: {tool_name: tool_obj}}
        self._tools: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Metadata storage: {namespace: {tool_name: ToolMetadata}}
        self._metadata: Dict[str, Dict[str, ToolMetadata]] = defaultdict(dict)
        
        # Configuration
        self._enable_statistics = enable_statistics
        self._validate_tools = validate_tools
        
        # Statistics tracking
        self._stats = RegistryStatistics() if enable_statistics else None
        
        # Thread safety
        self._lock = asyncio.Lock()
        
        logger.debug("InMemoryToolRegistry initialized")

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    async def register_tool(
        self,
        tool: Any,
        name: Optional[str] = None,
        namespace: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a tool in the registry with comprehensive validation.
        
        Args:
            tool: The tool object to register
            name: Optional tool name (derived from tool if not provided)
            namespace: Namespace for the tool
            metadata: Optional additional metadata
            
        Raises:
            ValueError: If tool validation fails
            TypeError: If tool doesn't meet interface requirements
        """
        async with self._lock:
            # Determine tool name
            tool_name = self._determine_tool_name(tool, name)
            
            # Validate tool if enabled
            if self._validate_tools:
                self._validate_tool(tool, tool_name)
            
            # Store the tool
            self._tools[namespace][tool_name] = tool
            
            # Create and store metadata
            tool_metadata = await self._create_tool_metadata(
                tool, tool_name, namespace, metadata
            )
            self._metadata[namespace][tool_name] = tool_metadata
            
            # Update statistics
            if self._stats:
                total_tools = sum(len(tools) for tools in self._tools.values())
                self._stats.update_on_registration(
                    namespace, total_tools, len(self._tools)
                )
            
            logger.debug(f"Registered tool '{namespace}.{tool_name}'")

    def _determine_tool_name(self, tool: Any, name: Optional[str]) -> str:
        """Determine the name to use for a tool."""
        if name:
            return name
        
        # Try various ways to get a name from the tool
        candidates = [
            getattr(tool, 'name', None),
            getattr(tool, 'tool_name', None),
            getattr(tool, '__name__', None),
            tool.__class__.__name__ if hasattr(tool, '__class__') else None,
        ]
        
        for candidate in candidates:
            if candidate and isinstance(candidate, str):
                return candidate
        
        # Fallback to representation
        return repr(tool)

    def _validate_tool(self, tool: Any, tool_name: str) -> None:
        """
        Validate that a tool meets the expected interface.
        
        Args:
            tool: Tool to validate
            tool_name: Name of the tool for error messages
            
        Raises:
            TypeError: If tool doesn't meet requirements
        """
        # Check if tool has execute method
        if not hasattr(tool, 'execute'):
            raise TypeError(f"Tool '{tool_name}' must have an 'execute' method")
        
        execute_method = getattr(tool, 'execute')
        if not callable(execute_method):
            raise TypeError(f"Tool '{tool_name}'.execute must be callable")
        
        # Check if execute method is async (recommended for async-native architecture)
        if not inspect.iscoroutinefunction(execute_method):
            logger.warning(
                f"Tool '{tool_name}'.execute is not async. "
                "Consider making it async for better performance."
            )

    async def _create_tool_metadata(
        self, 
        tool: Any, 
        name: str, 
        namespace: str, 
        metadata: Optional[Dict[str, Any]]
    ) -> ToolMetadata:
        """
        Create comprehensive metadata for a tool.
        
        Args:
            tool: The tool object
            name: Tool name
            namespace: Tool namespace
            metadata: Optional additional metadata
            
        Returns:
            ToolMetadata instance
        """
        # Check if tool has execute method and if it's async
        has_execute = hasattr(tool, 'execute')
        is_async = has_execute and inspect.iscoroutinefunction(getattr(tool, 'execute'))
        
        # Extract description from various sources
        description = None
        if metadata and 'description' in metadata:
            description = metadata['description']
        elif hasattr(tool, 'description'):
            description = getattr(tool, 'description')
        elif hasattr(tool, '__doc__') and tool.__doc__:
            description = inspect.getdoc(tool)
        
        # Build metadata dictionary
        meta_dict: Dict[str, Any] = {
            "name": name,
            "namespace": namespace,
            "is_async": is_async,
        }
        
        if description:
            meta_dict["description"] = description.strip()
        
        # Add custom metadata, storing complex fields in execution_options
        if metadata:
            # Extract core metadata fields that are direct ToolMetadata attributes
            core_fields = ["version", "source", "source_name", "requires_auth", "tags", 
                          "concurrency_limit", "timeout", "rate_limit", "supports_streaming",
                          "argument_schema", "result_schema", "dependencies"]
            
            for key in core_fields:
                if key in metadata:
                    meta_dict[key] = metadata[key]
            
            # Store remaining metadata in execution_options
            execution_options = {}
            for key, value in metadata.items():
                if key not in core_fields and key != "description":
                    execution_options[key] = value
            
            if execution_options:
                meta_dict["execution_options"] = execution_options
        
        # Add tool introspection data to execution_options
        introspection_data = {
            "tool_type": type(tool).__name__,
            "tool_module": getattr(type(tool), '__module__', 'unknown'),
            "has_execute_method": has_execute,
        }
        
        if "execution_options" not in meta_dict:
            meta_dict["execution_options"] = {}
        meta_dict["execution_options"].update(introspection_data)
        
        return ToolMetadata(**meta_dict)

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    async def get_tool(self, name: str, namespace: str = "default") -> Optional[Any]:
        """
        Retrieve a tool by name and namespace.
        
        Args:
            name: Tool name
            namespace: Tool namespace
            
        Returns:
            Tool object if found, None otherwise
        """
        # No lock needed for read operations
        return self._tools.get(namespace, {}).get(name)

    async def get_tool_strict(self, name: str, namespace: str = "default") -> Any:
        """
        Get a tool with strict validation, raising if not found.
        
        Args:
            name: Tool name
            namespace: Tool namespace
            
        Returns:
            Tool object
            
        Raises:
            ToolNotFoundError: If tool is not found
        """
        tool = await self.get_tool(name, namespace)
        if tool is None:
            raise ToolNotFoundError(f"{namespace}.{name}")
        return tool

    async def get_metadata(
        self, name: str, namespace: str = "default"
    ) -> Optional[ToolMetadata]:
        """
        Get metadata for a tool.
        
        Args:
            name: Tool name
            namespace: Tool namespace
            
        Returns:
            ToolMetadata if found, None otherwise
        """
        return self._metadata.get(namespace, {}).get(name)

    # ------------------------------------------------------------------ #
    # Listing and Discovery
    # ------------------------------------------------------------------ #

    async def list_tools(self, namespace: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        Return a list of (namespace, name) tuples.
        
        Args:
            namespace: Optional namespace filter
            
        Returns:
            List of (namespace, tool_name) tuples
        """
        if namespace is not None:
            # Return tools from specific namespace
            tools = self._tools.get(namespace, {})
            return [(namespace, name) for name in tools.keys()]

        # Return tools from all namespaces
        result: List[Tuple[str, str]] = []
        for ns, tools in self._tools.items():
            result.extend((ns, name) for name in tools.keys())
        return result

    async def list_namespaces(self) -> List[str]:
        """
        List all namespaces.
        
        Returns:
            List of namespace names
        """
        return list(self._tools.keys())

    async def list_metadata(self, namespace: Optional[str] = None) -> List[ToolMetadata]:
        """
        Return all ToolMetadata objects.

        Args:
            namespace: Optional namespace filter
                • None (default) - metadata from all namespaces
                • "some_ns" - only that namespace

        Returns:
            List of ToolMetadata objects
        """
        if namespace is not None:
            return list(self._metadata.get(namespace, {}).values())

        # Flatten all metadata
        result: List[ToolMetadata] = []
        for ns_meta in self._metadata.values():
            result.extend(ns_meta.values())
        return result

    # ------------------------------------------------------------------ #
    # Management and Utility Methods
    # ------------------------------------------------------------------ #

    async def remove_tool(self, name: str, namespace: str = "default") -> bool:
        """
        Remove a tool from the registry.
        
        Args:
            name: Tool name
            namespace: Tool namespace
            
        Returns:
            True if tool was removed, False if not found
        """
        async with self._lock:
            namespace_tools = self._tools.get(namespace, {})
            namespace_metadata = self._metadata.get(namespace, {})
            
            if name in namespace_tools:
                del namespace_tools[name]
                namespace_metadata.pop(name, None)
                
                # Clean up empty namespaces
                if not namespace_tools:
                    self._tools.pop(namespace, None)
                    self._metadata.pop(namespace, None)
                
                logger.debug(f"Removed tool '{namespace}.{name}'")
                return True
            
            return False

    async def clear_namespace(self, namespace: str) -> int:
        """
        Clear all tools from a namespace.
        
        Args:
            namespace: Namespace to clear
            
        Returns:
            Number of tools removed
        """
        async with self._lock:
            namespace_tools = self._tools.get(namespace, {})
            count = len(namespace_tools)
            
            if count > 0:
                self._tools.pop(namespace, None)
                self._metadata.pop(namespace, None)
                logger.debug(f"Cleared namespace '{namespace}' ({count} tools)")
            
            return count

    async def clear_all(self) -> int:
        """
        Clear all tools from the registry.
        
        Returns:
            Total number of tools removed
        """
        async with self._lock:
            total_count = sum(len(tools) for tools in self._tools.values())
            self._tools.clear()
            self._metadata.clear()
            
            if self._stats:
                self._stats.tool_count = 0
                self._stats.namespace_count = 0
            
            logger.debug(f"Cleared all tools ({total_count} tools)")
            return total_count

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with registry statistics
        """
        stats = {
            "total_tools": sum(len(tools) for tools in self._tools.values()),
            "total_namespaces": len(self._tools),
            "namespaces": {
                ns: len(tools) for ns, tools in self._tools.items()
            },
            "statistics_enabled": self._enable_statistics,
            "validation_enabled": self._validate_tools,
        }
        
        if self._stats:
            stats.update({
                "registrations_count": self._stats.registrations_count,
                "last_registration_time": self._stats.last_registration_time,
            })
        
        return stats

    async def tool_exists(self, name: str, namespace: str = "default") -> bool:
        """
        Check if a tool exists in the registry.
        
        Args:
            name: Tool name
            namespace: Tool namespace
            
        Returns:
            True if tool exists, False otherwise
        """
        return name in self._tools.get(namespace, {})

    async def search_tools(
        self, 
        query: str, 
        namespace: Optional[str] = None,
        search_descriptions: bool = True
    ) -> List[ToolMetadata]:
        """
        Search for tools by name or description.
        
        Args:
            query: Search query
            namespace: Optional namespace filter
            search_descriptions: Whether to search in descriptions
            
        Returns:
            List of matching ToolMetadata objects
        """
        query_lower = query.lower()
        matches = []
        
        metadata_list = await self.list_metadata(namespace)
        
        for metadata in metadata_list:
            # Search in name
            if query_lower in metadata.name.lower():
                matches.append(metadata)
                continue
            
            # Search in description if enabled
            if search_descriptions and metadata.description:
                if query_lower in metadata.description.lower():
                    matches.append(metadata)
        
        return matches

    # ------------------------------------------------------------------ #
    # String representation
    # ------------------------------------------------------------------ #

    def __str__(self) -> str:
        """String representation of the registry."""
        total_tools = sum(len(tools) for tools in self._tools.values())
        namespace_count = len(self._tools)
        return f"InMemoryToolRegistry(tools={total_tools}, namespaces={namespace_count})"

    def __repr__(self) -> str:
        """Detailed representation of the registry."""
        return (
            f"InMemoryToolRegistry("
            f"tools={sum(len(tools) for tools in self._tools.values())}, "
            f"namespaces={list(self._tools.keys())}, "
            f"statistics_enabled={self._enable_statistics}, "
            f"validation_enabled={self._validate_tools})"
        )
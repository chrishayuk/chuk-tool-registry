# chuk_tool_registry/core/interface.py
"""
Enhanced interface for asynchronous tool registries with validation support.
"""
from __future__ import annotations

from typing import Protocol, Any, Dict, List, Optional, Tuple, TypeVar, runtime_checkable

# imports
from .metadata import ToolMetadata
from .validation import ValidationConfig

T = TypeVar('T')

@runtime_checkable
class ToolRegistryInterface(Protocol):
    """
    Enhanced protocol for an async tool registry with validation support.
    
    Implementations should allow registering tools and retrieving them by name and namespace,
    with optional validation capabilities for type safety and data integrity.
    """
    
    # ================================================================== #
    # Core Registration Methods
    # ================================================================== #
    
    async def register_tool(
        self, 
        tool: Any, 
        name: Optional[str] = None,
        namespace: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        validation_config: Optional[ValidationConfig] = None,
        enable_validation: Optional[bool] = None
    ) -> None:
        """
        Register a tool implementation asynchronously with optional validation.

        Args:
            tool: The tool class or instance with an `execute` method.
            name: Optional explicit name; if omitted, uses tool.__name__.
            namespace: Namespace for the tool (default: "default").
            metadata: Optional additional metadata for the tool.
            validation_config: Optional validation configuration for this tool.
            enable_validation: Whether to enable validation for this specific tool.
        """
        ...

    # ================================================================== #
    # Core Retrieval Methods
    # ================================================================== #

    async def get_tool(self, name: str, namespace: str = "default") -> Optional[Any]:
        """
        Retrieve a registered tool by name and namespace asynchronously.
        
        Returns the validation-wrapped version if validation is enabled for the tool.
        
        Args:
            name: The name of the tool.
            namespace: The namespace of the tool (default: "default").
            
        Returns:
            The tool implementation (potentially validation-wrapped) or None if not found.
        """
        ...

    async def get_tool_strict(self, name: str, namespace: str = "default") -> Any:
        """
        Retrieve a registered tool by name and namespace, raising if not found.
        
        Args:
            name: The name of the tool.
            namespace: The namespace of the tool (default: "default").
            
        Returns:
            The tool implementation (potentially validation-wrapped).
            
        Raises:
            ToolNotFoundError: If the tool is not found in the registry.
        """
        ...

    async def get_metadata(self, name: str, namespace: str = "default") -> Optional[ToolMetadata]:
        """
        Retrieve metadata for a registered tool asynchronously.
        
        Args:
            name: The name of the tool.
            namespace: The namespace of the tool (default: "default").
            
        Returns:
            ToolMetadata if found, None otherwise.
        """
        ...

    # ================================================================== #
    # Listing and Discovery Methods
    # ================================================================== #

    async def list_tools(self, namespace: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        List all registered tool names asynchronously, optionally filtered by namespace.
        
        Args:
            namespace: Optional namespace filter.
            
        Returns:
            List of (namespace, name) tuples.
        """
        ...

    async def list_namespaces(self) -> List[str]:
        """
        List all registered namespaces asynchronously.
        
        Returns:
            List of namespace names.
        """
        ...
        
    async def list_metadata(self, namespace: Optional[str] = None) -> List[ToolMetadata]:
        """
        Return all ToolMetadata objects asynchronously.

        Args:
            namespace: Optional filter by namespace.
                • None (default) - metadata from all namespaces
                • "some_ns" - only that namespace

        Returns:
            List of ToolMetadata objects.
        """
        ...

    # ================================================================== #
    # Validation-Specific Methods (Required for Tests)
    # ================================================================== #

    async def get_original_tool(self, name: str, namespace: str = "default") -> Optional[Any]:
        """
        Get the original tool without validation wrapper.
        
        Args:
            name: Tool name
            namespace: Tool namespace
            
        Returns:
            Original tool object if found, None otherwise
        """
        ...

    async def get_validation_config(
        self, name: str, namespace: str = "default"
    ) -> Optional[ValidationConfig]:
        """
        Get validation configuration for a tool.
        
        Args:
            name: Tool name
            namespace: Tool namespace
            
        Returns:
            ValidationConfig if tool has validation enabled, None otherwise
        """
        ...

    async def execute_tool_with_validation(
        self, 
        name: str, 
        namespace: str = "default", 
        **kwargs
    ) -> Any:
        """
        Execute a tool with integrated validation and error handling.
        
        Args:
            name: Tool name
            namespace: Tool namespace
            **kwargs: Arguments to pass to the tool
            
        Returns:
            Tool execution result
            
        Raises:
            ToolNotFoundError: If tool is not found
            ToolValidationError: If validation fails
            ToolExecutionError: If execution fails
        """
        ...

    async def list_validation_enabled_tools(self, namespace: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        List tools that have validation enabled.
        
        Args:
            namespace: Optional namespace filter
            
        Returns:
            List of (namespace, tool_name) tuples for validation-enabled tools
        """
        ...

    async def tool_has_validation(self, name: str, namespace: str = "default") -> bool:
        """
        Check if a tool has validation enabled.
        
        Args:
            name: Tool name
            namespace: Tool namespace
            
        Returns:
            True if tool has validation enabled, False otherwise
        """
        ...

    # ================================================================== #
    # Management and Utility Methods
    # ================================================================== #

    async def remove_tool(self, name: str, namespace: str = "default") -> bool:
        """
        Remove a tool from the registry.
        
        Args:
            name: Tool name
            namespace: Tool namespace
            
        Returns:
            True if tool was removed, False if not found
        """
        ...

    async def clear_namespace(self, namespace: str) -> int:
        """
        Clear all tools from a namespace.
        
        Args:
            namespace: Namespace to clear
            
        Returns:
            Number of tools removed
        """
        ...

    async def clear_all(self) -> int:
        """
        Clear all tools from the registry.
        
        Returns:
            Total number of tools removed
        """
        ...

    async def tool_exists(self, name: str, namespace: str = "default") -> bool:
        """
        Check if a tool exists in the registry.
        
        Args:
            name: Tool name
            namespace: Tool namespace
            
        Returns:
            True if tool exists, False otherwise
        """
        ...

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with registry statistics
        """
        ...

    async def search_tools(
        self, 
        query: str, 
        namespace: Optional[str] = None,
        search_descriptions: bool = True,
        validation_only: bool = False
    ) -> List[ToolMetadata]:
        """
        Search for tools by name or description.
        
        Args:
            query: Search query
            namespace: Optional namespace filter
            search_descriptions: Whether to search in descriptions
            validation_only: If True, only return validation-enabled tools
            
        Returns:
            List of matching ToolMetadata objects
        """
        ...

    async def validate_all_tools(self) -> Dict[str, Any]:
        """
        Validate all tools in the registry for consistency.
        
        Returns:
            Validation report with any issues found
        """
        ...


# Helper class that provides default implementations for tests
class DefaultToolRegistryMixin:
    """Mixin class that provides default implementations for optional interface methods."""
    
    async def get_original_tool(self, name: str, namespace: str = "default") -> Optional[Any]:
        """Default implementation falls back to get_tool."""
        return await self.get_tool(name, namespace)

    async def get_validation_config(self, name: str, namespace: str = "default") -> Optional[ValidationConfig]:
        """Default implementation returns None (no validation)."""
        return None

    async def execute_tool_with_validation(self, name: str, namespace: str = "default", **kwargs) -> Any:
        """Default implementation: get tool and execute."""
        tool = await self.get_tool_strict(name, namespace)
        return await tool.execute(**kwargs)

    async def list_validation_enabled_tools(self, namespace: Optional[str] = None) -> List[Tuple[str, str]]:
        """Default implementation returns empty list."""
        return []

    async def tool_has_validation(self, name: str, namespace: str = "default") -> bool:
        """Default implementation returns False."""
        return False

    async def remove_tool(self, name: str, namespace: str = "default") -> bool:
        """Default implementation indicates operation not supported."""
        return False

    async def clear_namespace(self, namespace: str) -> int:
        """Default implementation indicates operation not supported."""
        return 0

    async def clear_all(self) -> int:
        """Default implementation indicates operation not supported."""
        return 0

    async def tool_exists(self, name: str, namespace: str = "default") -> bool:
        """Default implementation using get_tool."""
        tool = await self.get_tool(name, namespace)
        return tool is not None

    async def get_statistics(self) -> Dict[str, Any]:
        """Default implementation provides basic stats."""
        tools = await self.list_tools()
        namespaces = await self.list_namespaces()
        
        return {
            "total_tools": len(tools),
            "total_namespaces": len(namespaces),
            "namespaces": {
                ns: len([t for t in tools if t[0] == ns]) 
                for ns in namespaces
            },
            "statistics_enabled": False,
            "validation_available": False,
        }

    async def search_tools(
        self, 
        query: str, 
        namespace: Optional[str] = None,
        search_descriptions: bool = True,
        validation_only: bool = False
    ) -> List[ToolMetadata]:
        """Default implementation provides basic search."""
        query_lower = query.lower()
        matches = []
        
        metadata_list = await self.list_metadata(namespace)
        
        for metadata in metadata_list:
            # Skip if we only want validation-enabled tools
            if validation_only and not await self.tool_has_validation(metadata.name, metadata.namespace):
                continue
            
            # Search in name
            if query_lower in metadata.name.lower():
                matches.append(metadata)
                continue
            
            # Search in description if enabled
            if search_descriptions and metadata.description:
                if query_lower in metadata.description.lower():
                    matches.append(metadata)
        
        return matches

    async def validate_all_tools(self) -> Dict[str, Any]:
        """Default implementation provides basic validation."""
        tools = await self.list_tools()
        issues = []
        
        for namespace, tool_name in tools:
            # Check if tool can be retrieved
            tool = await self.get_tool(tool_name, namespace)
            if tool is None:
                issues.append(f"{namespace}.{tool_name}: Tool retrieval failed")
                continue
            
            # Check if tool has execute method
            if not hasattr(tool, 'execute'):
                issues.append(f"{namespace}.{tool_name}: Missing execute method")
            
            # Check if metadata exists
            metadata = await self.get_metadata(tool_name, namespace)
            if metadata is None:
                issues.append(f"{namespace}.{tool_name}: Missing metadata")
        
        return {
            "total_tools": len(tools),
            "validation_enabled_tools": 0,
            "issues_found": len(issues),
            "issues": issues,
            "validation_percentage": 0.0,
            "registry_healthy": len(issues) == 0
        }


# ================================================================== #
# Type Aliases and Helper Types
# ================================================================== #

# Type alias for the enhanced interface
EnhancedToolRegistry = ToolRegistryInterface

# Type alias for validation-enabled registries
ValidatedToolRegistry = ToolRegistryInterface
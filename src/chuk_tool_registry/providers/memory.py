# chuk_tool_registry/providers/memory.py
"""
Enhanced in-memory implementation with integrated validation support.

Key improvements:
1. Integrated validation system
2. Configurable validation per tool
3. Enhanced error handling with validation
4. Better performance tracking
5. Registry-level validation configuration
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from collections import defaultdict

# Core imports
from chuk_tool_registry.core.exceptions import ToolNotFoundError, ToolValidationError
from chuk_tool_registry.core.interface import ToolRegistryInterface
from chuk_tool_registry.core.metadata import ToolMetadata
from chuk_tool_registry.core.validation import (
    ValidationConfig, 
    validate_tool_execution, 
    create_validation_wrapper
)

logger = logging.getLogger(__name__)


class ValidationWrapper:
    """Wrapper that adds validation to a tool while preserving its interface."""
    
    def __init__(self, original_tool: Any, wrapped_execute: callable, tool_name: str):
        self.original_tool = original_tool
        self.wrapped_execute = wrapped_execute
        self.tool_name = tool_name
        
        # Store reference to original execute method for signature introspection
        self.original_execute = getattr(original_tool, 'execute', None)
        
        # Copy attributes from original tool
        for attr in dir(original_tool):
            if not attr.startswith('_') and attr != 'execute':
                try:
                    setattr(self, attr, getattr(original_tool, attr))
                except (AttributeError, TypeError):
                    pass
    
    async def execute(self, **kwargs):
        """Execute with validation wrapper."""
        return await self.wrapped_execute(**kwargs)
    
    def __getattr__(self, name):
        """Delegate attribute access to original tool."""
        return getattr(self.original_tool, name)


class RegistryStatistics:
    """Enhanced statistics tracking for the registry."""
    
    def __init__(self):
        self.tool_count = 0
        self.namespace_count = 0
        self.registrations_count = 0
        self.executions_count = 0
        self.validation_errors_count = 0
        self.last_registration_time: Optional[float] = None
        self.last_execution_time: Optional[float] = None
        self.validation_enabled_tools = 0
    
    def update_on_registration(self, namespace: str, tool_count: int, namespace_count: int, validation_enabled: bool = False):
        """Update statistics after a tool registration."""
        import time
        self.tool_count = tool_count
        self.namespace_count = namespace_count
        self.registrations_count += 1
        self.last_registration_time = time.time()
        if validation_enabled:
            self.validation_enabled_tools += 1
    
    def update_on_execution(self, validation_error: bool = False):
        """Update statistics after a tool execution."""
        import time
        self.executions_count += 1
        self.last_execution_time = time.time()
        if validation_error:
            self.validation_errors_count += 1


class InMemoryToolRegistry(ToolRegistryInterface):
    """
    Enhanced in-memory implementation with integrated validation support.

    Features:
    - Namespace-based organization
    - Thread-safe async operations
    - Rich metadata management
    - Integrated validation system
    - Tool validation and execution tracking
    - Configurable validation per tool
    - Performance monitoring
    """

    def __init__(
        self, 
        *, 
        enable_statistics: bool = True, 
        validate_tools: bool = True,
        default_validation_config: Optional[ValidationConfig] = None,
        enable_validation_by_default: bool = False
    ):
        """
        Initialize the registry with validation support.
        
        Args:
            enable_statistics: Whether to track registry statistics
            validate_tools: Whether to validate tools during registration
            default_validation_config: Default validation configuration
            enable_validation_by_default: Whether to enable validation for all tools
        """
        # Core storage: {namespace: {tool_name: tool_obj}}
        self._tools: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Metadata storage: {namespace: {tool_name: ToolMetadata}}
        self._metadata: Dict[str, Dict[str, ToolMetadata]] = defaultdict(dict)
        
        # Validation configuration per tool: {namespace: {tool_name: ValidationConfig}}
        self._validation_configs: Dict[str, Dict[str, ValidationConfig]] = defaultdict(dict)
        
        # Wrapped tools for validation: {namespace: {tool_name: wrapped_tool}}
        self._validation_wrappers: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Configuration
        self._enable_statistics = enable_statistics
        self._validate_tools = validate_tools
        self._default_validation_config = default_validation_config or ValidationConfig()
        self._enable_validation_by_default = enable_validation_by_default
        
        # Statistics tracking
        self._stats = RegistryStatistics() if enable_statistics else None
        
        # Thread safety
        self._lock = asyncio.Lock()
        
        logger.debug(f"InMemoryToolRegistry initialized with validation support")

    # ------------------------------------------------------------------ #
    # Enhanced Registration with Validation
    # ------------------------------------------------------------------ #

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
        Register a tool with optional validation configuration.
        
        Args:
            tool: The tool object to register
            name: Optional tool name (derived from tool if not provided)
            namespace: Namespace for the tool
            metadata: Optional additional metadata
            validation_config: Optional validation configuration for this tool
            enable_validation: Whether to enable validation for this specific tool
            
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
            
            # Determine validation settings
            should_validate = self._should_enable_validation(enable_validation, metadata)
            validation_cfg = validation_config or self._default_validation_config
            
            # Store the original tool
            self._tools[namespace][tool_name] = tool
            
            # Create validation wrapper if needed
            if should_validate:
                try:
                    execute_method = getattr(tool, 'execute', None)
                    if execute_method and callable(execute_method):
                        # Create a custom wrapper that preserves the original method for validation
                        async def validated_execute_wrapper(**kwargs):
                            from chuk_tool_registry.core.validation import validate_arguments, validate_result
                            
                            # Use the original execute method for signature introspection
                            if validation_cfg.validate_arguments:
                                kwargs = validate_arguments(tool_name, execute_method, kwargs, validation_cfg)
                            
                            # Execute the original method
                            result = await execute_method(**kwargs)
                            
                            # Validate result if enabled
                            if validation_cfg.validate_results:
                                result = validate_result(tool_name, execute_method, result, validation_cfg)
                            
                            return result
                        
                        # Create a wrapper object that preserves the tool interface
                        validation_wrapper = ValidationWrapper(tool, validated_execute_wrapper, tool_name)
                        self._validation_wrappers[namespace][tool_name] = validation_wrapper
                        self._validation_configs[namespace][tool_name] = validation_cfg
                        
                        logger.debug(f"Created validation wrapper for '{namespace}.{tool_name}'")
                except Exception as e:
                    logger.warning(f"Failed to create validation wrapper for '{namespace}.{tool_name}': {e}")
            
            # Create and store metadata
            tool_metadata = await self._create_tool_metadata(
                tool, tool_name, namespace, metadata, should_validate, validation_cfg
            )
            self._metadata[namespace][tool_name] = tool_metadata
            
            # Update statistics
            if self._stats:
                total_tools = sum(len(tools) for tools in self._tools.values())
                self._stats.update_on_registration(
                    namespace, total_tools, len(self._tools), should_validate
                )
            
            logger.debug(f"Registered tool '{namespace}.{tool_name}' (validation: {should_validate})")

    def _should_enable_validation(self, enable_validation: Optional[bool], metadata: Optional[Dict[str, Any]]) -> bool:
        """Determine if validation should be enabled for a tool."""
        # Explicit parameter takes precedence
        if enable_validation is not None:
            return enable_validation
        
        # Check metadata for validation flag
        if metadata and 'enable_validation' in metadata:
            return bool(metadata['enable_validation'])
        
        # Fall back to registry default
        return self._enable_validation_by_default

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
        metadata: Optional[Dict[str, Any]],
        validation_enabled: bool,
        validation_config: ValidationConfig
    ) -> ToolMetadata:
        """
        Create comprehensive metadata for a tool including validation info.
        
        Args:
            tool: The tool object
            name: Tool name
            namespace: Tool namespace
            metadata: Optional additional metadata
            validation_enabled: Whether validation is enabled for this tool
            validation_config: Validation configuration used
            
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
                if key not in core_fields and key not in ["description", "enable_validation"]:
                    execution_options[key] = value
            
            if execution_options:
                meta_dict["execution_options"] = execution_options
        
        # Add tool introspection data to execution_options
        introspection_data = {
            "tool_type": type(tool).__name__,
            "tool_module": getattr(type(tool), '__module__', 'unknown'),
            "has_execute_method": has_execute,
            "validation_enabled": validation_enabled,
        }
        
        # Add validation configuration details
        if validation_enabled:
            introspection_data.update({
                "validation_config": {
                    "validate_arguments": validation_config.validate_arguments,
                    "validate_results": validation_config.validate_results,
                    "strict_mode": validation_config.strict_mode,
                    "allow_extra_args": validation_config.allow_extra_args,
                    "coerce_types": validation_config.coerce_types,
                }
            })
        
        if "execution_options" not in meta_dict:
            meta_dict["execution_options"] = {}
        meta_dict["execution_options"].update(introspection_data)
        
        return ToolMetadata(**meta_dict)

    # ------------------------------------------------------------------ #
    # Enhanced Retrieval with Validation Support
    # ------------------------------------------------------------------ #

    async def get_tool(self, name: str, namespace: str = "default") -> Optional[Any]:
        """
        Retrieve a tool by name and namespace, returning validation wrapper if available.
        
        Args:
            name: Tool name
            namespace: Tool namespace
            
        Returns:
            Tool object (potentially validation-wrapped) if found, None otherwise
        """
        # Check for validation wrapper first
        if name in self._validation_wrappers.get(namespace, {}):
            return self._validation_wrappers[namespace][name]
        
        # Fall back to original tool
        return self._tools.get(namespace, {}).get(name)

    async def get_tool_strict(self, name: str, namespace: str = "default") -> Any:
        """
        Get a tool with strict validation, raising if not found.
        
        Args:
            name: Tool name
            namespace: Tool namespace
            
        Returns:
            Tool object (potentially validation-wrapped)
            
        Raises:
            ToolNotFoundError: If tool is not found
        """
        tool = await self.get_tool(name, namespace)
        if tool is None:
            raise ToolNotFoundError(f"{namespace}.{name}")
        return tool

    async def get_original_tool(self, name: str, namespace: str = "default") -> Optional[Any]:
        """
        Get the original tool without validation wrapper.
        
        Args:
            name: Tool name
            namespace: Tool namespace
            
        Returns:
            Original tool object if found, None otherwise
        """
        return self._tools.get(namespace, {}).get(name)

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
        return self._validation_configs.get(namespace, {}).get(name)

    # ------------------------------------------------------------------ #
    # Enhanced Execution with Validation
    # ------------------------------------------------------------------ #

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
        try:
            # Get the tool (validation wrapper if available)
            tool = await self.get_tool_strict(name, namespace)
            
            # Execute the tool
            result = await tool.execute(**kwargs)
            
            # Update statistics
            if self._stats:
                self._stats.update_on_execution(validation_error=False)
            
            return result
            
        except ToolValidationError as e:
            # Update statistics for validation error
            if self._stats:
                self._stats.update_on_execution(validation_error=True)
            raise
        except Exception as e:
            # Update statistics for general execution error
            if self._stats:
                self._stats.update_on_execution(validation_error=False)
            # Re-raise as ToolExecutionError if not already a tool error
            if not isinstance(e, (ToolNotFoundError, ToolValidationError)):
                from chuk_tool_registry.core.exceptions import ToolExecutionError
                raise ToolExecutionError(f"{namespace}.{name}", e) from e
            raise

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

    async def list_validation_enabled_tools(self, namespace: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        List tools that have validation enabled.
        
        Args:
            namespace: Optional namespace filter
            
        Returns:
            List of (namespace, tool_name) tuples for validation-enabled tools
        """
        result = []
        
        if namespace is not None:
            # Check specific namespace
            for tool_name in self._validation_wrappers.get(namespace, {}):
                result.append((namespace, tool_name))
        else:
            # Check all namespaces
            for ns, tools in self._validation_wrappers.items():
                for tool_name in tools:
                    result.append((ns, tool_name))
        
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
            namespace_wrappers = self._validation_wrappers.get(namespace, {})
            namespace_configs = self._validation_configs.get(namespace, {})
            
            if name in namespace_tools:
                del namespace_tools[name]
                namespace_metadata.pop(name, None)
                namespace_wrappers.pop(name, None)
                namespace_configs.pop(name, None)
                
                # Clean up empty namespaces
                if not namespace_tools:
                    self._tools.pop(namespace, None)
                    self._metadata.pop(namespace, None)
                    self._validation_wrappers.pop(namespace, None)
                    self._validation_configs.pop(namespace, None)
                
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
                self._validation_wrappers.pop(namespace, None)
                self._validation_configs.pop(namespace, None)
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
            self._validation_wrappers.clear()
            self._validation_configs.clear()
            
            if self._stats:
                self._stats.tool_count = 0
                self._stats.namespace_count = 0
                self._stats.validation_enabled_tools = 0
            
            logger.debug(f"Cleared all tools ({total_count} tools)")
            return total_count

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get enhanced registry statistics including validation info.
        
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
            "validation_available": True,  # Always available now
            "validation_enabled_by_default": self._enable_validation_by_default,
            "tools_with_validation": sum(len(wrappers) for wrappers in self._validation_wrappers.values()),
        }
        
        if self._stats:
            stats.update({
                "registrations_count": self._stats.registrations_count,
                "executions_count": self._stats.executions_count,
                "validation_errors_count": self._stats.validation_errors_count,
                "last_registration_time": self._stats.last_registration_time,
                "last_execution_time": self._stats.last_execution_time,
                "validation_enabled_tools": self._stats.validation_enabled_tools,
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

    async def tool_has_validation(self, name: str, namespace: str = "default") -> bool:
        """
        Check if a tool has validation enabled.
        
        Args:
            name: Tool name
            namespace: Tool namespace
            
        Returns:
            True if tool has validation enabled, False otherwise
        """
        return name in self._validation_wrappers.get(namespace, {})

    async def search_tools(
        self, 
        query: str, 
        namespace: Optional[str] = None,
        search_descriptions: bool = True,
        validation_only: bool = False
    ) -> List[ToolMetadata]:
        """
        Search for tools by name or description with validation filtering.
        
        Args:
            query: Search query
            namespace: Optional namespace filter
            search_descriptions: Whether to search in descriptions
            validation_only: If True, only return validation-enabled tools
            
        Returns:
            List of matching ToolMetadata objects
        """
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
        """
        Validate all tools in the registry for consistency.
        
        Returns:
            Validation report with any issues found
        """
        issues = []
        total_tools = 0
        validation_enabled_count = 0
        
        for namespace, tools in self._tools.items():
            for tool_name, tool in tools.items():
                total_tools += 1
                
                # Check if tool still has execute method
                if not hasattr(tool, 'execute'):
                    issues.append(f"{namespace}.{tool_name}: Missing execute method")
                
                # Check metadata consistency
                metadata = self._metadata.get(namespace, {}).get(tool_name)
                if metadata is None:
                    issues.append(f"{namespace}.{tool_name}: Missing metadata")
                
                # Check validation wrapper consistency
                has_wrapper = tool_name in self._validation_wrappers.get(namespace, {})
                has_config = tool_name in self._validation_configs.get(namespace, {})
                
                if has_wrapper != has_config:
                    issues.append(f"{namespace}.{tool_name}: Validation wrapper/config mismatch")
                
                if has_wrapper:
                    validation_enabled_count += 1
        
        return {
            "total_tools": total_tools,
            "validation_enabled_tools": validation_enabled_count,
            "issues_found": len(issues),
            "issues": issues,
            "validation_percentage": (validation_enabled_count / total_tools * 100) if total_tools > 0 else 0,
            "registry_healthy": len(issues) == 0
        }

    # ------------------------------------------------------------------ #
    # String representation
    # ------------------------------------------------------------------ #

    def __str__(self) -> str:
        """String representation of the registry."""
        total_tools = sum(len(tools) for tools in self._tools.values())
        validation_tools = sum(len(wrappers) for wrappers in self._validation_wrappers.values())
        namespace_count = len(self._tools)
        return (
            f"InMemoryToolRegistry(tools={total_tools}, "
            f"validation_enabled={validation_tools}, "
            f"namespaces={namespace_count})"
        )

    def __repr__(self) -> str:
        """Detailed representation of the registry."""
        total_tools = sum(len(tools) for tools in self._tools.values())
        validation_tools = sum(len(wrappers) for wrappers in self._validation_wrappers.values())
        return (
            f"InMemoryToolRegistry("
            f"tools={total_tools}, "
            f"validation_enabled={validation_tools}, "
            f"namespaces={list(self._tools.keys())}, "
            f"statistics_enabled={self._enable_statistics}, "
            f"validation_available={True}, "
            f"validation_by_default={self._enable_validation_by_default})"
        )
# src/chuk_tool_registry/discovery/auto_register.py
"""
Async auto-register helpers for functions and LangChain tools.

Clean implementation assuming Pydantic and anyio are always available.
"""

from __future__ import annotations

import asyncio
import inspect
import types
from typing import (
    Callable, Type, get_type_hints, Any, Optional, Dict, 
    Union, TypeVar, Protocol, runtime_checkable, List
)

import anyio
from pydantic import BaseModel, create_model, Field

try:
    from langchain.tools.base import BaseTool  # type: ignore
except ModuleNotFoundError:
    BaseTool = None

# Core imports
from chuk_tool_registry.core.provider import ToolRegistryProvider
from chuk_tool_registry.core.validation import (
    ValidationConfig, 
    create_validation_wrapper,
    validate_arguments,
    validate_result
)
from .decorators import register_tool

# Type variables
F = TypeVar('F', bound=Callable)
T = TypeVar('T')


@runtime_checkable
class ExecutableTool(Protocol):
    """Protocol for objects that can be executed as tools."""
    async def execute(self, **kwargs: Any) -> Any: ...


class SchemaGenerationError(Exception):
    """Raised when schema generation fails."""
    pass


class ToolRegistrationError(Exception):
    """Raised when tool registration fails."""
    pass


# ────────────────────────────────────────────────────────────────────────────
# Schema Generation
# ────────────────────────────────────────────────────────────────────────────

def _resolve_type_hint(hint: Any, default_type: Type = str) -> Type:
    """Resolve a type hint to a concrete type."""
    # Handle empty/missing annotations
    if hint in (inspect._empty, None):
        return default_type
    
    # Handle string annotations and forward references
    if isinstance(hint, str):
        return default_type
    
    # Handle Union types (including Optional)
    if hasattr(hint, '__origin__') and hint.__origin__ is Union:
        # For Optional[T] (Union[T, None]), extract T
        args = hint.__args__
        non_none_args = [arg for arg in args if arg is not type(None)]
        if non_none_args:
            return non_none_args[0]
        return default_type
    
    # Return the hint if it's a valid type
    if isinstance(hint, type):
        return hint
    
    return default_type


def _create_function_schema(func: Callable, validation_config: Optional[ValidationConfig] = None) -> Type[BaseModel]:
    """Create a Pydantic model schema from a function signature."""
    validation_config = validation_config or ValidationConfig()
    
    try:
        # Get type hints
        try:
            hints = get_type_hints(func)
        except (NameError, AttributeError, TypeError):
            hints = getattr(func, '__annotations__', {})
        
        # Remove return annotation
        hints.pop("return", None)
    
        # Build Pydantic fields from function signature
        fields: Dict[str, tuple] = {}
        signature = inspect.signature(func)
        
        # Process all parameters
        for param_name, param in signature.parameters.items():
            # Skip *args and **kwargs
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            
            # Get type hint for this parameter
            hint = hints.get(param_name, param.annotation)
            if hint == inspect.Parameter.empty:
                # For parameters without type hints, explicitly use str instead of Any
                # This ensures the schema has "type": "string"
                hint = str
            
            resolved_type = _resolve_type_hint(hint, default_type=str)
            
            # Determine if the parameter is required
            if param.default is inspect.Parameter.empty:
                # Required parameter
                fields[param_name] = (resolved_type, Field(..., description=f"Parameter {param_name}"))
            else:
                # Optional parameter with default
                fields[param_name] = (resolved_type, Field(default=param.default, description=f"Parameter {param_name}"))
        
        # Create the model
        model_name = f"{func.__name__.title()}Schema"
        return create_model(model_name, **fields)
    
    except Exception as e:
        raise SchemaGenerationError(f"Failed to create schema for {func.__name__}: {e}") from e


# ────────────────────────────────────────────────────────────────────────────
# Tool Wrapper Classes
# ────────────────────────────────────────────────────────────────────────────

class FunctionToolWrapper:
    """Wrapper that adapts a function to the tool interface with validation support."""
    
    def __init__(
        self, 
        func: Callable, 
        name: str, 
        description: str,
        validation_config: Optional[ValidationConfig] = None,
        enable_validation: bool = False
    ):
        self.func = func
        self.original_func = func  # Keep reference for signature introspection
        self.name = name
        self.description = description
        self.is_async = inspect.iscoroutinefunction(func)
        self.validation_config = validation_config or ValidationConfig()
        self.enable_validation = enable_validation
        
        # Create validation wrapper if enabled
        self._validation_wrapper = None
        if self.enable_validation:
            try:
                self._validation_wrapper = self._create_validation_wrapper()
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to create validation wrapper for {name}: {e}")
                self.enable_validation = False
    
    def _create_validation_wrapper(self):
        """Create a custom validation wrapper that preserves function signature."""
        async def validated_wrapper(**kwargs):
            # Use original function for validation introspection
            if self.validation_config.validate_arguments:
                kwargs = validate_arguments(self.name, self.original_func, kwargs, self.validation_config)
            
            # Execute the original function
            if self.is_async:
                result = await self.func(**kwargs)
            else:
                # Run sync function in thread pool or directly if anyio not available
                if anyio is not None:
                    result = await anyio.to_thread.run_sync(lambda: self.func(**kwargs))
                else:
                    # Fallback: run synchronously (not ideal but works for testing)
                    result = self.func(**kwargs)
            
            # Validate result if enabled
            if self.validation_config.validate_results:
                result = validate_result(self.name, self.original_func, result, self.validation_config)
            
            return result
        
        return validated_wrapper
    
    async def execute(self, **kwargs: Any) -> Any:
        """Execute the wrapped function with optional validation."""
        if self.enable_validation and self._validation_wrapper:
            return await self._validation_wrapper(**kwargs)
        elif self.is_async:
            return await self.func(**kwargs)
        else:
            # Run sync function in thread pool or directly if anyio not available
            if anyio is not None:
                return await anyio.to_thread.run_sync(lambda: self.func(**kwargs))
            else:
                # Fallback: run synchronously (not ideal but works for testing)
                return self.func(**kwargs)
    
    def get_schema(self) -> Optional[Dict[str, Any]]:
        """Get the function's argument schema."""
        try:
            schema_model = _create_function_schema(self.original_func, self.validation_config)
            return schema_model.model_json_schema()
        except SchemaGenerationError:
            return None
    
    def __str__(self) -> str:
        validation_status = " (validated)" if self.enable_validation else ""
        return f"FunctionTool({self.name}{validation_status})"


class LangChainToolWrapper:
    """Wrapper that adapts a LangChain tool to the tool interface with validation."""
    
    def __init__(
        self, 
        langchain_tool: Any, 
        name: str, 
        description: str,
        validation_config: Optional[ValidationConfig] = None,
        enable_validation: bool = False
    ):
        self.langchain_tool = langchain_tool
        self.name = name
        self.description = description
        self.validation_config = validation_config or ValidationConfig()
        self.enable_validation = enable_validation
        
        # Create validation wrapper if enabled
        self._validation_wrapper = None
        if self.enable_validation:
            try:
                # Try to create a validation wrapper around the LangChain tool's run method
                run_method = getattr(langchain_tool, 'run', None) or getattr(langchain_tool, 'arun', None)
                if run_method:
                    self._validation_wrapper = create_validation_wrapper(
                        name, run_method, self.validation_config
                    )
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to create validation wrapper for LangChain tool {name}: {e}")
                self.enable_validation = False
    
    async def execute(self, **kwargs: Any) -> Any:
        """Execute the LangChain tool with optional validation."""
        if self.enable_validation and self._validation_wrapper:
            return await self._validation_wrapper(**kwargs)
        
        # Standard LangChain execution
        if hasattr(self.langchain_tool, 'arun') and callable(self.langchain_tool.arun):
            return await self.langchain_tool.arun(**kwargs)
        elif hasattr(self.langchain_tool, 'run') and callable(self.langchain_tool.run):
            # Run sync method in thread pool or directly if anyio not available
            if anyio is not None:
                return await anyio.to_thread.run_sync(lambda: self.langchain_tool.run(**kwargs))
            else:
                # Fallback: run synchronously (not ideal but works for testing)
                return self.langchain_tool.run(**kwargs)
        else:
            raise AttributeError(f"LangChain tool {self.langchain_tool} has no run or arun method")
    
    def get_schema(self) -> Optional[Dict[str, Any]]:
        """Get the LangChain tool's argument schema."""
        if hasattr(self.langchain_tool, 'args_schema') and self.langchain_tool.args_schema:
            try:
                return self.langchain_tool.args_schema.model_json_schema()
            except Exception:
                return None
        return None
    
    def __str__(self) -> str:
        validation_status = " (validated)" if self.enable_validation else ""
        return f"LangChainTool({self.name}{validation_status})"


# ────────────────────────────────────────────────────────────────────────────
# Registration Functions
# ────────────────────────────────────────────────────────────────────────────

async def register_fn_tool(
    func: Callable,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    namespace: str = "default",
    include_schema: bool = True,
    validation_config: Optional[ValidationConfig] = None,
    enable_validation: bool = False,
    **metadata: Any
) -> None:
    """Register a plain function as a tool with validation support."""
    if not callable(func):
        raise ToolRegistrationError(f"Expected callable, got {type(func)}")
    
    try:
        # Determine tool properties
        tool_name = name or func.__name__
        tool_description = description or inspect.getdoc(func) or f"Function tool: {tool_name}"
        
        # Create validation configuration
        validation_cfg = validation_config or ValidationConfig()
        
        # Create the enhanced tool wrapper
        tool_wrapper = FunctionToolWrapper(
            func, 
            tool_name, 
            tool_description,
            validation_cfg,
            enable_validation
        )
        
        # Build metadata - put custom fields that should go in execution_options there directly
        tool_metadata = {
            "description": tool_description,
            "is_async": True,  # Wrapper is always async
            "source": "function",
            "source_name": func.__qualname__,
            "original_function": func.__name__,
            "enable_validation": enable_validation,
        }
        
        # Add custom metadata to the main dict (InMemoryToolRegistry will move non-core fields to execution_options)
        tool_metadata.update(metadata)
        
        # Add schema if requested and available
        if include_schema:
            try:
                schema = tool_wrapper.get_schema()
                if schema:
                    tool_metadata["argument_schema"] = schema
            except Exception as e:
                # Put schema error in the metadata dict so it gets moved to execution_options
                tool_metadata["schema_generation_error"] = str(e)
        
        # Add validation configuration to metadata
        if enable_validation:
            tool_metadata["validation_config"] = {
                "validate_arguments": validation_cfg.validate_arguments,
                "validate_results": validation_cfg.validate_results,
                "strict_mode": validation_cfg.strict_mode,
                "allow_extra_args": validation_cfg.allow_extra_args,
                "coerce_types": validation_cfg.coerce_types,
            }
        
        # Register with the registry
        registry = await ToolRegistryProvider.get_registry()
        
        # Use enhanced registration method if available
        if hasattr(registry, 'register_tool') and 'validation_config' in inspect.signature(registry.register_tool).parameters:
            await registry.register_tool(
                tool_wrapper,
                name=tool_name,
                namespace=namespace,
                metadata=tool_metadata,
                validation_config=validation_cfg,
                enable_validation=enable_validation
            )
        else:
            # Fallback to standard registration
            await registry.register_tool(
                tool_wrapper,
                name=tool_name,
                namespace=namespace,
                metadata=tool_metadata
            )
        
    except Exception as e:
        raise ToolRegistrationError(f"Failed to register function {func.__name__}: {e}") from e


async def register_langchain_tool(
    tool: Any,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    namespace: str = "default",
    validation_config: Optional[ValidationConfig] = None,
    enable_validation: bool = False,
    **metadata: Any
) -> None:
    """Register a LangChain BaseTool instance with validation support."""
    if BaseTool is None:
        raise RuntimeError(
            "LangChain is required for register_langchain_tool(). "
            "Install with: pip install langchain"
        )

    if not isinstance(tool, BaseTool):
        raise TypeError(
            f"Expected langchain.tools.base.BaseTool, got {type(tool).__name__}"
        )

    try:
        # Extract tool properties
        tool_name = name or getattr(tool, 'name', None) or tool.__class__.__name__
        tool_description = (
            description or 
            getattr(tool, 'description', None) or 
            inspect.getdoc(tool) or 
            f"LangChain tool: {tool_name}"
        )
        
        # Create validation configuration
        validation_cfg = validation_config or ValidationConfig()
        
        # Create enhanced wrapper
        tool_wrapper = LangChainToolWrapper(
            tool, 
            tool_name, 
            tool_description,
            validation_cfg,
            enable_validation
        )
        
        # Build metadata
        tool_metadata = {
            "description": tool_description,
            "is_async": True,  # Wrapper is always async
            "source": "langchain",
            "source_name": tool.__class__.__qualname__,
            "langchain_tool_type": type(tool).__name__,
            "enable_validation": enable_validation,
            **metadata
        }
        
        # Add LangChain-specific metadata if available
        schema = tool_wrapper.get_schema()
        if schema:
            tool_metadata["argument_schema"] = schema
        
        # Add validation configuration to metadata
        if enable_validation:
            tool_metadata["validation_config"] = {
                "validate_arguments": validation_cfg.validate_arguments,
                "validate_results": validation_cfg.validate_results,
                "strict_mode": validation_cfg.strict_mode,
                "allow_extra_args": validation_cfg.allow_extra_args,
                "coerce_types": validation_cfg.coerce_types,
            }
        
        # Register with the registry
        registry = await ToolRegistryProvider.get_registry()
        
        # Use enhanced registration method if available
        if hasattr(registry, 'register_tool') and 'validation_config' in inspect.signature(registry.register_tool).parameters:
            await registry.register_tool(
                tool_wrapper,
                name=tool_name,
                namespace=namespace,
                metadata=tool_metadata,
                validation_config=validation_cfg,
                enable_validation=enable_validation
            )
        else:
            # Fallback to standard registration
            await registry.register_tool(
                tool_wrapper,
                name=tool_name,
                namespace=namespace,
                metadata=tool_metadata
            )
        
    except Exception as e:
        raise ToolRegistrationError(f"Failed to register LangChain tool {tool}: {e}") from e


# ────────────────────────────────────────────────────────────────────────────
# Batch Registration
# ────────────────────────────────────────────────────────────────────────────

async def register_function_batch(
    functions: Dict[str, Callable],
    *,
    namespace: str = "default",
    description_prefix: str = "",
    validation_config: Optional[ValidationConfig] = None,
    enable_validation: bool = False,
    **common_metadata: Any
) -> Dict[str, bool]:
    """Register multiple functions as tools in batch with validation support."""
    results = {}
    
    for tool_name, func in functions.items():
        try:
            description = f"{description_prefix}{func.__name__}" if description_prefix else None
            await register_fn_tool(
                func,
                name=tool_name,
                description=description,
                namespace=namespace,
                validation_config=validation_config,
                enable_validation=enable_validation,
                **common_metadata
            )
            results[tool_name] = True
        except Exception:
            results[tool_name] = False
    
    return results


async def register_module_functions(
    module: types.ModuleType,
    *,
    namespace: Optional[str] = None,
    include_private: bool = False,
    function_filter: Optional[Callable[[Callable], bool]] = None,
    validation_config: Optional[ValidationConfig] = None,
    enable_validation: bool = False,
    **common_metadata: Any
) -> Dict[str, bool]:
    """Register all functions from a module as tools with validation support."""
    if namespace is None:
        namespace = module.__name__
    
    functions = {}
    
    for name in dir(module):
        if not include_private and name.startswith('_'):
            continue
            
        obj = getattr(module, name)
        if not callable(obj) or not inspect.isfunction(obj):
            continue
            
        if function_filter and not function_filter(obj):
            continue
            
        functions[name] = obj
    
    return await register_function_batch(
        functions,
        namespace=namespace,
        validation_config=validation_config,
        enable_validation=enable_validation,
        **common_metadata
    )


# ────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ────────────────────────────────────────────────────────────────────────────

def validate_tool_function(func: Callable, validation_config: Optional[ValidationConfig] = None) -> bool:
    """
    Validate that a function can be used as a tool.
    
    Args:
        func: Function to validate
        validation_config: Optional validation configuration
        
    Returns:
        True if function is valid for tool registration, False otherwise
    """
    # Basic checks
    if not callable(func):
        return False
    
    try:
        # Check if we can introspect the signature
        inspect.signature(func)
        return True
    except (ValueError, TypeError):
        return False


def analyze_tool_function(func: Callable, validation_config: Optional[ValidationConfig] = None) -> Dict[str, Any]:
    """
    Analyze a function for tool compatibility and validation support.
    
    Args:
        func: Function to analyze
        validation_config: Optional validation configuration
        
    Returns:
        Dict with detailed analysis results
    """
    results = {
        "is_callable": callable(func),
        "is_function": inspect.isfunction(func),
        "is_async": inspect.iscoroutinefunction(func),
        "has_signature": False,
        "has_type_hints": False,
        "validation_compatible": False,
        "schema_generation_possible": False,
        "recommendations": [],
        "issues": []
    }
    
    if not callable(func):
        results["issues"].append("Object is not callable")
        return results
    
    try:
        # Check if we can introspect the signature
        signature = inspect.signature(func)
        results["has_signature"] = True
        
        # Check for type hints
        try:
            hints = get_type_hints(func)
            param_hints = {k: v for k, v in hints.items() if k != 'return'}
            results["has_type_hints"] = len(param_hints) > 0
        except (NameError, AttributeError, TypeError):
            results["has_type_hints"] = False
            param_hints = {}
        
        # Check validation compatibility
        if results["has_type_hints"]:
            try:
                # Try to create a schema
                _create_function_schema(func, validation_config)
                results["validation_compatible"] = True
                results["schema_generation_possible"] = True
            except SchemaGenerationError:
                results["validation_compatible"] = False
                results["schema_generation_possible"] = False
        else:
            # No type hints, but still compatible (will use Any)
            results["validation_compatible"] = True
            results["schema_generation_possible"] = True
        
        # Generate recommendations
        if not results["is_async"]:
            results["recommendations"].append("Consider making function async for better performance")
        
        if not results["has_type_hints"]:
            results["recommendations"].append("Add type hints for better validation")
        
    except (ValueError, TypeError) as e:
        results["issues"].append(f"Signature introspection failed: {e}")
    
    return results


async def get_registered_function_tools(namespace: Optional[str] = None, validation_only: bool = False) -> Dict[str, Any]:
    """Get all registered function tools with validation information."""
    registry = await ToolRegistryProvider.get_registry()
    all_metadata = await registry.list_metadata(namespace)
    
    function_tools = {}
    for metadata in all_metadata:
        # Filter for function tools
        if getattr(metadata, 'source', None) != "function":
            continue
        
        # Filter for validation-enabled tools if requested
        validation_enabled = getattr(metadata, 'execution_options', {}).get('validation_enabled', False)
        if validation_only and not validation_enabled:
            continue
        
        key = f"{metadata.namespace}.{metadata.name}" if namespace is None else metadata.name
        
        # Add validation information
        tool_info = {
            "metadata": metadata,
            "validation_enabled": validation_enabled,
            "validation_config": getattr(metadata, 'execution_options', {}).get('validation_config'),
            "has_schema": metadata.argument_schema is not None,
            "original_function": getattr(metadata, 'execution_options', {}).get('original_function'),
        }
        
        function_tools[key] = tool_info
    
    return function_tools


async def get_validation_statistics() -> Dict[str, Any]:
    """Get statistics about validation usage across registered tools."""
    registry = await ToolRegistryProvider.get_registry()
    all_metadata = await registry.list_metadata()
    
    stats = {
        "total_tools": len(all_metadata),
        "validation_enabled_tools": 0,
        "validation_available": True,  # Always true now
        "tools_with_schemas": 0,
        "function_tools": 0,
        "langchain_tools": 0,
        "class_tools": 0,
        "by_namespace": {},
        "validation_configs": {
            "strict_mode": 0,
            "lenient_mode": 0,
            "validate_arguments": 0,
            "validate_results": 0,
        }
    }
    
    for metadata in all_metadata:
        namespace = metadata.namespace
        if namespace not in stats["by_namespace"]:
            stats["by_namespace"][namespace] = {
                "total": 0,
                "validation_enabled": 0,
                "with_schemas": 0
            }
        
        stats["by_namespace"][namespace]["total"] += 1
        
        # Check validation status
        validation_enabled = getattr(metadata, 'execution_options', {}).get('validation_enabled', False)
        if validation_enabled:
            stats["validation_enabled_tools"] += 1
            stats["by_namespace"][namespace]["validation_enabled"] += 1
            
            # Analyze validation configuration
            validation_config = getattr(metadata, 'execution_options', {}).get('validation_config', {})
            if validation_config.get('strict_mode', False):
                stats["validation_configs"]["strict_mode"] += 1
            else:
                stats["validation_configs"]["lenient_mode"] += 1
            
            if validation_config.get('validate_arguments', True):
                stats["validation_configs"]["validate_arguments"] += 1
            
            if validation_config.get('validate_results', True):
                stats["validation_configs"]["validate_results"] += 1
        
        # Check for schemas
        if metadata.argument_schema is not None:
            stats["tools_with_schemas"] += 1
            stats["by_namespace"][namespace]["with_schemas"] += 1
        
        # Count by source type
        source = getattr(metadata, 'source', 'unknown')
        if source == 'function':
            stats["function_tools"] += 1
        elif source == 'langchain':
            stats["langchain_tools"] += 1
        elif source in ['class', 'decorator']:
            stats["class_tools"] += 1
    
    # Calculate percentages
    if stats["total_tools"] > 0:
        stats["validation_percentage"] = (stats["validation_enabled_tools"] / stats["total_tools"]) * 100
        stats["schema_percentage"] = (stats["tools_with_schemas"] / stats["total_tools"]) * 100
    else:
        stats["validation_percentage"] = 0.0
        stats["schema_percentage"] = 0.0
    
    return stats


async def validate_registry_tools() -> Dict[str, Any]:
    """Validate all tools in the registry for validation compatibility."""
    registry = await ToolRegistryProvider.get_registry()
    all_tools = await registry.list_tools()
    
    results = {
        "total_tools": len(all_tools),
        "compatible_tools": 0,
        "incompatible_tools": 0,
        "validation_enabled": 0,
        "recommendations": [],
        "tool_analysis": {}
    }
    
    for namespace, tool_name in all_tools:
        try:
            # Get the original tool
            if hasattr(registry, 'get_original_tool'):
                tool = await registry.get_original_tool(tool_name, namespace)
            else:
                tool = await registry.get_tool(tool_name, namespace)
            
            if tool is None:
                continue
            
            # Get metadata
            metadata = await registry.get_metadata(tool_name, namespace)
            
            # Analyze the tool
            tool_key = f"{namespace}.{tool_name}"
            
            if hasattr(tool, 'func'):
                # It's a function wrapper
                analysis = analyze_tool_function(tool.func)
            elif hasattr(tool, 'execute'):
                # Try to analyze the execute method
                analysis = analyze_tool_function(tool.execute)
            else:
                analysis = {"validation_compatible": False, "issues": ["No execute method or function found"]}
            
            # Check if validation is currently enabled
            validation_enabled = getattr(metadata, 'execution_options', {}).get('validation_enabled', False)
            if validation_enabled:
                results["validation_enabled"] += 1
            
            # Count compatibility
            if analysis.get("validation_compatible", False):
                results["compatible_tools"] += 1
            else:
                results["incompatible_tools"] += 1
            
            # Store analysis
            results["tool_analysis"][tool_key] = {
                "analysis": analysis,
                "validation_enabled": validation_enabled,
                "source": getattr(metadata, 'source', 'unknown'),
                "has_schema": metadata.argument_schema is not None if metadata else False
            }
            
            # Add recommendations
            if analysis.get("validation_compatible", False) and not validation_enabled:
                results["recommendations"].append(f"{tool_key}: Enable validation (compatible)")
            elif not analysis.get("validation_compatible", False) and validation_enabled:
                results["recommendations"].append(f"{tool_key}: Fix validation issues or disable validation")
        
        except Exception as e:
            tool_key = f"{namespace}.{tool_name}"
            results["tool_analysis"][tool_key] = {
                "analysis": {"validation_compatible": False, "issues": [f"Analysis failed: {e}"]},
                "validation_enabled": False,
                "source": "unknown",
                "has_schema": False
            }
            results["incompatible_tools"] += 1
    
    # Calculate percentages
    if results["total_tools"] > 0:
        results["compatibility_percentage"] = (results["compatible_tools"] / results["total_tools"]) * 100
        results["validation_adoption"] = (results["validation_enabled"] / results["total_tools"]) * 100
    else:
        results["compatibility_percentage"] = 0.0
        results["validation_adoption"] = 0.0
    
    return results


# ────────────────────────────────────────────────────────────────────────────
# Tool Creation Helpers
# ────────────────────────────────────────────────────────────────────────────

def create_validated_function_tool(
    func: Callable,
    name: Optional[str] = None,
    description: Optional[str] = None,
    validation_config: Optional[ValidationConfig] = None
) -> FunctionToolWrapper:
    """Create a validated function tool wrapper without registering it."""
    tool_name = name or func.__name__
    tool_description = description or inspect.getdoc(func) or f"Function tool: {tool_name}"
    validation_cfg = validation_config or ValidationConfig()
    
    return FunctionToolWrapper(
        func, 
        tool_name, 
        tool_description,
        validation_cfg,
        enable_validation=True
    )


def create_validated_langchain_tool(
    langchain_tool: Any,
    name: Optional[str] = None,
    description: Optional[str] = None,
    validation_config: Optional[ValidationConfig] = None
) -> LangChainToolWrapper:
    """Create a validated LangChain tool wrapper without registering it."""
    if BaseTool is None:
        raise RuntimeError("LangChain is required for LangChain tool creation")
    
    if not isinstance(langchain_tool, BaseTool):
        raise TypeError(f"Expected BaseTool, got {type(langchain_tool).__name__}")
    
    tool_name = name or getattr(langchain_tool, 'name', None) or langchain_tool.__class__.__name__
    tool_description = (
        description or 
        getattr(langchain_tool, 'description', None) or 
        f"LangChain tool: {tool_name}"
    )
    validation_cfg = validation_config or ValidationConfig()
    
    return LangChainToolWrapper(
        langchain_tool, 
        tool_name, 
        tool_description,
        validation_cfg,
        enable_validation=True
    )
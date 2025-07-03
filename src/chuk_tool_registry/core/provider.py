# chuk_tool_registry/core/provider.py
"""
Gglobal access to the async tool registry instance.

Key improvements:
1. Cleaner singleton pattern with better separation of concerns
2. Improved type safety
3. Reduced code duplication
4. Better error handling and documentation
"""
from __future__ import annotations

import asyncio
import sys
from typing import Optional, Callable, Awaitable, TypeVar
from contextlib import asynccontextmanager

# registry
from chuk_tool_registry.core.interface import ToolRegistryInterface

T = TypeVar('T', bound=ToolRegistryInterface)

# --------------------------------------------------------------------------- #
# Module-level singleton management
# --------------------------------------------------------------------------- #

class RegistryState:
    """
    Encapsulates the registry singleton state for better organization.
    """
    def __init__(self):
        self.registry: Optional[ToolRegistryInterface] = None
        self.lock = asyncio.Lock()
        self.factory: Callable[[], Awaitable[ToolRegistryInterface]] = self._default_factory
    
    async def _default_factory(self) -> ToolRegistryInterface:
        """Create the default in-memory registry."""
        from ..providers.memory import InMemoryToolRegistry
        return InMemoryToolRegistry()
    
    def set_factory(self, factory: Callable[[], Awaitable[ToolRegistryInterface]]) -> None:
        """Set a custom registry factory function."""
        self.factory = factory
    
    async def get_or_create(self) -> ToolRegistryInterface:
        """Get the registry, creating it if needed (thread-safe)."""
        if self.registry is None:
            async with self.lock:
                # Double-check pattern
                if self.registry is None:
                    self.registry = await self.factory()
        return self.registry
    
    async def set_registry(self, registry: Optional[ToolRegistryInterface]) -> None:
        """Set or clear the registry (thread-safe)."""
        async with self.lock:
            self.registry = registry
    
    async def reset(self) -> None:
        """Reset the registry state."""
        async with self.lock:
            self.registry = None


# Global state instance
_registry_state = RegistryState()


# --------------------------------------------------------------------------- #
# Public module-level functions
# --------------------------------------------------------------------------- #

async def get_registry() -> ToolRegistryInterface:
    """
    Return the process-wide registry, creating it on first use.
    
    This function is thread-safe and will only create the registry once,
    even with concurrent calls.
    
    Returns:
        The global registry instance
    """
    return await _registry_state.get_or_create()


async def set_registry(registry: Optional[ToolRegistryInterface]) -> None:
    """
    Replace or clear the global registry.

    Args:
        registry: New registry instance, or None to reset
        
    Note:
        Passing None resets the singleton so that the next get_registry()
        call recreates it using the current factory.
    """
    await _registry_state.set_registry(registry)


def set_registry_factory(factory: Callable[[], Awaitable[ToolRegistryInterface]]) -> None:
    """
    Set a custom factory function for creating the default registry.
    
    Args:
        factory: Async function that returns a ToolRegistryInterface instance
        
    Example:
        >>> async def my_factory():
        ...     return MyCustomRegistry()
        >>> set_registry_factory(my_factory)
    """
    _registry_state.set_factory(factory)


@asynccontextmanager
async def temporary_registry(registry: ToolRegistryInterface):
    """
    Context manager for temporarily using a different registry.
    
    Args:
        registry: Registry to use temporarily
        
    Example:
        >>> test_registry = InMemoryToolRegistry()
        >>> async with temporary_registry(test_registry):
        ...     # Use test_registry for this block
        ...     await register_tool(my_tool)
        ...     tool = await get_registry().get_tool("my_tool")
    """
    original = await get_registry() if _registry_state.registry else None
    await set_registry(registry)
    try:
        yield registry
    finally:
        await set_registry(original)


# --------------------------------------------------------------------------- #
# Provider class for consistent access patterns
# --------------------------------------------------------------------------- #

class ToolRegistryProvider:
    """
    Static provider class for registry access with improved organization.
    
    This class provides static methods for registry access and maintains
    its own cache separate from the module-level functions.
    """
    
    _instance_cache: Optional[ToolRegistryInterface] = None
    _cache_lock = asyncio.Lock()

    @classmethod
    async def get_registry(cls) -> ToolRegistryInterface:
        """
        Get the registry instance, using the provider's cache.
        
        This method provides an alternative access pattern that maintains
        its own cache separate from the module-level functions.
        
        Returns:
            The registry instance
        """
        if cls._instance_cache is None:
            async with cls._cache_lock:
                if cls._instance_cache is None:
                    # Use the module-level getter to ensure consistency
                    cls._instance_cache = await get_registry()
        return cls._instance_cache

    @classmethod
    async def set_registry(cls, registry: Optional[ToolRegistryInterface]) -> None:
        """
        Set the provider's cached registry instance.
        
        Args:
            registry: New registry instance, or None to clear cache
            
        Note:
            This only affects the provider's cache, not the module-level registry.
            Use the module-level set_registry() for global changes.
        """
        async with cls._cache_lock:
            cls._instance_cache = registry

    @classmethod
    async def reset(cls) -> None:
        """
        Reset both provider cache and module-level registry.
        
        This is primarily used in tests to ensure clean state.
        """
        async with cls._cache_lock:
            cls._instance_cache = None
        await _registry_state.reset()

    @classmethod
    async def get_global_registry(cls) -> ToolRegistryInterface:
        """
        Get the module-level registry directly, bypassing provider cache.
        
        Returns:
            The module-level registry instance
        """
        return await get_registry()

    @classmethod
    @asynccontextmanager
    async def isolated_registry(cls, registry: ToolRegistryInterface):
        """
        Context manager for using an isolated registry in the provider.
        
        Args:
            registry: Registry to use in isolation
            
        Example:
            >>> test_registry = InMemoryToolRegistry()
            >>> async with ToolRegistryProvider.isolated_registry(test_registry):
            ...     # Provider methods use test_registry
            ...     reg = await ToolRegistryProvider.get_registry()
            ...     assert reg is test_registry
        """
        original = cls._instance_cache
        await cls.set_registry(registry)
        try:
            yield registry
        finally:
            await cls.set_registry(original)


# --------------------------------------------------------------------------- #
# Utility functions for testing and debugging
# --------------------------------------------------------------------------- #

async def get_registry_info() -> dict:
    """
    Get information about the current registry state.
    
    Returns:
        Dict with registry state information
    """
    registry = await get_registry()
    info = {
        'registry_type': type(registry).__name__,
        'registry_module': type(registry).__module__,
        'has_provider_cache': ToolRegistryProvider._instance_cache is not None,
        'provider_cache_type': (
            type(ToolRegistryProvider._instance_cache).__name__ 
            if ToolRegistryProvider._instance_cache else None
        ),
    }
    
    # Add registry-specific info if available
    if hasattr(registry, 'list_namespaces'):
        try:
            info['namespaces'] = await registry.list_namespaces()
        except Exception:
            info['namespaces'] = 'error_retrieving'
    
    if hasattr(registry, 'list_tools'):
        try:
            tools = await registry.list_tools()
            info['tool_count'] = len(tools)
        except Exception:
            info['tool_count'] = 'error_retrieving'
    
    return info


async def reset_all_registry_state() -> None:
    """
    Reset all registry state (module-level and provider).
    
    This is useful for tests that need completely clean state.
    """
    await ToolRegistryProvider.reset()


# --------------------------------------------------------------------------- #
# Type-safe registry creation helpers
# --------------------------------------------------------------------------- #

async def create_registry(registry_type: type[T], *args, **kwargs) -> T:
    """
    Create a registry of a specific type with type safety.
    
    Args:
        registry_type: The registry class to instantiate
        *args: Positional arguments for the registry constructor
        **kwargs: Keyword arguments for the registry constructor
        
    Returns:
        The created registry instance
        
    Example:
        >>> from chuk_tool_registry.providers.memory import InMemoryToolRegistry
        >>> registry = await create_registry(InMemoryToolRegistry)
        >>> assert isinstance(registry, InMemoryToolRegistry)
    """
    if not issubclass(registry_type, ToolRegistryInterface):
        raise TypeError(f"{registry_type} is not a ToolRegistryInterface")
    
    return registry_type(*args, **kwargs)


def ensure_registry_interface(obj: object) -> ToolRegistryInterface:
    """
    Ensure an object implements the ToolRegistryInterface.
    
    Args:
        obj: Object to check
        
    Returns:
        The object, typed as ToolRegistryInterface
        
    Raises:
        TypeError: If the object doesn't implement the interface
    """
    if not isinstance(obj, ToolRegistryInterface):
        raise TypeError(f"Object {obj} does not implement ToolRegistryInterface")
    return obj
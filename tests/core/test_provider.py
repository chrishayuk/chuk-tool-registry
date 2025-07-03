# tests/core/test_provider.py
"""
Tests for the provider module (core/provider.py) functionality.

This module tests the global registry provider system, including:
- Registry state management
- Singleton pattern implementation
- Factory functions
- Context managers
- Provider class functionality
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Optional

from chuk_tool_registry.core.provider import (
    RegistryState,
    get_registry,
    set_registry,
    set_registry_factory,
    temporary_registry,
    ToolRegistryProvider,
    get_registry_info,
    reset_all_registry_state,
    create_registry,
    ensure_registry_interface,
    _registry_state
)
from chuk_tool_registry.core.interface import ToolRegistryInterface
from chuk_tool_registry.providers.memory import InMemoryToolRegistry


class TestRegistryState:
    """Test the RegistryState class."""
    
    def test_initialization(self):
        """Test RegistryState initializes correctly."""
        state = RegistryState()
        assert state.registry is None
        assert state.lock is not None
        assert callable(state.factory)
    
    @pytest.mark.asyncio
    async def test_default_factory(self):
        """Test the default factory creates InMemoryToolRegistry."""
        state = RegistryState()
        registry = await state.factory()
        assert isinstance(registry, InMemoryToolRegistry)
    
    def test_set_factory(self):
        """Test setting a custom factory."""
        state = RegistryState()
        
        async def custom_factory():
            return "custom_registry"
        
        state.set_factory(custom_factory)
        assert state.factory == custom_factory
    
    @pytest.mark.asyncio
    async def test_get_or_create_first_time(self):
        """Test get_or_create creates registry on first call."""
        state = RegistryState()
        assert state.registry is None
        
        registry = await state.get_or_create()
        assert registry is not None
        assert state.registry is registry
        assert isinstance(registry, InMemoryToolRegistry)
    
    @pytest.mark.asyncio
    async def test_get_or_create_subsequent_calls(self):
        """Test get_or_create returns same registry on subsequent calls."""
        state = RegistryState()
        
        registry1 = await state.get_or_create()
        registry2 = await state.get_or_create()
        
        assert registry1 is registry2
    
    @pytest.mark.asyncio
    async def test_get_or_create_concurrent_safety(self):
        """Test get_or_create is safe with concurrent calls."""
        state = RegistryState()
        
        # Create multiple concurrent calls
        tasks = [state.get_or_create() for _ in range(10)]
        registries = await asyncio.gather(*tasks)
        
        # All should return the same instance
        first_registry = registries[0]
        for registry in registries:
            assert registry is first_registry
    
    @pytest.mark.asyncio
    async def test_set_registry(self):
        """Test setting a registry explicitly."""
        state = RegistryState()
        mock_registry = AsyncMock(spec=ToolRegistryInterface)
        
        await state.set_registry(mock_registry)
        assert state.registry is mock_registry
    
    @pytest.mark.asyncio
    async def test_reset(self):
        """Test resetting the state."""
        state = RegistryState()
        
        # Create a registry first
        await state.get_or_create()
        assert state.registry is not None
        
        # Reset should clear it
        await state.reset()
        assert state.registry is None


class TestModuleLevelFunctions:
    """Test module-level provider functions."""
    
    @pytest.mark.asyncio
    async def test_get_registry_creates_on_first_call(self):
        """Test get_registry creates registry on first call."""
        # Reset state to ensure clean test
        await reset_all_registry_state()
        
        registry = await get_registry()
        assert registry is not None
        assert isinstance(registry, InMemoryToolRegistry)
    
    @pytest.mark.asyncio
    async def test_get_registry_returns_same_instance(self):
        """Test get_registry returns the same instance."""
        await reset_all_registry_state()
        
        registry1 = await get_registry()
        registry2 = await get_registry()
        
        assert registry1 is registry2
    
    @pytest.mark.asyncio
    async def test_set_registry_changes_global(self):
        """Test set_registry changes the global registry."""
        await reset_all_registry_state()
        
        mock_registry = AsyncMock(spec=ToolRegistryInterface)
        await set_registry(mock_registry)
        
        retrieved = await get_registry()
        assert retrieved is mock_registry
    
    @pytest.mark.asyncio
    async def test_set_registry_none_resets(self):
        """Test setting registry to None resets it."""
        await reset_all_registry_state()
        
        # Get a registry first
        original = await get_registry()
        assert original is not None
        
        # Set to None
        await set_registry(None)
        
        # Next call should create a new one
        new_registry = await get_registry()
        assert new_registry is not original
    
    def test_set_registry_factory_changes_default(self):
        """Test set_registry_factory changes the default factory."""
        original_factory = _registry_state.factory
        
        async def custom_factory():
            mock = AsyncMock(spec=ToolRegistryInterface)
            mock.custom_marker = True
            return mock
        
        try:
            set_registry_factory(custom_factory)
            assert _registry_state.factory == custom_factory
        finally:
            # Restore original factory
            _registry_state.set_factory(original_factory)
    
    @pytest.mark.asyncio
    async def test_custom_factory_used(self):
        """Test that custom factory is actually used."""
        await reset_all_registry_state()
        
        async def custom_factory():
            mock = AsyncMock(spec=ToolRegistryInterface)
            mock.custom_marker = True
            return mock
        
        original_factory = _registry_state.factory
        
        try:
            set_registry_factory(custom_factory)
            registry = await get_registry()
            assert hasattr(registry, 'custom_marker')
            assert registry.custom_marker is True
        finally:
            _registry_state.set_factory(original_factory)
            await reset_all_registry_state()


class TestTemporaryRegistryContextManager:
    """Test the temporary_registry context manager."""
    
    @pytest.mark.asyncio
    async def test_temporary_registry_isolation(self):
        """Test temporary registry provides isolation."""
        await reset_all_registry_state()
        
        # Get original registry
        original = await get_registry()
        
        # Use temporary registry
        temp_registry = AsyncMock(spec=ToolRegistryInterface)
        temp_registry.temp_marker = True
        
        async with temporary_registry(temp_registry):
            current = await get_registry()
            assert current is temp_registry
            assert hasattr(current, 'temp_marker')
        
        # Should restore original
        restored = await get_registry()
        assert restored is original
        assert not hasattr(restored, 'temp_marker')
    
    @pytest.mark.asyncio
    async def test_temporary_registry_exception_handling(self):
        """Test temporary registry restores on exception."""
        await reset_all_registry_state()
        
        original = await get_registry()
        temp_registry = AsyncMock(spec=ToolRegistryInterface)
        
        try:
            async with temporary_registry(temp_registry):
                # Verify we're using temp registry
                current = await get_registry()
                assert current is temp_registry
                
                # Raise an exception
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should still restore original
        restored = await get_registry()
        assert restored is original
    
    @pytest.mark.asyncio
    async def test_temporary_registry_nested(self):
        """Test nested temporary registries work correctly."""
        await reset_all_registry_state()
        
        original = await get_registry()
        temp1 = AsyncMock(spec=ToolRegistryInterface)
        temp1.level = 1
        temp2 = AsyncMock(spec=ToolRegistryInterface)
        temp2.level = 2
        
        async with temporary_registry(temp1):
            current1 = await get_registry()
            assert current1 is temp1
            
            async with temporary_registry(temp2):
                current2 = await get_registry()
                assert current2 is temp2
                assert current2.level == 2
            
            # Should restore temp1
            restored1 = await get_registry()
            assert restored1 is temp1
            assert restored1.level == 1
        
        # Should restore original
        restored_original = await get_registry()
        assert restored_original is original


class TestToolRegistryProvider:
    """Test the ToolRegistryProvider class."""
    
    @pytest.mark.asyncio
    async def test_provider_get_registry_caches(self):
        """Test provider get_registry caches the result."""
        await ToolRegistryProvider.reset()
        
        registry1 = await ToolRegistryProvider.get_registry()
        registry2 = await ToolRegistryProvider.get_registry()
        
        assert registry1 is registry2
    
    @pytest.mark.asyncio
    async def test_provider_set_registry(self):
        """Test provider set_registry changes cached instance."""
        await ToolRegistryProvider.reset()
        
        mock_registry = AsyncMock(spec=ToolRegistryInterface)
        await ToolRegistryProvider.set_registry(mock_registry)
        
        retrieved = await ToolRegistryProvider.get_registry()
        assert retrieved is mock_registry
    
    @pytest.mark.asyncio
    async def test_provider_reset_clears_cache(self):
        """Test provider reset clears the cache."""
        await ToolRegistryProvider.reset()
        
        # Get a registry to populate cache
        registry1 = await ToolRegistryProvider.get_registry()
        assert ToolRegistryProvider._instance_cache is not None
        
        # Reset should clear cache
        await ToolRegistryProvider.reset()
        assert ToolRegistryProvider._instance_cache is None
    
    @pytest.mark.asyncio
    async def test_provider_get_global_registry(self):
        """Test provider get_global_registry bypasses cache."""
        await reset_all_registry_state()
        await ToolRegistryProvider.reset()
        
        # Set a different registry in provider cache
        mock_registry = AsyncMock(spec=ToolRegistryInterface)
        await ToolRegistryProvider.set_registry(mock_registry)
        
        # get_global_registry should bypass provider cache
        global_registry = await ToolRegistryProvider.get_global_registry()
        provider_registry = await ToolRegistryProvider.get_registry()
        
        assert global_registry is not provider_registry
        assert provider_registry is mock_registry
    
    @pytest.mark.asyncio
    async def test_provider_isolated_registry_context(self):
        """Test provider isolated_registry context manager."""
        await ToolRegistryProvider.reset()
        
        # Get original provider registry
        original = await ToolRegistryProvider.get_registry()
        
        # Use isolated registry
        isolated = AsyncMock(spec=ToolRegistryInterface)
        isolated.isolated_marker = True
        
        async with ToolRegistryProvider.isolated_registry(isolated):
            current = await ToolRegistryProvider.get_registry()
            assert current is isolated
            assert hasattr(current, 'isolated_marker')
        
        # Should restore original
        restored = await ToolRegistryProvider.get_registry()
        assert restored is original


class TestUtilityFunctions:
    """Test utility functions in the provider module."""
    
    @pytest.mark.asyncio
    async def test_get_registry_info(self):
        """Test get_registry_info returns expected structure."""
        await reset_all_registry_state()
        
        info = await get_registry_info()
        
        assert isinstance(info, dict)
        assert 'registry_type' in info
        assert 'registry_module' in info
        assert 'has_provider_cache' in info
        assert 'provider_cache_type' in info
        
        # Registry type should be InMemoryToolRegistry by default
        assert info['registry_type'] == 'InMemoryToolRegistry'
    
    @pytest.mark.asyncio
    async def test_get_registry_info_with_namespaces(self):
        """Test get_registry_info includes namespace info when available."""
        await reset_all_registry_state()
        
        # Get registry and register a tool to create namespace
        registry = await get_registry()
        
        class TestTool:
            async def execute(self):
                return "test"
        
        await registry.register_tool(TestTool(), name="test", namespace="test_ns")
        
        info = await get_registry_info()
        
        assert 'namespaces' in info
        assert isinstance(info['namespaces'], list)
        assert 'test_ns' in info['namespaces']
    
    @pytest.mark.asyncio
    async def test_get_registry_info_with_tools(self):
        """Test get_registry_info includes tool count when available."""
        await reset_all_registry_state()
        
        # Get registry and register tools
        registry = await get_registry()
        
        class TestTool:
            async def execute(self):
                return "test"
        
        await registry.register_tool(TestTool(), name="test1")
        await registry.register_tool(TestTool(), name="test2")
        
        info = await get_registry_info()
        
        assert 'tool_count' in info
        assert info['tool_count'] == 2
    
    @pytest.mark.asyncio
    async def test_reset_all_registry_state(self):
        """Test reset_all_registry_state clears everything."""
        # Get registries to populate state
        await get_registry()
        await ToolRegistryProvider.get_registry()
        
        # Both should have state
        assert _registry_state.registry is not None
        assert ToolRegistryProvider._instance_cache is not None
        
        # Reset should clear both
        await reset_all_registry_state()
        
        assert _registry_state.registry is None
        assert ToolRegistryProvider._instance_cache is None
    
    @pytest.mark.asyncio
    async def test_create_registry_with_valid_type(self):
        """Test create_registry with valid registry type."""
        registry = await create_registry(InMemoryToolRegistry)
        
        assert isinstance(registry, InMemoryToolRegistry)
        assert isinstance(registry, ToolRegistryInterface)
    
    @pytest.mark.asyncio
    async def test_create_registry_with_args(self):
        """Test create_registry passes arguments correctly."""
        registry = await create_registry(
            InMemoryToolRegistry, 
            enable_statistics=False,
            validate_tools=False
        )
        
        assert isinstance(registry, InMemoryToolRegistry)
        # Check that arguments were passed (if the constructor supports them)
        assert registry._enable_statistics is False
        assert registry._validate_tools is False
    
    @pytest.mark.asyncio
    async def test_create_registry_with_invalid_type(self):
        """Test create_registry raises error for invalid type."""
        class NotARegistry:
            pass
        
        with pytest.raises(TypeError, match="is not a ToolRegistryInterface"):
            await create_registry(NotARegistry)
    
    def test_ensure_registry_interface_valid(self):
        """Test ensure_registry_interface with valid object."""
        registry = InMemoryToolRegistry()
        
        result = ensure_registry_interface(registry)
        assert result is registry
    
    def test_ensure_registry_interface_invalid(self):
        """Test ensure_registry_interface with invalid object."""
        not_a_registry = "not a registry"
        
        with pytest.raises(TypeError, match="does not implement ToolRegistryInterface"):
            ensure_registry_interface(not_a_registry)


class TestConcurrencyAndThreadSafety:
    """Test concurrent access and thread safety."""
    
    @pytest.mark.asyncio
    async def test_concurrent_get_registry_calls(self):
        """Test many concurrent get_registry calls return same instance."""
        await reset_all_registry_state()
        
        # Create many concurrent calls
        tasks = [get_registry() for _ in range(50)]
        registries = await asyncio.gather(*tasks)
        
        # All should be the same instance
        first_registry = registries[0]
        for registry in registries:
            assert registry is first_registry
    
    @pytest.mark.asyncio
    async def test_concurrent_provider_calls(self):
        """Test concurrent provider calls are safe."""
        await ToolRegistryProvider.reset()
        
        # Create many concurrent provider calls
        tasks = [ToolRegistryProvider.get_registry() for _ in range(50)]
        registries = await asyncio.gather(*tasks)
        
        # All should be the same instance
        first_registry = registries[0]
        for registry in registries:
            assert registry is first_registry
    
    @pytest.mark.asyncio
    async def test_mixed_module_and_provider_calls(self):
        """Test mixing module-level and provider calls."""
        await reset_all_registry_state()
        
        # Mix module-level and provider calls
        tasks = []
        for i in range(25):
            if i % 2 == 0:
                tasks.append(get_registry())
            else:
                tasks.append(ToolRegistryProvider.get_global_registry())
        
        registries = await asyncio.gather(*tasks)
        
        # All module-level calls should return same instance
        first_module_registry = registries[0]
        for i, registry in enumerate(registries):
            if i % 2 == 0:  # Module-level calls
                assert registry is first_module_registry


@pytest.fixture(autouse=True)
def cleanup_registry_state():
    """Cleanup registry state after each test."""
    yield
    # Use asyncio.run to handle the async cleanup in a sync fixture
    import asyncio
    asyncio.run(reset_all_registry_state())
    asyncio.run(ToolRegistryProvider.reset())
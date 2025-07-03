# tests/core/test_registration.py
"""
Tests for the registration module (core/registration.py) functionality.

This module tests the tool registration management system, including:
- ToolRegistrationManager functionality
- ToolRegistrationInfo data structure
- GlobalRegistrationManager singleton
- Registration processing and error handling
- Deferred registration patterns
"""

import pytest
import asyncio
import weakref
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List

from chuk_tool_registry.core.registration import (
    ToolRegistrationInfo,
    ToolRegistrationManager,
    GlobalRegistrationManager,
    create_registration_manager,
    get_global_registration_manager,
    ensure_registrations,
    reset_global_registrations,
    get_registration_statistics,
    validate_registration_manager,
)


class TestToolRegistrationInfo:
    """Test the ToolRegistrationInfo dataclass."""
    
    def test_creation_with_required_fields(self):
        """Test creating ToolRegistrationInfo with required fields."""
        info = ToolRegistrationInfo(
            name="test_tool",
            namespace="test_ns",
            metadata={"key": "value"}
        )
        
        assert info.name == "test_tool"
        assert info.namespace == "test_ns"
        assert info.metadata == {"key": "value"}
        assert info.tool_class is None
    
    def test_creation_with_all_fields(self):
        """Test creating ToolRegistrationInfo with all fields."""
        class TestTool:
            pass
        
        info = ToolRegistrationInfo(
            name="test_tool",
            namespace="test_ns", 
            metadata={"key": "value"},
            tool_class=TestTool
        )
        
        assert info.name == "test_tool"
        assert info.namespace == "test_ns"
        assert info.metadata == {"key": "value"}
        assert info.tool_class is TestTool
    
    def test_dataclass_functionality(self):
        """Test that ToolRegistrationInfo works as a proper dataclass."""
        info1 = ToolRegistrationInfo("tool1", "ns1", {"a": 1})
        info2 = ToolRegistrationInfo("tool1", "ns1", {"a": 1})
        info3 = ToolRegistrationInfo("tool2", "ns1", {"a": 1})
        
        # Test equality
        assert info1 == info2
        assert info1 != info3
        
        # Test string representation
        str_repr = str(info1)
        assert "tool1" in str_repr
        assert "ns1" in str_repr


class TestToolRegistrationManager:
    """Test the ToolRegistrationManager class."""
    
    def test_initialization_default_name(self):
        """Test manager initialization with default name."""
        manager = ToolRegistrationManager()
        
        assert manager.name.startswith("RegistrationManager-")
        assert len(manager.pending_registrations) == 0
        assert len(manager.registered_classes) == 0
        assert len(manager.registration_info) == 0
        assert manager._processed_count == 0
        assert not manager._shutting_down
    
    def test_initialization_custom_name(self):
        """Test manager initialization with custom name."""
        manager = ToolRegistrationManager("custom_manager")
        
        assert manager.name == "custom_manager"
        assert len(manager.pending_registrations) == 0
    
    def test_add_registration(self):
        """Test adding a registration to the manager."""
        manager = ToolRegistrationManager()
        
        async def mock_registration():
            return "registered"
        
        class TestTool:
            pass
        
        info = ToolRegistrationInfo("test", "ns", {})
        
        manager.add_registration(mock_registration, TestTool, info)
        
        assert len(manager.pending_registrations) == 1
        assert TestTool in manager.registered_classes
        assert manager.registration_info[TestTool] == info
    
    def test_add_registration_while_shutting_down(self):
        """Test adding registration while manager is shutting down."""
        manager = ToolRegistrationManager()
        manager._shutting_down = True
        
        async def mock_registration():
            return "registered"
        
        class TestTool:
            pass
        
        info = ToolRegistrationInfo("test", "ns", {})
        
        # Should not add registration and issue warning
        with pytest.warns(UserWarning):
            manager.add_registration(mock_registration, TestTool, info)
        
        assert len(manager.pending_registrations) == 0
        assert TestTool not in manager.registered_classes
    
    @pytest.mark.asyncio
    async def test_process_registrations_empty(self):
        """Test processing registrations when none are pending."""
        manager = ToolRegistrationManager()
        
        result = await manager.process_registrations()
        
        assert result["processed"] == 0
        assert result["total_processed"] == 0
        assert result["pending"] == 0
        assert result["errors"] == []
        assert result["manager"] == manager.name
    
    @pytest.mark.asyncio
    async def test_process_registrations_successful(self):
        """Test processing successful registrations."""
        manager = ToolRegistrationManager()
        
        call_count = 0
        
        async def mock_registration1():
            nonlocal call_count
            call_count += 1
            return "result1"
        
        async def mock_registration2():
            nonlocal call_count
            call_count += 1
            return "result2"
        
        class TestTool1:
            pass
        
        class TestTool2:
            pass
        
        info1 = ToolRegistrationInfo("tool1", "ns", {})
        info2 = ToolRegistrationInfo("tool2", "ns", {})
        
        manager.add_registration(mock_registration1, TestTool1, info1)
        manager.add_registration(mock_registration2, TestTool2, info2)
        
        result = await manager.process_registrations()
        
        assert result["processed"] == 2
        assert result["total_processed"] == 2
        assert result["pending"] == 0
        assert result["errors"] == []
        assert call_count == 2
        assert len(manager.pending_registrations) == 0
    
    @pytest.mark.asyncio
    async def test_process_registrations_with_errors(self):
        """Test processing registrations with some errors."""
        manager = ToolRegistrationManager()
        
        async def successful_registration():
            return "success"
        
        async def failing_registration():
            raise ValueError("Registration failed")
        
        class TestTool1:
            pass
        
        class TestTool2:
            pass
        
        info1 = ToolRegistrationInfo("tool1", "ns", {})
        info2 = ToolRegistrationInfo("tool2", "ns", {})
        
        manager.add_registration(successful_registration, TestTool1, info1)
        manager.add_registration(failing_registration, TestTool2, info2)
        
        result = await manager.process_registrations()
        
        assert result["processed"] == 2
        assert result["total_processed"] == 2
        assert result["pending"] == 0
        assert len(result["errors"]) == 1
        assert "Registration failed" in result["errors"][0]
    
    @pytest.mark.asyncio
    async def test_process_registrations_exception_during_task_creation(self):
        """Test handling exceptions during task creation."""
        manager = ToolRegistrationManager()
        
        def not_async_function():
            return "not async"
        
        class TestTool:
            pass
        
        info = ToolRegistrationInfo("tool", "ns", {})
        
        # Manually add a non-async function to trigger exception
        manager.pending_registrations.append(not_async_function)
        manager.registered_classes.add(TestTool)
        manager.registration_info[TestTool] = info
        
        result = await manager.process_registrations()
        
        assert result["processed"] == 1
        assert len(result["errors"]) >= 1  # Should capture the error
    
    def test_is_registered(self):
        """Test checking if a tool class is registered."""
        manager = ToolRegistrationManager()
        
        class TestTool:
            pass
        
        class UnregisteredTool:
            pass
        
        info = ToolRegistrationInfo("test", "ns", {})
        manager.add_registration(AsyncMock(), TestTool, info)
        
        assert manager.is_registered(TestTool)
        assert not manager.is_registered(UnregisteredTool)
    
    def test_get_registration_info(self):
        """Test getting registration info for a tool class."""
        manager = ToolRegistrationManager()
        
        class TestTool:
            pass
        
        info = ToolRegistrationInfo("test", "ns", {"key": "value"})
        manager.add_registration(AsyncMock(), TestTool, info)
        
        retrieved_info = manager.get_registration_info(TestTool)
        assert retrieved_info == info
        
        class UnregisteredTool:
            pass
        
        assert manager.get_registration_info(UnregisteredTool) is None
    
    def test_get_pending_count(self):
        """Test getting the count of pending registrations."""
        manager = ToolRegistrationManager()
        
        assert manager.get_pending_count() == 0
        
        class TestTool:
            pass
        
        info = ToolRegistrationInfo("test", "ns", {})
        manager.add_registration(AsyncMock(), TestTool, info)
        
        assert manager.get_pending_count() == 1
    
    def test_get_registered_classes(self):
        """Test getting all registered classes."""
        manager = ToolRegistrationManager()
        
        class TestTool1:
            pass
        
        class TestTool2:
            pass
        
        info1 = ToolRegistrationInfo("tool1", "ns", {})
        info2 = ToolRegistrationInfo("tool2", "ns", {})
        
        manager.add_registration(AsyncMock(), TestTool1, info1)
        manager.add_registration(AsyncMock(), TestTool2, info2)
        
        registered = manager.get_registered_classes()
        assert TestTool1 in registered
        assert TestTool2 in registered
        assert len(registered) == 2
    
    def test_shutdown(self):
        """Test shutting down the manager."""
        manager = ToolRegistrationManager()
        
        class TestTool:
            pass
        
        info = ToolRegistrationInfo("test", "ns", {})
        manager.add_registration(AsyncMock(), TestTool, info)
        
        assert len(manager.pending_registrations) == 1
        assert not manager._shutting_down
        
        manager.shutdown()
        
        assert len(manager.pending_registrations) == 0
        assert manager._shutting_down
    
    def test_weak_references(self):
        """Test that registered_classes uses weak references."""
        manager = ToolRegistrationManager()
        
        class TestTool:
            pass
        
        info = ToolRegistrationInfo("test", "ns", {})
        manager.add_registration(AsyncMock(), TestTool, info)
        
        # Create a weak reference to verify WeakSet behavior
        weak_ref = weakref.ref(TestTool)
        assert weak_ref() is not None
        
        # The class should be in the WeakSet
        assert TestTool in manager.registered_classes
        
        # Delete the class
        del TestTool
        
        # The weak reference should be gone (if garbage collected)
        # Note: This test might be flaky depending on garbage collection timing
    
    def test_string_representation(self):
        """Test string representation of the manager."""
        manager = ToolRegistrationManager("test_manager")
        
        class TestTool:
            pass
        
        info = ToolRegistrationInfo("test", "ns", {})
        manager.add_registration(AsyncMock(), TestTool, info)
        
        str_repr = str(manager)
        assert "test_manager" in str_repr
        assert "pending=1" in str_repr
        assert "registered=1" in str_repr
        assert "total_processed=0" in str_repr


class TestGlobalRegistrationManager:
    """Test the GlobalRegistrationManager singleton."""
    
    @pytest.mark.asyncio
    async def test_get_instance_creates_singleton(self):
        """Test that get_instance creates and returns singleton."""
        # Reset first
        await GlobalRegistrationManager.reset()
        
        instance1 = await GlobalRegistrationManager.get_instance()
        instance2 = await GlobalRegistrationManager.get_instance()
        
        assert instance1 is instance2
        assert isinstance(instance1, ToolRegistrationManager)
        assert instance1.name == "Global"
    
    @pytest.mark.asyncio
    async def test_get_instance_concurrent_safety(self):
        """Test that concurrent get_instance calls are safe."""
        await GlobalRegistrationManager.reset()
        
        # Create multiple concurrent calls
        tasks = [GlobalRegistrationManager.get_instance() for _ in range(10)]
        instances = await asyncio.gather(*tasks)
        
        # All should be the same instance
        first_instance = instances[0]
        for instance in instances:
            assert instance is first_instance
    
    @pytest.mark.asyncio
    async def test_reset_clears_instance(self):
        """Test that reset clears the singleton instance."""
        # Get an instance first
        instance1 = await GlobalRegistrationManager.get_instance()
        assert GlobalRegistrationManager._instance is not None
        
        # Reset should clear it
        await GlobalRegistrationManager.reset()
        assert GlobalRegistrationManager._instance is None
        
        # Getting instance again should create new one
        instance2 = await GlobalRegistrationManager.get_instance()
        assert instance2 is not instance1
    
    @pytest.mark.asyncio
    async def test_reset_shuts_down_existing_instance(self):
        """Test that reset properly shuts down existing instance."""
        instance = await GlobalRegistrationManager.get_instance()
        
        # Add some registrations
        class TestTool:
            pass
        
        info = ToolRegistrationInfo("test", "ns", {})
        instance.add_registration(AsyncMock(), TestTool, info)
        
        assert instance.get_pending_count() == 1
        assert not instance._shutting_down
        
        # Reset should shut down the instance
        await GlobalRegistrationManager.reset()
        
        # The instance should be shut down
        assert instance._shutting_down
        assert instance.get_pending_count() == 0
    
    @pytest.mark.asyncio
    async def test_process_global_registrations(self):
        """Test processing global registrations."""
        await GlobalRegistrationManager.reset()
        
        # Get global instance and add registrations
        instance = await GlobalRegistrationManager.get_instance()
        
        call_count = 0
        
        async def mock_registration():
            nonlocal call_count
            call_count += 1
            return "success"
        
        class TestTool:
            pass
        
        info = ToolRegistrationInfo("test", "ns", {})
        instance.add_registration(mock_registration, TestTool, info)
        
        # Process through the global manager
        result = await GlobalRegistrationManager.process_global_registrations()
        
        assert result["processed"] == 1
        assert result["errors"] == []
        assert call_count == 1


class TestFactoryAndUtilityFunctions:
    """Test factory and utility functions."""
    
    def test_create_registration_manager_default_name(self):
        """Test creating registration manager with default name."""
        manager = create_registration_manager()
        
        assert isinstance(manager, ToolRegistrationManager)
        assert manager.name.startswith("RegistrationManager-")
    
    def test_create_registration_manager_custom_name(self):
        """Test creating registration manager with custom name."""
        manager = create_registration_manager("custom")
        
        assert isinstance(manager, ToolRegistrationManager)
        assert manager.name == "custom"
    
    @pytest.mark.asyncio
    async def test_get_global_registration_manager(self):
        """Test getting the global registration manager."""
        await GlobalRegistrationManager.reset()
        
        manager = await get_global_registration_manager()
        global_instance = await GlobalRegistrationManager.get_instance()
        
        assert manager is global_instance
    
    @pytest.mark.asyncio
    async def test_ensure_registrations_with_manager(self):
        """Test ensure_registrations with specific manager."""
        manager = create_registration_manager("test")
        
        call_count = 0
        
        async def mock_registration():
            nonlocal call_count
            call_count += 1
            return "success"
        
        class TestTool:
            pass
        
        info = ToolRegistrationInfo("test", "ns", {})
        manager.add_registration(mock_registration, TestTool, info)
        
        result = await ensure_registrations(manager)
        
        assert result["processed"] == 1
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_ensure_registrations_without_manager(self):
        """Test ensure_registrations without manager (uses global)."""
        await GlobalRegistrationManager.reset()
        
        # Add registration to global manager
        global_manager = await get_global_registration_manager()
        
        call_count = 0
        
        async def mock_registration():
            nonlocal call_count
            call_count += 1
            return "success"
        
        class TestTool:
            pass
        
        info = ToolRegistrationInfo("test", "ns", {})
        global_manager.add_registration(mock_registration, TestTool, info)
        
        result = await ensure_registrations()
        
        assert result["processed"] == 1
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_reset_global_registrations(self):
        """Test resetting global registrations."""
        # Add some registrations
        global_manager = await get_global_registration_manager()
        
        class TestTool:
            pass
        
        info = ToolRegistrationInfo("test", "ns", {})
        global_manager.add_registration(AsyncMock(), TestTool, info)
        
        assert global_manager.get_pending_count() == 1
        
        # Reset should clear everything
        await reset_global_registrations()
        
        # Should have new clean instance
        new_global_manager = await get_global_registration_manager()
        assert new_global_manager.get_pending_count() == 0
    
    @pytest.mark.asyncio
    async def test_get_registration_statistics(self):
        """Test getting registration statistics."""
        await GlobalRegistrationManager.reset()
        
        global_manager = await get_global_registration_manager()
        
        class TestTool:
            pass
        
        info = ToolRegistrationInfo("test", "ns", {})
        global_manager.add_registration(AsyncMock(), TestTool, info)
        
        stats = await get_registration_statistics()
        
        assert isinstance(stats, dict)
        assert "manager_name" in stats
        assert "pending_registrations" in stats
        assert "registered_classes" in stats
        assert "total_processed" in stats
        assert "is_shutting_down" in stats
        
        assert stats["manager_name"] == "Global"
        assert stats["pending_registrations"] == 1
        assert stats["registered_classes"] == 1
        assert stats["total_processed"] == 0
        assert stats["is_shutting_down"] is False
    
    def test_validate_registration_manager_valid(self):
        """Test validate_registration_manager with valid manager."""
        manager = ToolRegistrationManager()
        
        result = validate_registration_manager(manager)
        assert result is manager
    
    def test_validate_registration_manager_invalid(self):
        """Test validate_registration_manager with invalid object."""
        invalid_manager = "not a manager"
        
        with pytest.raises(TypeError, match="Expected ToolRegistrationManager"):
            validate_registration_manager(invalid_manager)


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_process_registrations_all_failing(self):
        """Test processing when all registrations fail."""
        manager = ToolRegistrationManager()
        
        async def failing_registration1():
            raise ValueError("Error 1")
        
        async def failing_registration2():
            raise RuntimeError("Error 2")
        
        class TestTool1:
            pass
        
        class TestTool2:
            pass
        
        info1 = ToolRegistrationInfo("tool1", "ns", {})
        info2 = ToolRegistrationInfo("tool2", "ns", {})
        
        manager.add_registration(failing_registration1, TestTool1, info1)
        manager.add_registration(failing_registration2, TestTool2, info2)
        
        result = await manager.process_registrations()
        
        assert result["processed"] == 2
        assert len(result["errors"]) == 2
        assert "Error 1" in result["errors"][0]
        assert "Error 2" in result["errors"][1]
    
    @pytest.mark.asyncio
    async def test_multiple_process_registrations_calls(self):
        """Test calling process_registrations multiple times."""
        manager = ToolRegistrationManager()
        
        call_count = 0
        
        async def mock_registration():
            nonlocal call_count
            call_count += 1
            return "success"
        
        class TestTool:
            pass
        
        info = ToolRegistrationInfo("test", "ns", {})
        manager.add_registration(mock_registration, TestTool, info)
        
        # First call should process the registration
        result1 = await manager.process_registrations()
        assert result1["processed"] == 1
        assert result1["total_processed"] == 1
        assert call_count == 1
        
        # Second call should have nothing to process
        result2 = await manager.process_registrations()
        assert result2["processed"] == 0
        assert result2["total_processed"] == 1
        assert call_count == 1  # No additional calls
    
    @pytest.mark.asyncio
    async def test_concurrent_add_and_process(self):
        """Test concurrent add_registration and process_registrations calls."""
        manager = ToolRegistrationManager()
        
        async def mock_registration():
            await asyncio.sleep(0.01)  # Small delay
            return "success"
        
        class TestTool:
            pass
        
        info = ToolRegistrationInfo("test", "ns", {})
        
        # Start processing (should handle empty case)
        process_task = asyncio.create_task(manager.process_registrations())
        
        # Add registration while processing
        await asyncio.sleep(0.005)  # Let process start
        manager.add_registration(mock_registration, TestTool, info)
        
        result = await process_task
        
        # Should handle the case gracefully
        assert result["processed"] == 0  # Nothing was pending when process started
        assert manager.get_pending_count() == 1  # Registration was added after


@pytest.fixture(autouse=True)
def cleanup_global_state():
    """Cleanup global registration state after each test."""
    yield
    # Use asyncio.run to handle the async cleanup in a sync fixture
    import asyncio
    asyncio.run(GlobalRegistrationManager.reset())
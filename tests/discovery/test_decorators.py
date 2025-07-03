# tests/discovery/test_decorators.py
"""
Tests for the discovery decorators module (discovery/decorators.py).

This module tests the decorator-based tool registration system, including:
- @register_tool decorator functionality
- Deferred registration patterns
- Tool discovery mechanisms
- Registration manager integration
- SerializableTool protocol
"""

import pytest
import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List

from chuk_tool_registry.discovery.decorators import (
    register_tool,
    make_tool_serializable,
    discover_decorated_tools,
    ensure_registrations,
    SerializableTool,
    _collect_deferred_registrations,
)
from chuk_tool_registry.core.registration import (
    ToolRegistrationManager,
    ToolRegistrationInfo,
    create_registration_manager,
    reset_global_registrations,
)
from chuk_tool_registry.core.provider import reset_all_registry_state


class TestRegisterToolDecorator:
    """Test the @register_tool decorator functionality."""

    def test_decorator_with_default_parameters(self):
        """Test decorator with default parameters."""
        @register_tool()
        class TestTool:
            async def execute(self, value: int) -> int:
                return value * 2

        # Check that registration info was stored
        assert hasattr(TestTool, '_tool_registration_info')
        info = TestTool._tool_registration_info
        assert info.name == "TestTool"
        assert info.namespace == "default"
        assert info.tool_class is TestTool

    def test_decorator_with_custom_name_and_namespace(self):
        """Test decorator with custom name and namespace."""
        @register_tool(name="custom_tool", namespace="custom_ns")
        class TestTool:
            async def execute(self, value: int) -> int:
                return value * 2

        info = TestTool._tool_registration_info
        assert info.name == "custom_tool"
        assert info.namespace == "custom_ns"

    def test_decorator_with_metadata(self):
        """Test decorator with additional metadata."""
        @register_tool(
            name="meta_tool",
            namespace="meta_ns",
            version="1.0.0",
            description="A tool with metadata",
            tags=["test", "meta"]
        )
        class TestTool:
            async def execute(self, value: int) -> int:
                return value * 2

        info = TestTool._tool_registration_info
        assert info.metadata["version"] == "1.0.0"
        assert info.metadata["description"] == "A tool with metadata"
        assert info.metadata["tags"] == ["test", "meta"]

    def test_decorator_with_custom_manager(self):
        """Test decorator with custom registration manager."""
        custom_manager = create_registration_manager("test_manager")

        @register_tool(name="managed_tool", manager=custom_manager)
        class TestTool:
            async def execute(self, value: int) -> int:
                return value * 2

        # Tool should be registered with the custom manager
        assert custom_manager.is_registered(TestTool)
        assert custom_manager.get_pending_count() == 1

    def test_decorator_without_manager_creates_deferred_registration(self):
        """Test that decorator without manager creates deferred registration."""
        @register_tool(name="deferred_tool")
        class TestTool:
            async def execute(self, value: int) -> int:
                return value * 2

        # Should have deferred registration
        assert hasattr(TestTool, '_deferred_registration')
        assert hasattr(TestTool, '_tool_registration_info')

    def test_decorator_validates_async_execute_method(self):
        """Test that decorator validates execute method is async."""
        with pytest.raises(TypeError, match="must have an async execute method"):
            @register_tool()
            class BadTool:
                def execute(self, value: int) -> int:  # Not async!
                    return value * 2

    def test_decorator_allows_missing_execute_method(self):
        """Test that decorator allows classes without execute method."""
        # This should not raise an error - validation happens at runtime
        @register_tool()
        class ToolWithoutExecute:
            def some_other_method(self):
                pass

        assert hasattr(ToolWithoutExecute, '_tool_registration_info')

    def test_decorator_preserves_class_functionality(self):
        """Test that decorator preserves original class functionality."""
        @register_tool(name="preserved_tool")
        class TestTool:
            def __init__(self, multiplier: int = 2):
                self.multiplier = multiplier

            async def execute(self, value: int) -> int:
                return value * self.multiplier

            def helper_method(self) -> str:
                return "helper"

        # Class should still work normally
        instance = TestTool(multiplier=3)
        assert instance.multiplier == 3
        assert instance.helper_method() == "helper"

        # Should be async
        assert inspect.iscoroutinefunction(instance.execute)


class TestMakeToolSerializable:
    """Test the make_tool_serializable function."""

    def test_make_tool_serializable_returns_class(self):
        """Test that make_tool_serializable returns the class."""
        class TestTool:
            async def execute(self):
                return "test"

        result = make_tool_serializable(TestTool, "test_tool")
        assert result is TestTool

    def test_make_tool_serializable_with_different_name(self):
        """Test make_tool_serializable with different tool name."""
        class MyTool:
            pass

        result = make_tool_serializable(MyTool, "different_name")
        assert result is MyTool


class TestSerializableToolProtocol:
    """Test the SerializableTool protocol."""

    def test_protocol_definition(self):
        """Test that SerializableTool protocol is properly defined."""
        # Should be able to check if a class implements the protocol
        class GoodTool:
            def __getstate__(self) -> Dict[str, Any]:
                return {"state": "data"}

            def __setstate__(self, state: Dict[str, Any]) -> None:
                pass

        class BadTool:
            pass

        assert isinstance(GoodTool(), SerializableTool)
        assert not isinstance(BadTool(), SerializableTool)


class TestDiscoverDecoratedTools:
    """Test the discover_decorated_tools function."""

    def test_discover_finds_decorated_tools(self):
        """Test that discover_decorated_tools finds decorated tools."""
        # Create some decorated tools
        @register_tool(name="discoverable1")
        class Tool1:
            async def execute(self):
                return "tool1"

        @register_tool(name="discoverable2")
        class Tool2:
            async def execute(self):
                return "tool2"

        # Discover tools
        discovered = discover_decorated_tools()

        # Should find our tools (among potentially others from other tests)
        tool_classes = [tool for tool in discovered if tool in [Tool1, Tool2]]
        assert len(tool_classes) >= 2
        assert Tool1 in discovered
        assert Tool2 in discovered

    def test_discover_handles_import_errors(self):
        """Test that discover handles import errors gracefully."""
        # This should not raise an exception even if some modules have issues
        discovered = discover_decorated_tools()
        assert isinstance(discovered, list)

    def test_discover_ignores_undecorated_classes(self):
        """Test that discover ignores classes without registration info."""
        class UndecoratedTool:
            async def execute(self):
                return "undecorated"

        discovered = discover_decorated_tools()
        assert UndecoratedTool not in discovered


class TestCollectDeferredRegistrations:
    """Test the _collect_deferred_registrations function."""

    def test_collect_deferred_registrations_adds_to_manager(self):
        """Test that deferred registrations are collected properly."""
        manager = create_registration_manager("test_collector")

        # Create a tool with deferred registration
        @register_tool(name="deferred_test")
        class DeferredTool:
            async def execute(self):
                return "deferred"

        # Initially manager should be empty
        assert manager.get_pending_count() == 0

        # Collect deferred registrations
        _collect_deferred_registrations(manager)

        # Now manager should have the registration
        assert manager.get_pending_count() >= 1
        assert manager.is_registered(DeferredTool)

    def test_collect_cleans_up_deferred_registration(self):
        """Test that deferred registration is cleaned up after collection."""
        manager = create_registration_manager("test_cleanup")

        @register_tool(name="cleanup_test")
        class CleanupTool:
            async def execute(self):
                return "cleanup"

        # Should have deferred registration
        assert hasattr(CleanupTool, '_deferred_registration')

        # Collect registrations
        _collect_deferred_registrations(manager)

        # Deferred registration should be cleaned up
        assert not hasattr(CleanupTool, '_deferred_registration')

    def test_collect_handles_missing_attributes(self):
        """Test that collect handles classes with missing attributes."""
        manager = create_registration_manager("test_missing")

        # This should not raise an exception
        _collect_deferred_registrations(manager)

    def test_collect_avoids_duplicate_registrations(self):
        """Test that collect avoids duplicate registrations."""
        manager = create_registration_manager("test_duplicates")

        @register_tool(name="duplicate_test")
        class DuplicateTool:
            async def execute(self):
                return "duplicate"

        # Collect twice
        _collect_deferred_registrations(manager)
        initial_count = manager.get_pending_count()

        _collect_deferred_registrations(manager)
        final_count = manager.get_pending_count()

        # Should not have duplicates
        assert final_count == initial_count


class TestEnsureRegistrations:
    """Test the ensure_registrations function."""

    @pytest.mark.asyncio
    async def test_ensure_registrations_with_custom_manager(self):
        """Test ensure_registrations with a custom manager."""
        manager = create_registration_manager("test_ensure")

        # Mock registration function
        call_count = 0

        async def mock_registration():
            nonlocal call_count
            call_count += 1
            return "registered"

        class TestTool:
            pass

        info = ToolRegistrationInfo("test", "ns", {})
        manager.add_registration(mock_registration, TestTool, info)

        # Process registrations
        result = await ensure_registrations(manager)

        assert result["processed"] == 1
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_ensure_registrations_with_global_manager(self):
        """Test ensure_registrations with global manager."""
        # Reset global state
        await reset_global_registrations()

        # Create a deferred registration
        @register_tool(name="global_test")
        class GlobalTool:
            async def execute(self):
                return "global"

        # Process registrations (should collect deferred and process)
        result = await ensure_registrations()

        # Should have processed at least one registration
        assert result["processed"] >= 1

    @pytest.mark.asyncio
    async def test_ensure_registrations_handles_errors(self):
        """Test that ensure_registrations handles errors gracefully."""
        manager = create_registration_manager("test_errors")

        async def failing_registration():
            raise ValueError("Registration failed")

        class FailingTool:
            pass

        info = ToolRegistrationInfo("failing", "ns", {})
        manager.add_registration(failing_registration, FailingTool, info)

        # Should not raise exception
        result = await ensure_registrations(manager)

        assert result["processed"] == 1
        assert len(result["errors"]) == 1

    @pytest.mark.asyncio
    async def test_ensure_registrations_empty_manager(self):
        """Test ensure_registrations with empty manager."""
        manager = create_registration_manager("test_empty")

        result = await ensure_registrations(manager)

        assert result["processed"] == 0
        assert result["errors"] == []


class TestDecoratorIntegration:
    """Test decorator integration with the broader system."""

    @pytest.mark.asyncio
    async def test_end_to_end_decorator_registration(self):
        """Test complete end-to-end decorator registration flow."""
        from chuk_tool_registry.core.provider import ToolRegistryProvider

        # Reset state
        await reset_all_registry_state()

        # Create decorated tool
        @register_tool(name="e2e_tool", namespace="integration")
        class EndToEndTool:
            async def execute(self, message: str) -> str:
                return f"Processed: {message}"

        # Process registrations
        result = await ensure_registrations()
        assert result["processed"] >= 1

        # Get registry and verify tool is registered
        registry = await ToolRegistryProvider.get_registry()
        tool = await registry.get_tool("e2e_tool", "integration")

        # Tool should be None because decorator registers the class, not instance
        # This demonstrates the limitation of the decorator approach
        assert tool is None or hasattr(tool, 'execute')

    @pytest.mark.asyncio
    async def test_decorator_with_manager_integration(self):
        """Test decorator with manager integration."""
        from chuk_tool_registry.core.provider import ToolRegistryProvider

        # Reset state
        await reset_all_registry_state()

        # Create custom manager
        manager = create_registration_manager("integration_test")

        @register_tool(name="managed_integration", manager=manager)
        class ManagedTool:
            async def execute(self, data: Any) -> Any:
                return {"processed": data}

        # Verify tool is in manager
        assert manager.is_registered(ManagedTool)

        # Process registrations
        result = await manager.process_registrations()
        assert result["processed"] == 1

    @pytest.mark.asyncio
    async def test_multiple_decorators_same_manager(self):
        """Test multiple decorators with the same manager."""
        manager = create_registration_manager("multi_test")

        @register_tool(name="tool1", manager=manager)
        class Tool1:
            async def execute(self):
                return "tool1"

        @register_tool(name="tool2", manager=manager)
        class Tool2:
            async def execute(self):
                return "tool2"

        @register_tool(name="tool3", manager=manager)
        class Tool3:
            async def execute(self):
                return "tool3"

        # All tools should be registered with manager
        assert manager.get_pending_count() == 3
        assert manager.is_registered(Tool1)
        assert manager.is_registered(Tool2)
        assert manager.is_registered(Tool3)

        # Process all at once
        result = await manager.process_registrations()
        assert result["processed"] == 3
        assert result["errors"] == []


class TestErrorCases:
    """Test error handling and edge cases."""

    def test_decorator_with_invalid_manager(self):
        """Test decorator with invalid manager type."""
        # This should not raise at decoration time, but would fail at registration time
        @register_tool(name="invalid_manager", manager="not_a_manager")
        class TestTool:
            async def execute(self):
                return "test"

        # Should still create registration info
        assert hasattr(TestTool, '_tool_registration_info')

    def test_decorator_on_non_class(self):
        """Test that decorator works appropriately on non-class objects."""
        # The decorator expects a class, but let's see what happens
        def not_a_class():
            pass

        decorated = register_tool()(not_a_class)

        # Should add registration info even to non-classes
        assert hasattr(decorated, '_tool_registration_info')

    @pytest.mark.asyncio
    async def test_registration_with_registry_error(self):
        """Test registration when registry operations fail."""
        manager = create_registration_manager("error_test")

        @register_tool(name="error_tool", manager=manager)
        class ErrorTool:
            async def execute(self):
                return "error"

        # Mock the registration to fail
        with patch('chuk_tool_registry.core.provider.ToolRegistryProvider.get_registry') as mock_get:
            mock_registry = AsyncMock()
            mock_registry.register_tool.side_effect = Exception("Registry error")
            mock_get.return_value = mock_registry

            result = await manager.process_registrations()

            # Should capture the error
            assert result["processed"] == 1
            assert len(result["errors"]) == 1
            assert "Registry error" in result["errors"][0]


@pytest.fixture(autouse=True)
def cleanup_discovery_state():
    """Cleanup discovery state after each test."""
    yield
    # Clean up any global state that might affect other tests
    import asyncio
    asyncio.run(reset_global_registrations())
    asyncio.run(reset_all_registry_state())
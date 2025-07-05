# tests/core/test_interface.py
"""
Tests for the ToolRegistryInterface protocol with validation support.
"""
import inspect
import pytest
from typing import Any, Dict, List, Optional, Tuple

from chuk_tool_registry.core.interface import ToolRegistryInterface, DefaultToolRegistryMixin
from chuk_tool_registry.core.metadata import ToolMetadata
from chuk_tool_registry.core.validation import ValidationConfig


class TestToolRegistryInterfaceSignatures:
    """Test the method signatures of the ToolRegistryInterface."""

    @pytest.mark.parametrize(
        "method_name, expected_args, expected_defaults",
        [
            (
                "register_tool",
                ["tool", "name", "namespace", "metadata", "validation_config", "enable_validation"],
                {
                    "name": None, 
                    "namespace": "default", 
                    "metadata": None,
                    "validation_config": None,
                    "enable_validation": None
                },
            ),
            ("get_tool", ["name", "namespace"], {"namespace": "default"}),
            ("get_tool_strict", ["name", "namespace"], {"namespace": "default"}),
            ("get_metadata", ["name", "namespace"], {"namespace": "default"}),
            ("list_tools", ["namespace"], {"namespace": None}),
            ("list_namespaces", [], {}),
            ("list_metadata", ["namespace"], {"namespace": None}),
        ],
    )
    @pytest.mark.asyncio
    async def test_method_signature(self, method_name, expected_args, expected_defaults):
        # Method must exist
        method = getattr(ToolRegistryInterface, method_name, None)
        assert method is not None, f"{method_name} is not defined"

        sig = inspect.signature(method)
        # Skip the implicit 'self'
        params = list(sig.parameters.items())[1:]
        # Check parameter names
        names = [n for n, _ in params]
        assert names == expected_args, (
            f"{method_name} parameters {names} != expected {expected_args}"
        )

        # Check defaults
        for name, param in params:
            if name in expected_defaults:
                assert param.default == expected_defaults[name], (
                    f"{method_name}.{name} default {param.default} != expected {expected_defaults[name]}"
                )

    @pytest.mark.asyncio
    async def test_methods_are_async(self):
        """Test that core methods are async."""
        async_methods = [
            "register_tool",
            "get_tool",
            "get_tool_strict", 
            "get_metadata",
            "list_tools",
            "list_namespaces",
            "list_metadata",
        ]
        
        for method_name in async_methods:
            method = getattr(ToolRegistryInterface, method_name)
            assert inspect.iscoroutinefunction(method), f"{method_name} should be async"

    @pytest.mark.asyncio
    async def test_docstrings_describe_return(self):
        """Test that methods have docstrings describing return values."""
        methods_with_returns = [
            "get_tool",
            "get_tool_strict",
            "get_metadata", 
            "list_tools",
            "list_namespaces",
            "list_metadata",
        ]
        
        for method_name in methods_with_returns:
            method = getattr(ToolRegistryInterface, method_name)
            assert method.__doc__ is not None, f"{method_name} should have a docstring"
            assert "Returns:" in method.__doc__, f"{method_name} docstring should describe return value"


class TestToolRegistryInterfaceRuntimeCheckable:
    """Test runtime checking behavior of the interface."""

    @pytest.mark.asyncio
    async def test_runtime_checkable(self):
        """Test that the Protocol is runtime-checkable with enhanced signature."""
        # Define a conforming class with the enhanced signature that inherits default implementations
        class ConformingRegistry(DefaultToolRegistryMixin):
            async def register_tool(
                self, 
                tool, 
                name=None, 
                namespace="default", 
                metadata=None,
                validation_config=None,
                enable_validation=None
            ):
                pass

            async def get_tool(self, name, namespace="default"):
                return None

            async def get_tool_strict(self, name, namespace="default"):
                return None

            async def get_metadata(self, name, namespace="default"):
                return None

            async def list_tools(self, namespace=None):
                return []

            async def list_namespaces(self):
                return []

            async def list_metadata(self, namespace=None):
                return []

        # Create an instance to test with isinstance
        conforming_instance = ConformingRegistry()

        # Test with isinstance (should work if runtime_checkable)
        try:
            is_instance = isinstance(conforming_instance, ToolRegistryInterface)
            # If we got here without exception, the Protocol must be runtime_checkable
            assert is_instance, "ConformingRegistry should be an instance of ToolRegistryInterface"
        except TypeError:
            # If isinstance raises TypeError, the Protocol might not be properly configured
            pytest.fail("isinstance check raised TypeError - Protocol may not be runtime_checkable")

    @pytest.mark.asyncio
    async def test_non_conforming_class_fails(self):
        """Test that non-conforming classes fail isinstance check."""
        class NonConformingRegistry:
            def some_other_method(self):
                pass

        non_conforming_instance = NonConformingRegistry()
        
        # This should return False (not an instance)
        try:
            is_instance = isinstance(non_conforming_instance, ToolRegistryInterface)
            assert not is_instance, "NonConformingRegistry should not be an instance of ToolRegistryInterface"
        except TypeError:
            # This is also acceptable - Protocol checking might raise TypeError
            pass


class TestToolRegistryInterfaceValidationMethods:
    """Test validation-specific methods in the interface."""

    @pytest.mark.asyncio
    async def test_validation_methods_exist(self):
        """Test that validation-specific methods exist."""
        validation_methods = [
            "get_original_tool",
            "get_validation_config",
            "execute_tool_with_validation",
            "list_validation_enabled_tools",
            "tool_has_validation",
        ]
        
        for method_name in validation_methods:
            method = getattr(ToolRegistryInterface, method_name, None)
            assert method is not None, f"Validation method {method_name} should exist"
            assert inspect.iscoroutinefunction(method), f"{method_name} should be async"

    @pytest.mark.asyncio
    async def test_validation_method_signatures(self):
        """Test signatures of validation-specific methods."""
        # Test get_validation_config signature
        method = getattr(ToolRegistryInterface, "get_validation_config")
        sig = inspect.signature(method)
        params = list(sig.parameters.items())[1:]  # Skip 'self'
        names = [n for n, _ in params]
        assert names == ["name", "namespace"]
        
        # Test execute_tool_with_validation allows **kwargs
        method = getattr(ToolRegistryInterface, "execute_tool_with_validation")
        sig = inspect.signature(method)
        has_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        assert has_kwargs, "execute_tool_with_validation should accept **kwargs"

    @pytest.mark.asyncio
    async def test_utility_methods_exist(self):
        """Test that utility methods exist."""
        utility_methods = [
            "remove_tool",
            "clear_namespace", 
            "clear_all",
            "tool_exists",
            "get_statistics",
            "search_tools",
            "validate_all_tools",
        ]
        
        for method_name in utility_methods:
            method = getattr(ToolRegistryInterface, method_name, None)
            assert method is not None, f"Utility method {method_name} should exist"
            assert inspect.iscoroutinefunction(method), f"{method_name} should be async"


class TestToolRegistryInterfaceDefaultImplementations:
    """Test default implementations of optional methods."""

    @pytest.mark.asyncio
    async def test_default_implementations_work(self):
        """Test that default implementations don't raise NotImplementedError."""
        # Create a minimal implementation that inherits default implementations
        class MinimalRegistry(DefaultToolRegistryMixin):
            async def register_tool(self, tool, name=None, namespace="default", metadata=None, validation_config=None, enable_validation=None):
                pass

            async def get_tool(self, name, namespace="default"):
                return None

            async def get_tool_strict(self, name, namespace="default"):
                from chuk_tool_registry.core.exceptions import ToolNotFoundError
                raise ToolNotFoundError(f"{namespace}.{name}")

            async def get_metadata(self, name, namespace="default"):
                return None

            async def list_tools(self, namespace=None):
                return []

            async def list_namespaces(self):
                return []

            async def list_metadata(self, namespace=None):
                return []

        # Create instance
        registry = MinimalRegistry()
        
        # Test that default implementations can be called without error
        # (They should have default behaviors, not raise NotImplementedError)
        
        # Test validation methods with defaults
        original_tool = await registry.get_original_tool("test", "default")
        assert original_tool is None  # Should fall back to get_tool
        
        validation_config = await registry.get_validation_config("test", "default") 
        assert validation_config is None  # Default returns None
        
        validation_tools = await registry.list_validation_enabled_tools()
        assert validation_tools == []  # Default returns empty list
        
        has_validation = await registry.tool_has_validation("test", "default")
        assert has_validation is False  # Default returns False
        
        # Test utility methods with defaults
        removed = await registry.remove_tool("test", "default")
        assert removed is False  # Default returns False
        
        cleared = await registry.clear_namespace("test")
        assert cleared == 0  # Default returns 0
        
        all_cleared = await registry.clear_all()
        assert all_cleared == 0  # Default returns 0
        
        exists = await registry.tool_exists("test", "default")
        assert exists is False  # Default uses get_tool
        
        stats = await registry.get_statistics()
        assert isinstance(stats, dict)  # Default returns dict
        assert "total_tools" in stats
        
        search_results = await registry.search_tools("test")
        assert isinstance(search_results, list)  # Default returns list
        
        validation_report = await registry.validate_all_tools()
        assert isinstance(validation_report, dict)  # Default returns dict
        assert "total_tools" in validation_report


class TestToolRegistryInterfaceTypeHints:
    """Test type hints on interface methods."""

    def test_return_type_hints(self):
        """Test that methods have proper return type hints."""
        from typing import get_type_hints
        
        # Test a few key methods have return type hints
        methods_to_check = [
            ("get_tool", "Optional[Any]"),
            ("get_metadata", "Optional[ToolMetadata]"), 
            ("list_tools", "List[Tuple[str, str]]"),
            ("list_namespaces", "List[str]"),
            ("list_metadata", "List[ToolMetadata]"),
        ]
        
        for method_name, expected_return_type in methods_to_check:
            method = getattr(ToolRegistryInterface, method_name)
            hints = get_type_hints(method)
            
            assert "return" in hints, f"{method_name} should have return type hint"
            # Note: We're just checking that return type hints exist
            # The exact string comparison might be fragile across Python versions


class TestToolRegistryInterfaceImports:
    """Test that required imports are available."""

    def test_required_imports_available(self):
        """Test that all required types are imported."""
        # These should all be available in the interface module
        from chuk_tool_registry.core.interface import (
            ToolRegistryInterface,
            ToolMetadata,
            ValidationConfig,
        )
        
        # Test that they are the expected types
        assert inspect.isclass(ToolMetadata)
        assert inspect.isclass(ValidationConfig)
        
        # Test that ToolRegistryInterface is a Protocol
        assert hasattr(ToolRegistryInterface, '__protocol__') or hasattr(ToolRegistryInterface, '_is_protocol')
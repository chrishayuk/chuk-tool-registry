# tests/discovery/test_auto_register.py
"""
Tests for the discovery auto_register module (discovery/auto_register.py).

This module tests the automatic registration system for functions and LangChain tools, including:
- Function tool registration and wrapping
- LangChain tool integration
- Schema generation from type hints
- Batch registration operations
- Tool wrapper classes
- Error handling and validation
"""

import pytest
import asyncio
import inspect
import types
from unittest.mock import AsyncMock, MagicMock, patch, create_autospec
from typing import Any, Dict, List, Optional, Union

from chuk_tool_registry.discovery.auto_register import (
    register_fn_tool,
    register_langchain_tool,
    register_function_batch,
    register_module_functions,
    validate_tool_function,
    get_registered_function_tools,
    FunctionToolWrapper,
    LangChainToolWrapper,
    SchemaGenerationError,
    ToolRegistrationError,
    _resolve_type_hint,
    _create_function_schema,
    ExecutableTool,
)
from chuk_tool_registry.core.provider import reset_all_registry_state, ToolRegistryProvider


class TestTypeHintResolution:
    """Test the _resolve_type_hint function."""

    def test_resolve_empty_hint(self):
        """Test resolving empty/missing hints."""
        assert _resolve_type_hint(inspect._empty) == str
        assert _resolve_type_hint(None) == str

    def test_resolve_string_hint(self):
        """Test resolving string type hints."""
        assert _resolve_type_hint("SomeType") == str
        assert _resolve_type_hint("int") == str

    def test_resolve_union_hint(self):
        """Test resolving Union type hints."""
        # Optional[int] = Union[int, None]
        optional_int = Union[int, type(None)]
        assert _resolve_type_hint(optional_int) == int

        # Union[str, int]
        union_type = Union[str, int]
        assert _resolve_type_hint(union_type) == str  # First non-None type

    def test_resolve_concrete_type(self):
        """Test resolving concrete types."""
        assert _resolve_type_hint(int) == int
        assert _resolve_type_hint(str) == str
        assert _resolve_type_hint(float) == float

    def test_resolve_invalid_hint_returns_default(self):
        """Test that invalid hints return the default type."""
        class InvalidType:
            def __getitem__(self, item):
                raise TypeError("Invalid type")

        assert _resolve_type_hint(InvalidType()) == str

    def test_resolve_with_custom_default(self):
        """Test resolving with custom default type."""
        assert _resolve_type_hint(None, default_type=int) == int
        assert _resolve_type_hint("invalid", default_type=float) == float


class TestSchemaGeneration:
    """Test the _create_function_schema function."""

    def test_create_schema_simple_function(self):
        """Test schema creation for simple function."""
        def simple_func(name: str, age: int = 25) -> str:
            return f"{name} is {age}"

        schema_class = _create_function_schema(simple_func)
        schema = schema_class.model_json_schema()

        assert "name" in schema["properties"]
        assert "age" in schema["properties"]
        assert "name" in schema["required"]
        assert "age" not in schema["required"]  # Has default

    def test_create_schema_no_annotations(self):
        """Test schema creation for function without type annotations."""
        def no_annotations(param1, param2="default"):
            return param1 + param2

        schema_class = _create_function_schema(no_annotations)
        schema = schema_class.model_json_schema()

        assert "param1" in schema["properties"]
        assert "param2" in schema["properties"]
        # Should default to string type
        assert schema["properties"]["param1"]["type"] == "string"

    def test_create_schema_with_complex_types(self):
        """Test schema creation with complex type hints."""
        def complex_func(
            items: List[str],
            mapping: Dict[str, int],
            optional: Optional[str] = None
        ) -> bool:
            return len(items) > 0

        schema_class = _create_function_schema(complex_func)
        schema = schema_class.model_json_schema()

        assert "items" in schema["properties"]
        assert "mapping" in schema["properties"]
        assert "optional" in schema["properties"]

    def test_create_schema_skips_var_args(self):
        """Test that schema creation skips *args and **kwargs."""
        def var_args_func(required: str, *args, **kwargs) -> str:
            return required

        schema_class = _create_function_schema(var_args_func)
        schema = schema_class.model_json_schema()

        assert "required" in schema["properties"]
        assert len(schema["properties"]) == 1  # Only required param

    def test_create_schema_handles_forward_references(self):
        """Test schema creation with forward references."""
        def forward_ref_func(value: "ForwardRef") -> str:
            return str(value)

        # Should not raise exception, should fallback to string
        schema_class = _create_function_schema(forward_ref_func)
        schema = schema_class.model_json_schema()

        assert "value" in schema["properties"]

    def test_create_schema_raises_on_error(self):
        """Test that schema creation raises SchemaGenerationError on failure."""
        # Create a function that will cause schema generation to fail
        def problematic_func():
            pass

        # Mock create_model to raise an exception
        with patch('chuk_tool_registry.discovery.auto_register.create_model') as mock_create:
            mock_create.side_effect = Exception("Pydantic error")

            with pytest.raises(SchemaGenerationError):
                _create_function_schema(problematic_func)


class TestFunctionToolWrapper:
    """Test the FunctionToolWrapper class."""

    def test_wrapper_initialization(self):
        """Test FunctionToolWrapper initialization."""
        def test_func(x: int) -> int:
            return x * 2

        wrapper = FunctionToolWrapper(test_func, "test_tool", "Test description")

        assert wrapper.func is test_func
        assert wrapper.name == "test_tool"
        assert wrapper.description == "Test description"
        assert wrapper.is_async is False

    def test_wrapper_detects_async_function(self):
        """Test that wrapper correctly detects async functions."""
        async def async_func(x: int) -> int:
            return x * 2

        wrapper = FunctionToolWrapper(async_func, "async_tool", "Async test")
        assert wrapper.is_async is True

    @pytest.mark.asyncio
    async def test_wrapper_executes_async_function(self):
        """Test wrapper execution of async function."""
        async def async_func(x: int, y: int = 10) -> int:
            return x + y

        wrapper = FunctionToolWrapper(async_func, "async_tool", "Test")
        result = await wrapper.execute(x=5, y=15)

        assert result == 20

    @pytest.mark.asyncio
    async def test_wrapper_executes_sync_function(self):
        """Test wrapper execution of sync function."""
        def sync_func(x: int, y: int = 10) -> int:
            return x * y

        wrapper = FunctionToolWrapper(sync_func, "sync_tool", "Test")
        result = await wrapper.execute(x=5, y=3)

        assert result == 15

    @pytest.mark.asyncio
    async def test_wrapper_executes_sync_function_without_anyio(self):
        """Test wrapper execution of sync function when anyio is not available."""
        def sync_func(x: int) -> int:
            return x ** 2

        wrapper = FunctionToolWrapper(sync_func, "sync_tool", "Test")

        # Mock anyio as None
        with patch('chuk_tool_registry.discovery.auto_register.anyio', None):
            result = await wrapper.execute(x=4)
            assert result == 16

    def test_wrapper_string_representation(self):
        """Test wrapper string representation."""
        def test_func():
            pass

        wrapper = FunctionToolWrapper(test_func, "test_tool", "Test")
        assert str(wrapper) == "FunctionTool(test_tool)"


class TestLangChainToolWrapper:
    """Test the LangChainToolWrapper class."""

    def test_langchain_wrapper_initialization(self):
        """Test LangChainToolWrapper initialization."""
        mock_tool = MagicMock()
        wrapper = LangChainToolWrapper(mock_tool, "lc_tool", "LangChain test")

        assert wrapper.langchain_tool is mock_tool
        assert wrapper.name == "lc_tool"
        assert wrapper.description == "LangChain test"

    @pytest.mark.asyncio
    async def test_wrapper_prefers_arun_method(self):
        """Test that wrapper prefers arun method if available."""
        mock_tool = MagicMock()
        mock_tool.arun = AsyncMock(return_value="async_result")
        mock_tool.run = MagicMock(return_value="sync_result")

        wrapper = LangChainToolWrapper(mock_tool, "lc_tool", "Test")
        result = await wrapper.execute(param="test")

        assert result == "async_result"
        mock_tool.arun.assert_called_once_with(param="test")
        mock_tool.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_wrapper_falls_back_to_run_method(self):
        """Test that wrapper falls back to run method."""
        mock_tool = MagicMock()
        # Remove arun method completely so hasattr returns False
        if hasattr(mock_tool, 'arun'):
            del mock_tool.arun
        mock_tool.run = MagicMock(return_value="sync_result")

        wrapper = LangChainToolWrapper(mock_tool, "lc_tool", "Test")

        # Test the fallback path without anyio to avoid threading complexity
        with patch('chuk_tool_registry.discovery.auto_register.anyio', None):
            result = await wrapper.execute(param="test")

            assert result == "sync_result"
            mock_tool.run.assert_called_once_with(param="test")

    @pytest.mark.asyncio
    async def test_wrapper_run_without_anyio(self):
        """Test wrapper run method without anyio."""
        mock_tool = MagicMock()
        # Ensure no arun method exists
        if hasattr(mock_tool, 'arun'):
            del mock_tool.arun
        mock_tool.run = MagicMock(return_value="direct_result")

        wrapper = LangChainToolWrapper(mock_tool, "lc_tool", "Test")

        with patch('chuk_tool_registry.discovery.auto_register.anyio', None):
            result = await wrapper.execute(param="test")

            assert result == "direct_result"
            mock_tool.run.assert_called_once_with(param="test")

    @pytest.mark.asyncio
    async def test_wrapper_raises_error_no_run_methods(self):
        """Test that wrapper raises error when no run methods available."""
        mock_tool = MagicMock()
        # Remove run methods
        del mock_tool.arun
        del mock_tool.run

        wrapper = LangChainToolWrapper(mock_tool, "lc_tool", "Test")

        with pytest.raises(AttributeError, match="has no run or arun method"):
            await wrapper.execute(param="test")

    def test_langchain_wrapper_string_representation(self):
        """Test LangChain wrapper string representation."""
        mock_tool = MagicMock()
        wrapper = LangChainToolWrapper(mock_tool, "lc_tool", "Test")
        assert str(wrapper) == "LangChainTool(lc_tool)"


class TestRegisterFnTool:
    """Test the register_fn_tool function."""

    @pytest.mark.asyncio
    async def test_register_simple_function(self):
        """Test registering a simple function."""
        await reset_all_registry_state()

        async def simple_func(x: int) -> int:
            return x * 2

        await register_fn_tool(simple_func, name="simple", namespace="test")

        # Verify registration
        registry = await ToolRegistryProvider.get_registry()
        tool = await registry.get_tool("simple", "test")

        assert tool is not None
        assert hasattr(tool, 'execute')

        result = await tool.execute(x=5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_register_function_with_defaults(self):
        """Test registering function with default parameters."""
        await reset_all_registry_state()

        def func_with_defaults(name: str, age: int = 25) -> str:
            return f"{name} is {age}"

        await register_fn_tool(func_with_defaults)

        registry = await ToolRegistryProvider.get_registry()
        tool = await registry.get_tool("func_with_defaults", "default")

        assert tool is not None
        result = await tool.execute(name="Alice")
        assert result == "Alice is 25"

        result = await tool.execute(name="Bob", age=30)
        assert result == "Bob is 30"

    @pytest.mark.asyncio
    async def test_register_function_with_metadata(self):
        """Test registering function with custom metadata."""
        await reset_all_registry_state()

        async def meta_func(value: str) -> str:
            return value.upper()

        await register_fn_tool(
            meta_func,
            name="meta_tool",
            description="Converts to uppercase",
            namespace="meta",
            version="1.0.0",
            author="Test Author"
        )

        registry = await ToolRegistryProvider.get_registry()
        metadata = await registry.get_metadata("meta_tool", "meta")

        assert metadata is not None
        assert metadata.description == "Converts to uppercase"
        assert metadata.version == "1.0.0"
        assert metadata.execution_options.get("author") == "Test Author"

    @pytest.mark.asyncio
    async def test_register_function_with_schema_generation(self):
        """Test function registration with schema generation."""
        await reset_all_registry_state()

        def typed_func(name: str, age: int, height: float = 5.5) -> Dict[str, Any]:
            return {"name": name, "age": age, "height": height}

        await register_fn_tool(typed_func, include_schema=True)

        registry = await ToolRegistryProvider.get_registry()
        metadata = await registry.get_metadata("typed_func", "default")

        assert metadata is not None
        # Schema should be in the argument_schema field directly
        assert metadata.argument_schema is not None
        assert "properties" in metadata.argument_schema
        assert "name" in metadata.argument_schema["properties"]
        assert "age" in metadata.argument_schema["properties"]

    @pytest.mark.asyncio
    async def test_register_function_without_schema(self):
        """Test function registration without schema generation."""
        await reset_all_registry_state()

        def no_schema_func(value: Any) -> Any:
            return value

        await register_fn_tool(no_schema_func, include_schema=False)

        registry = await ToolRegistryProvider.get_registry()
        metadata = await registry.get_metadata("no_schema_func", "default")

        assert metadata is not None
        # Schema should be None when not generated
        assert metadata.argument_schema is None

    @pytest.mark.asyncio
    async def test_register_function_schema_error_handling(self):
        """Test handling of schema generation errors."""
        await reset_all_registry_state()

        def problematic_func(value):  # No type hints
            return value

        # Mock schema generation to fail
        with patch('chuk_tool_registry.discovery.auto_register._create_function_schema') as mock_schema:
            mock_schema.side_effect = SchemaGenerationError("Schema failed")

            # Should not raise exception, but continue without schema
            await register_fn_tool(problematic_func, include_schema=True)

            registry = await ToolRegistryProvider.get_registry()
            metadata = await registry.get_metadata("problematic_func", "default")

            assert metadata is not None
            assert "schema_generation_error" in metadata.execution_options

    @pytest.mark.asyncio
    async def test_register_non_callable_raises_error(self):
        """Test that registering non-callable raises error."""
        await reset_all_registry_state()

        with pytest.raises(ToolRegistrationError, match="Expected callable"):
            await register_fn_tool("not_callable")

    @pytest.mark.asyncio
    async def test_register_function_uses_docstring_description(self):
        """Test that function docstring is used as description."""
        await reset_all_registry_state()

        def documented_func(x: int) -> int:
            """This function doubles the input value."""
            return x * 2

        await register_fn_tool(documented_func)

        registry = await ToolRegistryProvider.get_registry()
        metadata = await registry.get_metadata("documented_func", "default")

        assert metadata.description == "This function doubles the input value."


class TestRegisterLangChainTool:
    """Test the register_langchain_tool function."""

    @pytest.mark.asyncio
    async def test_register_langchain_tool_not_available(self):
        """Test registration when LangChain is not available."""
        await reset_all_registry_state()

        with patch('chuk_tool_registry.discovery.auto_register.BaseTool', None):
            with pytest.raises(RuntimeError, match="LangChain is required"):
                await register_langchain_tool(MagicMock())

    @pytest.mark.asyncio
    async def test_register_invalid_langchain_tool(self):
        """Test registration of invalid LangChain tool."""
        await reset_all_registry_state()

        # Create a real BaseTool class for isinstance check
        class MockBaseTool:
            pass

        with patch('chuk_tool_registry.discovery.auto_register.BaseTool', MockBaseTool):
            # String will fail isinstance check naturally
            with pytest.raises(TypeError) as exc_info:
                await register_langchain_tool("not_a_tool")
            
            # Check that error message contains expected text
            assert "BaseTool" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_register_valid_langchain_tool(self):
        """Test registration of valid LangChain tool."""
        await reset_all_registry_state()

        # Create a real BaseTool class and instance
        class MockBaseTool:
            pass
        
        # Create tool instance that will pass isinstance check
        mock_tool = MockBaseTool()
        mock_tool.name = "mock_tool"
        mock_tool.description = "Mock LangChain tool"
        mock_tool.__class__.__qualname__ = "MockTool"

        with patch('chuk_tool_registry.discovery.auto_register.BaseTool', MockBaseTool):
            # Now isinstance(mock_tool, MockBaseTool) will return True naturally
            await register_langchain_tool(mock_tool)

            registry = await ToolRegistryProvider.get_registry()
            tool = await registry.get_tool("mock_tool", "default")

            assert tool is not None
            assert hasattr(tool, 'execute')

    @pytest.mark.asyncio
    async def test_register_langchain_tool_with_schema(self):
        """Test LangChain tool registration with schema."""
        await reset_all_registry_state()

        # Create mock tool with schema
        class MockBaseTool:
            pass
        
        mock_tool = MockBaseTool()
        mock_tool.name = "schema_tool"
        mock_tool.description = "Tool with schema"
        mock_tool.args_schema = MagicMock()
        mock_tool.args_schema.model_json_schema.return_value = {"type": "object"}

        with patch('chuk_tool_registry.discovery.auto_register.BaseTool', MockBaseTool):
            await register_langchain_tool(mock_tool)

            registry = await ToolRegistryProvider.get_registry()
            metadata = await registry.get_metadata("schema_tool", "default")

            assert metadata is not None
            # Schema should be in the argument_schema field directly
            assert metadata.argument_schema is not None
            assert metadata.argument_schema == {"type": "object"}


class TestBatchRegistration:
    """Test batch registration functions."""

    @pytest.mark.asyncio
    async def test_register_function_batch_success(self):
        """Test successful batch registration."""
        await reset_all_registry_state()

        def func1(x: int) -> int:
            return x + 1

        async def func2(x: int) -> int:
            return x * 2

        def func3(x: int) -> int:
            return x ** 2

        functions = {
            "increment": func1,
            "double": func2,
            "square": func3
        }

        results = await register_function_batch(
            functions,
            namespace="batch_test",
            description_prefix="Batch: "
        )

        assert all(results.values())
        assert len(results) == 3

        # Verify all tools are registered
        registry = await ToolRegistryProvider.get_registry()
        tools = await registry.list_tools("batch_test")
        tool_names = [name for ns, name in tools if ns == "batch_test"]

        assert "increment" in tool_names
        assert "double" in tool_names
        assert "square" in tool_names

    @pytest.mark.asyncio
    async def test_register_function_batch_with_failures(self):
        """Test batch registration with some failures."""
        await reset_all_registry_state()

        def good_func(x: int) -> int:
            return x + 1

        functions = {
            "good": good_func,
            "bad": "not_a_function"  # This will fail
        }

        results = await register_function_batch(functions, namespace="mixed_test")

        assert results["good"] is True
        assert results["bad"] is False

    @pytest.mark.asyncio
    async def test_register_module_functions(self):
        """Test registering all functions from a module."""
        await reset_all_registry_state()

        # Create a mock module
        mock_module = types.ModuleType("test_module")
        mock_module.__name__ = "test_module"

        def public_func(x: int) -> int:
            return x + 1

        def _private_func(x: int) -> int:
            return x - 1

        def another_public(x: str) -> str:
            return x.upper()

        # Add functions to mock module
        setattr(mock_module, 'public_func', public_func)
        setattr(mock_module, '_private_func', _private_func)
        setattr(mock_module, 'another_public', another_public)
        setattr(mock_module, 'not_a_function', "string_value")

        results = await register_module_functions(
            mock_module,
            namespace="module_test",
            include_private=False
        )

        # Should register only public functions
        assert "public_func" in results
        assert "another_public" in results
        assert "_private_func" not in results
        assert "not_a_function" not in results

    @pytest.mark.asyncio
    async def test_register_module_functions_with_filter(self):
        """Test module function registration with custom filter."""
        await reset_all_registry_state()

        mock_module = types.ModuleType("filtered_module")
        mock_module.__name__ = "filtered_module"

        def int_func(x: int) -> int:
            return x

        def str_func(x: str) -> str:
            return x

        setattr(mock_module, 'int_func', int_func)
        setattr(mock_module, 'str_func', str_func)

        # Filter to only include functions with 'int' in name
        def name_filter(func):
            return 'int' in func.__name__

        results = await register_module_functions(
            mock_module,
            function_filter=name_filter
        )

        assert "int_func" in results
        assert "str_func" not in results


class TestUtilityFunctions:
    """Test utility functions."""

    def test_validate_tool_function_valid(self):
        """Test validation of valid functions."""
        def valid_func(x: int) -> int:
            return x

        assert validate_tool_function(valid_func) is True

    def test_validate_tool_function_invalid(self):
        """Test validation of invalid functions."""
        assert validate_tool_function("not_a_function") is False
        assert validate_tool_function(None) is False

    def test_validate_tool_function_problematic_signature(self):
        """Test validation of function with problematic signature."""
        # Create a mock function that raises on signature inspection
        mock_func = MagicMock()
        with patch('inspect.signature', side_effect=ValueError("Bad signature")):
            assert validate_tool_function(mock_func) is False

    @pytest.mark.asyncio
    async def test_get_registered_function_tools(self):
        """Test getting registered function tools."""
        await reset_all_registry_state()

        def test_func(x: int) -> int:
            return x

        await register_fn_tool(test_func, namespace="func_test")

        function_tools = await get_registered_function_tools("func_test")

        assert len(function_tools) >= 1
        assert any("test_func" in key for key in function_tools.keys())

    @pytest.mark.asyncio
    async def test_get_registered_function_tools_all_namespaces(self):
        """Test getting function tools from all namespaces."""
        await reset_all_registry_state()

        def test_func1(x: int) -> int:
            return x

        def test_func2(x: str) -> str:
            return x

        await register_fn_tool(test_func1, namespace="ns1")
        await register_fn_tool(test_func2, namespace="ns2")

        function_tools = await get_registered_function_tools()

        # Should include tools from both namespaces
        assert len(function_tools) >= 2


class TestExecutableToolProtocol:
    """Test the ExecutableTool protocol."""

    def test_executable_tool_protocol(self):
        """Test ExecutableTool protocol checking."""
        class GoodTool:
            async def execute(self, **kwargs):
                return "result"

        class BadTool:
            def not_execute(self):
                pass

        assert isinstance(GoodTool(), ExecutableTool)
        assert not isinstance(BadTool(), ExecutableTool)


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_registration_error_propagation(self):
        """Test that registration errors are properly propagated."""
        await reset_all_registry_state()

        def test_func():
            pass

        # Mock registry to raise an error
        with patch('chuk_tool_registry.core.provider.ToolRegistryProvider.get_registry') as mock_get:
            mock_registry = AsyncMock()
            mock_registry.register_tool.side_effect = Exception("Registry error")
            mock_get.return_value = mock_registry

            with pytest.raises(ToolRegistrationError, match="Failed to register function"):
                await register_fn_tool(test_func)

    @pytest.mark.asyncio
    async def test_wrapper_execution_error_handling(self):
        """Test error handling in wrapper execution."""
        def error_func():
            raise ValueError("Function error")

        wrapper = FunctionToolWrapper(error_func, "error_tool", "Error test")

        with pytest.raises(ValueError, match="Function error"):
            await wrapper.execute()


@pytest.fixture(autouse=True)
def cleanup_auto_register_state():
    """Cleanup auto-register state after each test."""
    yield
    import asyncio
    asyncio.run(reset_all_registry_state())
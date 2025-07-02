"""
Simplified tests for auto-registration functionality.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch

from chuk_tool_registry.discovery.auto_register import (
    register_fn_tool,
    register_langchain_tool,
    _auto_schema,
)
from chuk_tool_registry.core.provider import ToolRegistryProvider


# Test functions
async def async_test_function(message: str, count: int = 1) -> str:
    """An async test function."""
    return f"{message} x {count}"


def sync_test_function(data: str) -> str:
    """A sync test function."""
    return f"Processed: {data}"


class TestAutoSchema:
    """Test automatic schema generation."""
    
    def test_auto_schema_basic(self):
        """Test basic schema generation."""
        def test_func(name: str, age: int) -> str:
            return f"{name}-{age}"
        
        schema_class = _auto_schema(test_func)
        instance = schema_class(name="test", age=25)
        assert instance.name == "test"
        assert instance.age == 25


class TestRegisterFnTool:
    """Test function registration."""
    
    @pytest.mark.asyncio
    async def test_register_async_function_basic(self):
        """Test basic async function registration."""
        # Reset registry
        await ToolRegistryProvider.set_registry(None)
        registry = await ToolRegistryProvider.get_registry()
        
        # Register function
        await register_fn_tool(async_test_function, name="async_tester")
        
        # Test it was registered
        tool = await registry.get_tool("async_tester")
        assert tool is not None
        
        # Test execution
        result = await tool.execute(message="Hello", count=2)
        assert result == "Hello x 2"
    
    @pytest.mark.asyncio
    async def test_register_sync_function_basic(self):
        """Test basic sync function registration."""
        # Reset registry  
        await ToolRegistryProvider.set_registry(None)
        registry = await ToolRegistryProvider.get_registry()
        
        # Register function
        await register_fn_tool(sync_test_function, name="sync_tester")
        
        # Test it was registered
        tool = await registry.get_tool("sync_tester")
        assert tool is not None
        
        # Test execution (sync function wrapped in async)
        result = await tool.execute(data="test input")
        assert result == "Processed: test input"
    
    @pytest.mark.asyncio
    async def test_register_function_with_namespace(self):
        """Test function registration with custom namespace."""
        # Reset registry
        await ToolRegistryProvider.set_registry(None)
        registry = await ToolRegistryProvider.get_registry()
        
        # Register in custom namespace
        await register_fn_tool(
            async_test_function, 
            name="namespaced_tool",
            namespace="custom"
        )
        
        # Should be in custom namespace
        tool = await registry.get_tool("namespaced_tool", namespace="custom")
        assert tool is not None
        
        # Should not be in default namespace
        default_tool = await registry.get_tool("namespaced_tool")
        assert default_tool is None


class TestLangChainRegistration:
    """Test LangChain tool registration."""
    
    @pytest.mark.asyncio
    async def test_langchain_not_available(self):
        """Test error when LangChain not available."""
        with patch('chuk_tool_registry.discovery.auto_register.BaseTool', None):
            mock_tool = Mock()
            
            with pytest.raises(RuntimeError, match="requires LangChain"):
                await register_langchain_tool(mock_tool)
    
    @pytest.mark.asyncio
    async def test_langchain_basic_registration(self):
        """Test basic LangChain tool registration."""
        # Reset registry
        await ToolRegistryProvider.set_registry(None)
        registry = await ToolRegistryProvider.get_registry()
        
        # Create mock tool
        mock_tool = Mock()
        mock_tool.name = "test_langchain"
        mock_tool.description = "Test tool"
        
        # Create a proper mock function with __name__ attribute
        def mock_run(input_text):
            return f"Result: {input_text}"
        
        # Set the run method to our real function (not a Mock)
        mock_tool.run = mock_run
        
        # Don't add arun method so it will use run method instead
        # (The code prefers arun if it exists, but arun is a Mock without __name__)
        
        # Create a mock BaseTool class that our mock_tool will "inherit" from
        class MockBaseTool:
            pass
        
        # Make our mock tool appear to be an instance of BaseTool
        # by setting its __class__ to our mock base tool
        mock_tool.__class__ = MockBaseTool
        
        # Mock the BaseTool import and make isinstance work properly
        with patch('chuk_tool_registry.discovery.auto_register.BaseTool', MockBaseTool):
            # Remove arun if it exists as a Mock attribute
            if hasattr(mock_tool, 'arun'):
                delattr(mock_tool, 'arun')
            await register_langchain_tool(mock_tool)
        
        # Should be registered
        tool = await registry.get_tool("test_langchain")
        assert tool is not None
        
        # Check metadata has langchain tag
        metadata = await registry.get_metadata("test_langchain")
        assert metadata is not None
        assert "langchain" in metadata.tags


class TestIntegration:
    """Integration tests."""
    
    @pytest.mark.asyncio 
    async def test_multiple_function_registration(self):
        """Test registering multiple functions."""
        # Reset registry
        await ToolRegistryProvider.set_registry(None)
        registry = await ToolRegistryProvider.get_registry()
        
        # Define functions
        async def func1(x: int) -> int:
            return x * 2
        
        def func2(y: str) -> str:
            return y.upper()
        
        # Register both
        await register_fn_tool(func1, name="doubler")
        await register_fn_tool(func2, name="upper")
        
        # Test both work
        tool1 = await registry.get_tool("doubler")
        tool2 = await registry.get_tool("upper")
        
        assert tool1 is not None
        assert tool2 is not None
        
        result1 = await tool1.execute(x=5)
        result2 = await tool2.execute(y="hello")
        
        assert result1 == 10
        assert result2 == "HELLO"


@pytest.mark.asyncio
async def test_complete_workflow():
    """Test a complete registration workflow."""
    # Reset registry
    await ToolRegistryProvider.set_registry(None)
    registry = await ToolRegistryProvider.get_registry()
    
    # Define tools
    async def greet(name: str) -> str:
        return f"Hello, {name}!"
    
    def calculate(a: int, b: int) -> int:
        return a + b
    
    # Register tools
    await register_fn_tool(greet, name="greeter")
    await register_fn_tool(calculate, name="calculator")
    
    # Test they work
    greeter = await registry.get_tool("greeter")
    calculator = await registry.get_tool("calculator")
    
    greeting = await greeter.execute(name="World")
    result = await calculator.execute(a=5, b=3)
    
    assert greeting == "Hello, World!"
    assert result == 8
    
    # Check registry state
    tools = await registry.list_tools()
    tool_names = {name for namespace, name in tools}
    assert "greeter" in tool_names
    assert "calculator" in tool_names
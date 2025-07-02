"""
Simplified tests for discovery module public API.
"""
import pytest
import asyncio

from chuk_tool_registry.discovery import (
    register_tool,
    ensure_registrations,
    register_fn_tool,
    register_all_pending,
    __version__,
)
from chuk_tool_registry.core.provider import ToolRegistryProvider
from chuk_tool_registry.discovery.decorators import (
    _PENDING_REGISTRATIONS,
    _REGISTERED_CLASSES,
)


class TestPublicAPI:
    """Test the public API exports."""
    
    def test_version_exists(self):
        """Test version is available."""
        assert isinstance(__version__, str)
        assert len(__version__) > 0
    
    def test_all_functions_callable(self):
        """Test all exported functions are callable."""
        assert callable(register_tool)
        assert callable(ensure_registrations)
        assert callable(register_fn_tool)
        assert callable(register_all_pending)
    
    def test_register_all_pending_alias(self):
        """Test register_all_pending is an alias."""
        # Should be the same function
        assert register_all_pending == ensure_registrations


class TestIntegratedWorkflow:
    """Test complete workflows using public API."""
    
    def setup_method(self):
        """Clean up before each test."""
        _PENDING_REGISTRATIONS.clear()
        _REGISTERED_CLASSES.clear()
    
    @pytest.mark.asyncio
    async def test_mixed_registration_workflow(self):
        """Test using both decorators and function registration."""
        # Reset registry
        await ToolRegistryProvider.set_registry(None)
        registry = await ToolRegistryProvider.get_registry()
        
        # Class-based tool with decorator
        @register_tool("string_tool")
        class StringTool:
            async def execute(self, text: str, action: str = "reverse") -> str:
                if action == "reverse":
                    return text[::-1]
                elif action == "upper":
                    return text.upper()
                else:
                    return text
        
        # Function-based tool
        async def math_tool(a: int, b: int) -> int:
            return a + b
        
        # Register function tool
        await register_fn_tool(math_tool, name="adder")
        
        # Register all pending (for decorator)
        await register_all_pending()
        
        # Test both tools work
        string_class = await registry.get_tool("string_tool")
        math_func = await registry.get_tool("adder")
        
        assert string_class is not None
        assert math_func is not None
        
        # Test string tool
        string_tool = string_class()
        result1 = await string_tool.execute(text="hello", action="reverse")
        assert result1 == "olleh"
        
        # Test math tool
        result2 = await math_func.execute(a=10, b=5)
        assert result2 == 15
    
    @pytest.mark.asyncio
    async def test_namespace_workflow(self):
        """Test workflow with namespaces."""
        # Reset registry
        await ToolRegistryProvider.set_registry(None)
        registry = await ToolRegistryProvider.get_registry()
        
        # Tools in different namespaces
        @register_tool("tool1", namespace="ns1")
        class Tool1:
            async def execute(self, x: int) -> str:
                return f"NS1: {x}"
        
        @register_tool("tool2", namespace="ns2")
        class Tool2:
            async def execute(self, x: int) -> str:
                return f"NS2: {x}"
        
        # Function in default namespace
        async def default_tool(x: int) -> str:
            return f"Default: {x}"
        
        await register_fn_tool(default_tool, name="default_func")
        await register_all_pending()
        
        # Test all tools in their respective namespaces
        tool1_class = await registry.get_tool("tool1", namespace="ns1")
        tool2_class = await registry.get_tool("tool2", namespace="ns2")
        default_func = await registry.get_tool("default_func")
        
        assert tool1_class is not None
        assert tool2_class is not None
        assert default_func is not None
        
        # Test cross-namespace isolation
        assert await registry.get_tool("tool1", namespace="ns2") is None
        assert await registry.get_tool("tool2", namespace="ns1") is None
        
        # Test execution
        tool1 = tool1_class()
        tool2 = tool2_class()
        
        result1 = await tool1.execute(x=1)
        result2 = await tool2.execute(x=2)
        result3 = await default_func.execute(x=3)
        
        assert result1 == "NS1: 1"
        assert result2 == "NS2: 2"
        assert result3 == "Default: 3"


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def setup_method(self):
        """Clean up before each test."""
        _PENDING_REGISTRATIONS.clear()
        _REGISTERED_CLASSES.clear()
    
    @pytest.mark.asyncio
    async def test_empty_ensure_registrations(self):
        """Test ensure_registrations with no pending tools."""
        # Should not error
        await ensure_registrations()
        await register_all_pending()  # Same thing
    
    @pytest.mark.asyncio
    async def test_function_error_propagation(self):
        """Test that function errors are properly propagated."""
        # Reset registry
        await ToolRegistryProvider.set_registry(None)
        registry = await ToolRegistryProvider.get_registry()
        
        # Function that can error
        async def error_func(should_fail: bool = True) -> str:
            if should_fail:
                raise ValueError("Test error")
            return "Success"
        
        await register_fn_tool(error_func, name="error_tool")
        
        tool = await registry.get_tool("error_tool")
        assert tool is not None
        
        # Should propagate error
        with pytest.raises(ValueError, match="Test error"):
            await tool.execute(should_fail=True)
        
        # Should work when not failing
        result = await tool.execute(should_fail=False)
        assert result == "Success"


@pytest.mark.asyncio
async def test_complete_example():
    """Test a complete real-world example."""
    # Clean up
    _PENDING_REGISTRATIONS.clear()
    _REGISTERED_CLASSES.clear()
    await ToolRegistryProvider.set_registry(None)
    registry = await ToolRegistryProvider.get_registry()
    
    # Define a processing pipeline
    @register_tool("text_analyzer")
    class TextAnalyzer:
        async def execute(self, text: str) -> dict:
            words = text.split()
            return {
                "word_count": len(words),
                "char_count": len(text),
                "first_word": words[0] if words else "",
                "last_word": words[-1] if words else ""
            }
    
    # Function for formatting
    async def format_result(analysis: dict) -> str:
        return f"Text has {analysis['word_count']} words and {analysis['char_count']} characters"
    
    # Simple math function
    def multiply(a: int, b: int) -> int:
        return a * b
    
    # Register everything
    await register_fn_tool(format_result, name="formatter")
    await register_fn_tool(multiply, name="multiplier")
    await register_all_pending()
    
    # Test the pipeline
    analyzer_class = await registry.get_tool("text_analyzer")
    formatter = await registry.get_tool("formatter")
    multiplier = await registry.get_tool("multiplier")
    
    # Run analysis
    analyzer = analyzer_class()
    analysis = await analyzer.execute(text="Hello world test")
    
    # Format result
    formatted = await formatter.execute(analysis=analysis)
    
    # Do some math
    product = await multiplier.execute(a=analysis["word_count"], b=2)
    
    # Verify results
    assert analysis["word_count"] == 3
    assert analysis["char_count"] == 16
    assert analysis["first_word"] == "Hello"
    assert analysis["last_word"] == "test"
    assert "3 words and 16 characters" in formatted
    assert product == 6
    
    # Check registry state
    tools = await registry.list_tools()
    tool_names = {name for namespace, name in tools}
    expected_tools = {"text_analyzer", "formatter", "multiplier"}
    assert expected_tools.issubset(tool_names)
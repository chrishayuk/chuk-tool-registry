#!/usr/bin/env python3
"""
Chuk Tool Registry - Comprehensive Package Diagnostic Script

This script performs a complete health check of the chuk_tool_registry package,
including dependency validation, core functionality testing, performance metrics,
and configuration analysis.

Usage:
    python diagnostic.py [--verbose] [--export-report] [--fix-issues]
"""

import asyncio
import sys
import os
import time
import traceback
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import importlib.util


@dataclass
class DiagnosticResult:
    """Result of a diagnostic check."""
    name: str
    status: str  # "PASS", "FAIL", "WARN", "SKIP"
    message: str
    details: Optional[Dict[str, Any]] = None
    duration_ms: Optional[float] = None
    exception: Optional[Exception] = None


class PackageDiagnostic:
    """Comprehensive diagnostic tool for chuk_tool_registry package."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[DiagnosticResult] = []
        self.start_time = time.time()
        
    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    def add_result(self, result: DiagnosticResult) -> None:
        """Add a diagnostic result."""
        self.results.append(result)
        status_emoji = {
            "PASS": "✅",
            "FAIL": "❌", 
            "WARN": "⚠️",
            "SKIP": "⏭️"
        }
        emoji = status_emoji.get(result.status, "❓")
        duration = f" ({result.duration_ms:.1f}ms)" if result.duration_ms else ""
        print(f"{emoji} {result.name}: {result.message}{duration}")
        
        if result.status == "FAIL" and result.exception and self.verbose:
            print(f"    Exception: {type(result.exception).__name__}: {result.exception}")
            if hasattr(result.exception, '__traceback__'):
                tb_lines = traceback.format_tb(result.exception.__traceback__)
                for line in tb_lines[-3:]:  # Show last 3 stack frames
                    print(f"    {line.strip()}")
    
    async def run_check(self, name: str, check_func, *args, **kwargs) -> DiagnosticResult:
        """Run a diagnostic check with timing and error handling."""
        start_time = time.time()
        try:
            self.log(f"Running check: {name}")
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func(*args, **kwargs)
            else:
                result = check_func(*args, **kwargs)
            
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(result, DiagnosticResult):
                result.duration_ms = duration_ms
                return result
            else:
                return DiagnosticResult(
                    name=name,
                    status="PASS",
                    message="Check completed successfully",
                    details=result if isinstance(result, dict) else None,
                    duration_ms=duration_ms
                )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return DiagnosticResult(
                name=name,
                status="FAIL",
                message=f"Check failed: {str(e)}",
                duration_ms=duration_ms,
                exception=e
            )
    
    # ------------------------------------------------------------------ #
    # Core Diagnostic Checks
    # ------------------------------------------------------------------ #
    
    def check_python_version(self) -> DiagnosticResult:
        """Check Python version compatibility."""
        version = sys.version_info
        
        if version >= (3, 9):
            status = "PASS"
            message = f"Python {version.major}.{version.minor}.{version.micro} (compatible)"
        elif version >= (3, 8):
            status = "WARN"
            message = f"Python {version.major}.{version.minor}.{version.micro} (minimum supported, upgrade recommended)"
        else:
            status = "FAIL"
            message = f"Python {version.major}.{version.minor}.{version.micro} (unsupported, requires 3.8+)"
        
        return DiagnosticResult(
            name="Python Version",
            status=status,
            message=message,
            details={
                "version": f"{version.major}.{version.minor}.{version.micro}",
                "implementation": sys.implementation.name,
                "platform": sys.platform
            }
        )
    
    def check_core_dependencies(self) -> DiagnosticResult:
        """Check if core dependencies are available."""
        required_deps = ["asyncio", "typing", "dataclasses", "weakref", "inspect"]
        # Pydantic is required for the registry metadata system
        critical_optional = ["pydantic"]  
        # These are truly optional and don't affect core functionality
        optional_deps = ["anyio", "langchain"]
        
        missing_required = []
        missing_critical = []
        missing_optional = []
        available_versions = {}
        
        # Check required dependencies (built-in Python modules)
        for dep in required_deps:
            try:
                module = importlib.import_module(dep)
                version = getattr(module, "__version__", "built-in")
                available_versions[dep] = version
            except ImportError:
                missing_required.append(dep)
        
        # Check critical optional dependencies (needed for core functionality)
        for dep in critical_optional:
            try:
                module = importlib.import_module(dep)
                version = getattr(module, "__version__", "unknown")
                available_versions[dep] = version
            except ImportError:
                missing_critical.append(dep)
        
        # Check truly optional dependencies (feature enhancements)
        for dep in optional_deps:
            try:
                module = importlib.import_module(dep)
                version = getattr(module, "__version__", "unknown")
                available_versions[dep] = version
            except ImportError:
                missing_optional.append(dep)
        
        # Determine status based on what's missing
        if missing_required:
            return DiagnosticResult(
                name="Core Dependencies",
                status="FAIL",
                message=f"Missing required dependencies: {', '.join(missing_required)}",
                details={
                    "missing_required": missing_required,
                    "missing_critical": missing_critical,
                    "missing_optional": missing_optional,
                    "available": available_versions
                }
            )
        
        if missing_critical:
            return DiagnosticResult(
                name="Core Dependencies",
                status="WARN",
                message=f"Missing critical dependencies: {', '.join(missing_critical)}",
                details={
                    "missing_critical": missing_critical,
                    "missing_optional": missing_optional,
                    "available": available_versions
                }
            )
        
        # All required and critical deps available - optional deps don't affect status
        message = "All required dependencies available"
        if missing_optional:
            message += f" (optional features unavailable: {', '.join(missing_optional)})"
        
        return DiagnosticResult(
            name="Core Dependencies",
            status="PASS",
            message=message,
            details={
                "missing_optional": missing_optional,
                "available": available_versions,
                "note": "Optional dependencies don't affect core functionality"
            }
        )
    
    def check_package_importability(self) -> DiagnosticResult:
        """Check if the package can be imported correctly."""
        import_errors = []
        import_warnings = []
        
        # Core imports
        core_imports = [
            "chuk_tool_registry",
            "chuk_tool_registry.core",
            "chuk_tool_registry.core.interface",
            "chuk_tool_registry.core.metadata", 
            "chuk_tool_registry.core.provider",
            "chuk_tool_registry.core.registration",
            "chuk_tool_registry.core.exceptions",
            "chuk_tool_registry.discovery",
            "chuk_tool_registry.providers",
            "chuk_tool_registry.providers.memory",
        ]
        
        for module_name in core_imports:
            try:
                importlib.import_module(module_name)
                self.log(f"Successfully imported {module_name}")
            except ImportError as e:
                import_errors.append(f"{module_name}: {str(e)}")
            except Exception as e:
                import_warnings.append(f"{module_name}: {str(e)}")
        
        if import_errors:
            return DiagnosticResult(
                name="Package Imports",
                status="FAIL",
                message=f"Failed to import core modules: {len(import_errors)} errors",
                details={
                    "import_errors": import_errors,
                    "import_warnings": import_warnings
                }
            )
        
        status = "PASS" if not import_warnings else "WARN"
        message = f"All core modules imported successfully"
        if import_warnings:
            message += f" ({len(import_warnings)} warnings)"
        
        return DiagnosticResult(
            name="Package Imports",
            status=status,
            message=message,
            details={"import_warnings": import_warnings}
        )
    
    async def check_core_functionality(self) -> DiagnosticResult:
        """Test core registry functionality."""
        try:
            # Import required components
            from chuk_tool_registry.core.provider import get_registry, set_registry
            from chuk_tool_registry.providers.memory import InMemoryToolRegistry
            from chuk_tool_registry.discovery import register_fn_tool
            
            # Create test registry
            test_registry = InMemoryToolRegistry()
            original_registry = None
            
            try:
                # Backup original registry
                try:
                    original_registry = await get_registry()
                except:
                    pass
                
                # Set test registry
                await set_registry(test_registry)
                
                # Test tool registration
                async def test_tool(x: int) -> int:
                    return x * 2
                
                await register_fn_tool(test_tool, name="test_doubler", namespace="test")
                
                # Test tool retrieval
                registry = await get_registry()
                tool = await registry.get_tool("test_doubler", "test")
                
                if tool is None:
                    raise Exception("Tool not found after registration")
                
                # Test tool execution
                result = await tool.execute(x=5)
                if result != 10:
                    raise Exception(f"Tool execution failed: expected 10, got {result}")
                
                # Test metadata
                metadata = await registry.get_metadata("test_doubler", "test")
                if metadata is None:
                    raise Exception("Metadata not found")
                
                # Test listing
                tools = await registry.list_tools("test")
                if ("test", "test_doubler") not in tools:
                    raise Exception("Tool not found in listing")
                
                return DiagnosticResult(
                    name="Core Functionality",
                    status="PASS",
                    message="All core operations working correctly",
                    details={
                        "test_result": result,
                        "metadata_found": metadata is not None,
                        "tools_listed": len(tools)
                    }
                )
                
            finally:
                # Restore original registry
                if original_registry:
                    await set_registry(original_registry)
                
        except Exception as e:
            return DiagnosticResult(
                name="Core Functionality",
                status="FAIL",
                message=f"Core functionality test failed: {str(e)}",
                exception=e
            )
    
    async def check_registration_system(self) -> DiagnosticResult:
        """Test the registration management system."""
        try:
            from chuk_tool_registry.core.registration import (
                create_registration_manager, ensure_registrations
            )
            from chuk_tool_registry.discovery import register_tool
            
            # Create isolated registration manager
            manager = create_registration_manager("diagnostic_test")
            
            # Test class registration
            @register_tool("test_calculator", namespace="test", manager=manager)
            class TestCalculator:
                async def execute(self, a: int, b: int) -> int:
                    return a + b
            
            # Check pending registrations
            pending_count = manager.get_pending_count()
            if pending_count == 0:
                raise Exception("No pending registrations found")
            
            # Process registrations
            result = await manager.process_registrations()
            
            if result["processed"] != 1:
                raise Exception(f"Expected 1 processed registration, got {result['processed']}")
            
            if result["errors"]:
                raise Exception(f"Registration errors: {result['errors']}")
            
            return DiagnosticResult(
                name="Registration System",
                status="PASS",
                message="Registration system working correctly",
                details={
                    "pending_before": pending_count,
                    "processed": result["processed"],
                    "errors": result["errors"]
                }
            )
            
        except Exception as e:
            return DiagnosticResult(
                name="Registration System",
                status="FAIL",
                message=f"Registration system test failed: {str(e)}",
                exception=e
            )
    
    async def check_performance_metrics(self) -> DiagnosticResult:
        """Check performance of core operations."""
        try:
            from chuk_tool_registry.providers.memory import InMemoryToolRegistry
            from chuk_tool_registry.discovery import register_fn_tool
            from chuk_tool_registry.core.provider import set_registry
            
            # Create test registry
            registry = InMemoryToolRegistry()
            await set_registry(registry)
            
            # Performance test: registration
            async def dummy_tool(x: int) -> int:
                return x
            
            start_time = time.time()
            for i in range(100):
                await register_fn_tool(dummy_tool, name=f"tool_{i}", namespace="perf_test")
            registration_time = (time.time() - start_time) * 1000
            
            # Performance test: retrieval
            start_time = time.time()
            for i in range(100):
                await registry.get_tool(f"tool_{i}", "perf_test")
            retrieval_time = (time.time() - start_time) * 1000
            
            # Performance test: listing
            start_time = time.time()
            tools = await registry.list_tools("perf_test")
            listing_time = (time.time() - start_time) * 1000
            
            metrics = {
                "registration_time_per_tool_ms": registration_time / 100,
                "retrieval_time_per_tool_ms": retrieval_time / 100,
                "listing_time_ms": listing_time,
                "total_tools_registered": 100,
                "tools_found": len(tools)
            }
            
            # Performance thresholds
            warnings = []
            if metrics["registration_time_per_tool_ms"] > 10:
                warnings.append("Slow registration performance")
            if metrics["retrieval_time_per_tool_ms"] > 1:
                warnings.append("Slow retrieval performance")
            if metrics["listing_time_ms"] > 50:
                warnings.append("Slow listing performance")
            
            status = "PASS" if not warnings else "WARN"
            message = "Performance metrics within acceptable range"
            if warnings:
                message = f"Performance issues detected: {', '.join(warnings)}"
            
            return DiagnosticResult(
                name="Performance Metrics",
                status=status,
                message=message,
                details=metrics
            )
            
        except Exception as e:
            return DiagnosticResult(
                name="Performance Metrics",
                status="FAIL",
                message=f"Performance test failed: {str(e)}",
                exception=e
            )
    
    def check_environment_configuration(self) -> DiagnosticResult:
        """Check environment configuration and settings."""
        config = {}
        warnings = []
        
        # Check environment variables
        env_vars = [
            "CHUK_TOOL_REGISTRY_PROVIDER",
            "CHUK_TOOL_REGISTRY_DEBUG",
            "PYTHONPATH"
        ]
        
        for var in env_vars:
            value = os.environ.get(var)
            config[var] = value
            if var == "CHUK_TOOL_REGISTRY_DEBUG" and value:
                self.log(f"Debug mode enabled: {value}")
        
        # Check current working directory
        config["cwd"] = str(Path.cwd())
        config["python_executable"] = sys.executable
        
        # Check if running in virtual environment
        config["in_venv"] = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        
        if not config["in_venv"]:
            warnings.append("Not running in virtual environment")
        
        # Check sys.path for package location
        package_paths = [p for p in sys.path if "chuk_tool_registry" in p]
        config["package_paths"] = package_paths
        
        status = "PASS" if not warnings else "WARN"
        message = "Environment configuration checked"
        if warnings:
            message += f" (warnings: {', '.join(warnings)})"
        
        return DiagnosticResult(
            name="Environment Config",
            status=status,
            message=message,
            details=config
        )
    
    async def check_async_compatibility(self) -> DiagnosticResult:
        """Check async/await compatibility and event loop handling."""
        try:
            # Check event loop
            loop = asyncio.get_running_loop()
            loop_type = type(loop).__name__
            
            # Test basic async operations
            async def async_test():
                await asyncio.sleep(0.001)
                return "async_works"
            
            result = await async_test()
            
            # Test concurrent operations
            async def concurrent_test():
                tasks = [asyncio.sleep(0.001) for _ in range(10)]
                await asyncio.gather(*tasks)
                return "concurrent_works"
            
            concurrent_result = await concurrent_test()
            
            # Test asyncio locks
            lock = asyncio.Lock()
            async with lock:
                lock_test = "locks_work"
            
            return DiagnosticResult(
                name="Async Compatibility",
                status="PASS",
                message="Async operations working correctly",
                details={
                    "event_loop_type": loop_type,
                    "async_test": result,
                    "concurrent_test": concurrent_result,
                    "lock_test": lock_test
                }
            )
            
        except Exception as e:
            return DiagnosticResult(
                name="Async Compatibility",
                status="FAIL",
                message=f"Async compatibility test failed: {str(e)}",
                exception=e
            )
    
    # ------------------------------------------------------------------ #
    # Main Diagnostic Runner
    # ------------------------------------------------------------------ #
    
    async def run_full_diagnostic(self) -> Dict[str, Any]:
        """Run the complete diagnostic suite."""
        print("🔍 Chuk Tool Registry - Package Diagnostic")
        print("=" * 50)
        
        # Run all diagnostic checks
        checks = [
            ("Python Version", self.check_python_version),
            ("Core Dependencies", self.check_core_dependencies),
            ("Package Imports", self.check_package_importability),
            ("Core Functionality", self.check_core_functionality),
            ("Registration System", self.check_registration_system),
            ("Performance Metrics", self.check_performance_metrics),
            ("Environment Config", self.check_environment_configuration),
            ("Async Compatibility", self.check_async_compatibility),
        ]
        
        for name, check_func in checks:
            result = await self.run_check(name, check_func)
            self.add_result(result)
        
        # Generate summary
        total_time = time.time() - self.start_time
        summary = self.generate_summary(total_time)
        
        print("\n" + "=" * 50)
        print("📊 DIAGNOSTIC SUMMARY")
        print("=" * 50)
        print(f"Total Checks: {summary['total_checks']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"⚠️  Warnings: {summary['warnings']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"⏭️  Skipped: {summary['skipped']}")
        print(f"⏱️  Total Time: {total_time:.2f}s")
        print(f"🎯 Success Rate: {summary['success_rate']:.1f}%")
        
        if summary['actionable_recommendations']:
            print("\n🔧 ACTIONABLE RECOMMENDATIONS:")
            for rec in summary['actionable_recommendations']:
                print(f"  • {rec}")
        
        if summary['optional_recommendations']:
            print("\n💡 OPTIONAL ENHANCEMENTS:")
            for rec in summary['optional_recommendations']:
                print(f"  • {rec}")
        
        return summary
    
    def generate_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate diagnostic summary."""
        status_counts = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0}
        actionable_recommendations = []
        optional_recommendations = []
        
        for result in self.results:
            status_counts[result.status] += 1
            
            # Generate recommendations based on results
            if result.status == "FAIL":
                actionable_recommendations.append(f"Fix {result.name}: {result.message}")
            elif result.status == "WARN":
                # Distinguish between actionable warnings and optional features
                if "optional" in result.message.lower() or "langchain" in result.message.lower():
                    # This is about optional features, not critical issues
                    if "langchain" in result.message.lower():
                        optional_recommendations.append("Install LangChain for enhanced tool integration: pip install langchain")
                    else:
                        optional_recommendations.append(f"Consider {result.name}: {result.message}")
                else:
                    # This is a real warning that should be addressed
                    actionable_recommendations.append(f"Address {result.name}: {result.message}")
        
        # Calculate success rate based only on PASS/FAIL, not warnings about optional features
        critical_checks = status_counts["PASS"] + status_counts["FAIL"] + len(actionable_recommendations)
        success_rate = (status_counts["PASS"] / (status_counts["PASS"] + status_counts["FAIL"])) * 100 if (status_counts["PASS"] + status_counts["FAIL"]) > 0 else 100
        
        return {
            "total_checks": len(self.results),
            "passed": status_counts["PASS"],
            "failed": status_counts["FAIL"],
            "warnings": status_counts["WARN"],
            "skipped": status_counts["SKIP"],
            "success_rate": success_rate,
            "total_time_seconds": total_time,
            "actionable_recommendations": actionable_recommendations,
            "optional_recommendations": optional_recommendations,
            "results": [asdict(result) for result in self.results],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    def export_report(self, filepath: str) -> None:
        """Export diagnostic report to JSON file."""
        summary = self.generate_summary(time.time() - self.start_time)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📄 Diagnostic report exported to: {filepath}")


# ------------------------------------------------------------------ #
# Command Line Interface
# ------------------------------------------------------------------ #

async def main():
    """Main entry point for the diagnostic script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Chuk Tool Registry Package Diagnostic Tool"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--export-report",
        type=str,
        help="Export diagnostic report to JSON file"
    )
    parser.add_argument(
        "--fix-issues",
        action="store_true",
        help="Attempt to fix common issues automatically"
    )
    
    args = parser.parse_args()
    
    # Run diagnostic
    diagnostic = PackageDiagnostic(verbose=args.verbose)
    summary = await diagnostic.run_full_diagnostic()
    
    # Export report if requested
    if args.export_report:
        diagnostic.export_report(args.export_report)
    
    # Auto-fix issues if requested
    if args.fix_issues:
        print("\n🔧 AUTO-FIX MODE (Not implemented yet)")
        print("This feature will be available in future versions.")
    
    # Exit with appropriate code
    if summary["failed"] > 0:
        print(f"\n❌ Diagnostic completed with {summary['failed']} failures")
        sys.exit(1)
    elif summary["actionable_recommendations"]:
        print(f"\n⚠️  Diagnostic completed with {len(summary['actionable_recommendations'])} actionable issues")
        sys.exit(0)
    else:
        print(f"\n✅ All diagnostics passed successfully!")
        if summary["optional_recommendations"]:
            print("💡 Consider optional enhancements listed above for additional features.")
        sys.exit(0)


if __name__ == "__main__":
    # Handle Windows event loop policy
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Diagnostic interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Diagnostic failed with unexpected error: {e}")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            traceback.print_exc()
        sys.exit(1)
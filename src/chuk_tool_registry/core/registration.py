# chuk_tool_registry/core/registration.py
"""
Tool registration management for the async-native tool registry.

This module provides the ToolRegistrationManager class and related functionality
for managing tool registrations in a clean, organized way without global state.
"""

import asyncio
import weakref
import atexit
import warnings
from typing import Any, Callable, List, Awaitable, Type, Optional, Dict, Set
from dataclasses import dataclass


@dataclass
class ToolRegistrationInfo:
    """Information about a tool's registration."""
    name: str
    namespace: str
    metadata: Dict[str, Any]
    tool_class: Optional[Type] = None


class ToolRegistrationManager:
    """
    Manages tool registrations without global state.
    
    This class provides a clean alternative to global registration lists,
    allowing for isolated registration contexts and better testability.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize a new registration manager.
        
        Args:
            name: Optional name for the manager (useful for debugging)
        """
        self.name = name or f"RegistrationManager-{id(self)}"
        self.pending_registrations: List[Callable[[], Awaitable]] = []
        self.registered_classes = weakref.WeakSet()
        self.registration_info: Dict[Type, ToolRegistrationInfo] = {}
        self._shutting_down = False
        self._processed_count = 0
    
    def add_registration(
        self, 
        registration_fn: Callable[[], Awaitable], 
        tool_class: Type,
        registration_info: ToolRegistrationInfo
    ) -> None:
        """Add a pending registration."""
        if self._shutting_down:
            warnings.warn(f"Registration manager {self.name} is shutting down, ignoring registration")
            return
        
        self.pending_registrations.append(registration_fn)
        self.registered_classes.add(tool_class)
        self.registration_info[tool_class] = registration_info
    
    async def process_registrations(self) -> Dict[str, Any]:
        """Process all pending tool registrations."""
        if not self.pending_registrations:
            return {
                "processed": 0,
                "total_processed": self._processed_count,
                "pending": 0,
                "errors": [],
                "manager": self.name
            }
        
        tasks = []
        errors = []
        
        for registration_fn in self.pending_registrations:
            try:
                tasks.append(asyncio.create_task(registration_fn()))
            except Exception as e:
                errors.append(str(e))
        
        processed_count = len(self.pending_registrations)
        self.pending_registrations.clear()
        
        # Execute all registration tasks
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    errors.append(str(result))
        
        self._processed_count += processed_count
        
        return {
            "processed": processed_count,
            "total_processed": self._processed_count,
            "pending": len(self.pending_registrations),
            "errors": errors,
            "manager": self.name
        }
    
    def is_registered(self, tool_class: Type) -> bool:
        """Check if a tool class is already registered with this manager."""
        return tool_class in self.registered_classes
    
    def get_registration_info(self, tool_class: Type) -> Optional[ToolRegistrationInfo]:
        """Get registration information for a tool class."""
        return self.registration_info.get(tool_class)
    
    def get_pending_count(self) -> int:
        """Get the number of pending registrations."""
        return len(self.pending_registrations)
    
    def get_registered_classes(self) -> Set[Type]:
        """Get all registered classes (as a set)."""
        return set(self.registered_classes)
    
    def shutdown(self) -> None:
        """Mark the manager as shutting down and clear all pending registrations."""
        self._shutting_down = True
        self.pending_registrations.clear()
        warnings.filterwarnings("ignore", message="coroutine.*was never awaited")
    
    def __str__(self) -> str:
        """String representation of the manager."""
        return (
            f"ToolRegistrationManager({self.name}, "
            f"pending={len(self.pending_registrations)}, "
            f"registered={len(self.registered_classes)}, "
            f"total_processed={self._processed_count})"
        )


class GlobalRegistrationManager:
    """Singleton wrapper for the global registration manager."""
    
    _instance: Optional[ToolRegistrationManager] = None
    _lock = asyncio.Lock()
    
    @classmethod
    async def get_instance(cls) -> ToolRegistrationManager:
        """Get the global registration manager instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = ToolRegistrationManager("Global")
        return cls._instance
    
    @classmethod
    async def reset(cls) -> None:
        """Reset the global manager (useful for testing)."""
        async with cls._lock:
            if cls._instance:
                cls._instance.shutdown()
            cls._instance = None
    
    @classmethod
    async def process_global_registrations(cls) -> Dict[str, Any]:
        """Process registrations in the global manager."""
        manager = await cls.get_instance()
        return await manager.process_registrations()


# Factory functions
def create_registration_manager(name: Optional[str] = None) -> ToolRegistrationManager:
    """Create a new registration manager."""
    return ToolRegistrationManager(name)


async def get_global_registration_manager() -> ToolRegistrationManager:
    """Get the global registration manager."""
    return await GlobalRegistrationManager.get_instance()


# Convenience functions
async def ensure_registrations(manager: Optional[ToolRegistrationManager] = None) -> Dict[str, Any]:
    """Process all pending tool registrations."""
    if manager is None:
        return await GlobalRegistrationManager.process_global_registrations()
    else:
        return await manager.process_registrations()


async def reset_global_registrations() -> None:
    """Reset the global registration manager (useful for testing)."""
    await GlobalRegistrationManager.reset()


async def get_registration_statistics() -> Dict[str, Any]:
    """Get statistics about the global registration manager."""
    global_manager = await get_global_registration_manager()
    
    return {
        "manager_name": global_manager.name,
        "pending_registrations": global_manager.get_pending_count(),
        "registered_classes": len(global_manager.get_registered_classes()),
        "total_processed": global_manager._processed_count,
        "is_shutting_down": global_manager._shutting_down,
    }


def validate_registration_manager(manager: Any) -> ToolRegistrationManager:
    """Validate that an object is a ToolRegistrationManager."""
    if not isinstance(manager, ToolRegistrationManager):
        raise TypeError(f"Expected ToolRegistrationManager, got {type(manager)}")
    return manager


# Global shutdown handling
def _handle_global_shutdown():
    """Handle application shutdown by cleaning up the global manager."""
    if GlobalRegistrationManager._instance:
        GlobalRegistrationManager._instance.shutdown()


atexit.register(_handle_global_shutdown)
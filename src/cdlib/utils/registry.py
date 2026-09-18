"""Generic registry pattern for pluggable components.

The registry pattern is the core mechanism that stops five people colliding.
Adding a component = writing one new file + one registry line. Never editing
trainer.py, build.py, or another person's component file.

Usage:
    # In registry definition (e.g., models/encoders/registry.py):
    from cdlib.utils.registry import Registry
    ENCODER_REGISTRY = Registry("ENCODER")

    # In a component file (e.g., models/encoders/resnet.py):
    from cdlib.models.encoders.registry import ENCODER_REGISTRY

    @ENCODER_REGISTRY.register("resnet18")
    class ResNet18Encoder(nn.Module):
        ...

    # In a builder (e.g., models/build.py):
    encoder = ENCODER_REGISTRY.build("resnet18", **kwargs)
"""

from __future__ import annotations

from typing import Any, Callable


class Registry:
    """A generic registry mapping string keys to callables (classes or factory functions).

    Features:
        - Duplicate key detection (raises on conflict, not silent overwrite)
        - Decorator and explicit registration
        - Build method that instantiates with kwargs
        - Introspection (list keys, check membership)
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: dict[str, Callable[..., Any]] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, key: str) -> Callable:
        """Decorator to register a class or function under `key`.

        Raises:
            KeyError: If `key` is already registered (no silent overwrites).

        Example:
            @MY_REGISTRY.register("my_component")
            class MyComponent:
                ...
        """

        def decorator(cls_or_fn: Callable) -> Callable:
            if key in self._registry:
                raise KeyError(
                    f"[{self._name}] Duplicate key '{key}': "
                    f"already registered to {self._registry[key].__qualname__}. "
                    f"Cannot register {cls_or_fn.__qualname__}."
                )
            self._registry[key] = cls_or_fn
            return cls_or_fn

        return decorator

    def register_explicit(self, key: str, cls_or_fn: Callable) -> None:
        """Non-decorator registration for cases where the decorator pattern is awkward."""
        if key in self._registry:
            raise KeyError(
                f"[{self._name}] Duplicate key '{key}': "
                f"already registered to {self._registry[key].__qualname__}."
            )
        self._registry[key] = cls_or_fn

    def build(self, key: str, **kwargs: Any) -> Any:
        """Instantiate the registered callable with the given kwargs.

        Raises:
            KeyError: If `key` is not registered.
        """
        if key not in self._registry:
            raise KeyError(
                f"[{self._name}] Key '{key}' not found. "
                f"Available keys: {sorted(self._registry.keys())}"
            )
        return self._registry[key](**kwargs)

    def get(self, key: str) -> Callable:
        """Return the registered callable without instantiating.

        Raises:
            KeyError: If `key` is not registered.
        """
        if key not in self._registry:
            raise KeyError(
                f"[{self._name}] Key '{key}' not found. "
                f"Available keys: {sorted(self._registry.keys())}"
            )
        return self._registry[key]

    def keys(self) -> list[str]:
        """Return sorted list of all registered keys."""
        return sorted(self._registry.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return f"Registry(name='{self._name}', keys={self.keys()})"

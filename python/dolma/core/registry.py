from typing import Callable, Dict, Generator, Generic, Tuple, Type, TypeVar

from .taggers import BaseTagger

T = TypeVar("T", bound=Type)
R = TypeVar("R", bound=Type)


class BaseRegistry(Generic[T]):
    """A registry for objects."""

    __all_registries: Dict[str, "BaseRegistry"] = {}

    def __new__(cls):
        # enforce singleton pattern for each registry
        if cls.__name__ not in cls.__all_registries:
            cls.__all_registries[cls.__name__] = super().__new__(cls)
        return cls.__all_registries[cls.__name__]

    def __init__(self) -> None:
        self.__registry: Dict[str, T] = {}

    @classmethod
    def items(cls) -> Generator[Tuple[str, T], None, None]:
        """Yield all items in the registry."""
        registry = cls()
        yield from ((s, registry.__registry[s]) for s in sorted(registry.__registry))

    @classmethod
    def add(cls, name: str) -> Callable[[R], R]:
        """Add a class to the registry."""
        registry = cls()

        def _add(tagger_self: T, tagger_name: str = name, taggers_dict: Dict[str, T] = registry.__registry) -> T:
            """Add a tagger to the registry using tagger_name as the name."""
            if tagger_name in taggers_dict and taggers_dict[tagger_name] != tagger_self:
                if tagger_self.__module__ == "__main__":
                    return tagger_self

                raise ValueError(f"Tagger {tagger_name} already exists")

            taggers_dict[tagger_name] = tagger_self
            return tagger_self

        return _add  # type: ignore

    @classmethod
    def remove(cls, name: str) -> bool:
        """Remove a tagger from the registry."""
        if name in (registry := cls()).__registry:
            registry.__registry.pop(name)
            return True
        return False

    @classmethod
    def has(cls, name: str) -> bool:
        """Check if a tagger exists in the registry."""
        return name in cls().__registry

    @classmethod
    def get(cls, name: str) -> T:
        """Get a tagger from the registry; raise ValueError if it doesn't exist."""
        if name not in (registry := cls()).__registry:
            tagger_names = ", ".join([tn for tn, _ in registry.items()])
            raise ValueError(f"Unknown tagger {name}; available taggers: {tagger_names}")
        return registry.__registry[name]


class TaggerRegistry(BaseRegistry[Type[BaseTagger]]):
    pass

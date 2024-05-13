from typing import Callable, Dict, Generator, Generic, Optional, Tuple, Type, TypeVar

from .taggers import BaseTagger

T = TypeVar("T")
R = TypeVar("R")


class BaseRegistry(Generic[T]):
    """A registry for objects."""

    _registry_of_registries: Dict[str, Type["BaseRegistry"]] = {}
    _registry_storage: Dict[str, Tuple[T, Optional[str]]]

    @classmethod
    def _add_to_registry_of_registries(cls) -> None:
        name = cls.__name__
        if name not in cls._registry_of_registries:
            cls._registry_of_registries[name] = cls

    @classmethod
    def registries(cls) -> Generator[Tuple[str, Type["BaseRegistry"]], None, None]:
        """Yield all registries in the registry of registries."""
        yield from sorted(cls._registry_of_registries.items())

    @classmethod
    def _get_storage(cls) -> Dict[str, Tuple[T, Optional[str]]]:
        if not hasattr(cls, "_registry_storage"):
            cls._registry_storage = {}
        return cls._registry_storage  # pyright: ignore

    @classmethod
    def items(cls) -> Generator[Tuple[str, T], None, None]:
        """Yield all items in the registry."""
        yield from sorted((n, t) for (n, (t, _)) in cls._get_storage().items())

    @classmethod
    def items_with_description(cls) -> Generator[Tuple[str, T, Optional[str]], None, None]:
        """Yield all items in the registry with their descriptions."""
        yield from sorted((n, t, d) for (n, (t, d)) in cls._get_storage().items())

    @classmethod
    def add(cls, name: str, desc: Optional[str] = None) -> Callable[[T], T]:
        """Add a class to the registry."""

        # Add the registry to the registry of registries
        cls._add_to_registry_of_registries()

        def _add(
            tagger_self: T,
            tagger_name: str = name,
            tagger_desc: Optional[str] = desc,
            tagger_cls: Type[BaseRegistry] = cls,
        ) -> T:
            """Add a tagger to the registry using tagger_name as the name."""
            if tagger_name in tagger_cls._get_storage() and tagger_cls.get(tagger_name) != tagger_self:
                if tagger_self.__module__ == "__main__":
                    return tagger_self

                raise ValueError(f"Tagger {tagger_name} already exists")
            tagger_cls._get_storage()[tagger_name] = (tagger_self, tagger_desc)
            return tagger_self

        return _add

    @classmethod
    def remove(cls, name: str) -> bool:
        """Remove a tagger from the registry."""
        if name in cls._get_storage():
            cls._get_storage().pop(name)
            return True
        return False

    @classmethod
    def has(cls, name: str) -> bool:
        """Check if a tagger exists in the registry."""
        return name in cls._get_storage()

    @classmethod
    def get(cls, name: str) -> T:
        """Get a tagger from the registry; raise ValueError if it doesn't exist."""
        if name not in cls._get_storage():
            tagger_names = ", ".join([tn for tn, _ in cls.items()])
            raise ValueError(f"Unknown tagger {name}; available taggers: {tagger_names}")
        t, _ = cls._get_storage()[name]
        return t


class TaggerRegistry(BaseRegistry[Type[BaseTagger]]):
    pass

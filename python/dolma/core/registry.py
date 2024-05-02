from typing import Callable, Dict, Generator, Generic, Tuple, Type, TypeVar

from .taggers import BaseTagger

T = TypeVar("T", bound=Type)
R = TypeVar("R", bound=Type)


class BaseRegistry(Generic[T]):
    """A registry for objects."""

    _registry_storage: Dict[str, T]

    @classmethod
    def _get_storage(cls) -> Dict[str, T]:
        if not hasattr(cls, "_registry_storage"):
            cls._registry_storage = {}
        return cls._registry_storage  # pyright: ignore

    @classmethod
    def items(cls) -> Generator[Tuple[str, T], None, None]:
        """Yield all items in the registry."""
        yield from sorted(cls._get_storage().items())

    @classmethod
    def add(cls, name: str) -> Callable[[R], R]:
        """Add a class to the registry."""

        def _add(tagger_self: T, tagger_name: str = name, cls_: Type[BaseRegistry] = cls) -> T:
            """Add a tagger to the registry using tagger_name as the name."""
            if tagger_name in cls_._get_storage() and cls_._get_storage()[tagger_name] != tagger_self:
                if tagger_self.__module__ == "__main__":
                    return tagger_self

                raise ValueError(f"Tagger {tagger_name} already exists")
            cls_._get_storage()[tagger_name] = tagger_self
            return tagger_self

        return _add  # type: ignore

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
        return cls._get_storage()[name]


class TaggerRegistry(BaseRegistry[Type[BaseTagger]]):
    pass

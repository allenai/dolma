from typing import Callable, Dict, Generator, Tuple, Type, TypeVar

from .taggers import BaseTagger

T = TypeVar("T", bound=BaseTagger)


class TaggerRegistry:
    """A registry for taggers."""

    __taggers: Dict[str, Type[BaseTagger]] = {}

    @classmethod
    def taggers(cls) -> Generator[Tuple[str, Type[BaseTagger]], None, None]:
        """Yield all taggers."""
        yield from cls.__taggers.items()

    @classmethod
    def add(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        """Add a tagger to the registry."""

        def _add(
            tagger_cls: Type[T],
            tagger_name: str = name,
            taggers_dict: Dict[str, Type[BaseTagger]] = cls.__taggers,
        ) -> Type[T]:
            """Add a tagger to the registry using tagger_name as the name."""
            if tagger_name in taggers_dict and taggers_dict[tagger_name] != tagger_cls:
                if tagger_cls.__module__ == "__main__":
                    return tagger_cls

                raise ValueError(f"Tagger {tagger_name} already exists")

            taggers_dict[tagger_name] = tagger_cls
            return tagger_cls

        return _add

    @classmethod
    def remove(cls, name: str) -> bool:
        """Remove a tagger from the registry."""
        if name in cls.__taggers:
            cls.__taggers.pop(name)
            return True
        return False

    @classmethod
    def has(cls, name: str) -> bool:
        """Check if a tagger exists in the registry."""
        return name in cls.__taggers

    @classmethod
    def get(cls, name: str) -> Type[BaseTagger]:
        """Get a tagger from the registry; raise ValueError if it doesn't exist."""
        if name not in cls.__taggers:
            raise ValueError(
                f"Unknown tagger {name}; available taggers: " + ", ".join([tn for tn, _ in cls.taggers()])
            )
        return cls.__taggers[name]

import unittest
from typing import Type

from dolma.core.registry import BaseRegistry


class TestNewRegistry(unittest.TestCase):
    def make_registry(self) -> Type[BaseRegistry]:
        class NewRegistry(BaseRegistry):
            pass

        return NewRegistry

    def test_add(self) -> None:
        registry = self.make_registry()

        # try adding one
        registry.add("test1")(str)
        self.assertTrue(registry.has("test1"))

        # check no overwrite
        registry.add("test2")(str)
        self.assertTrue(registry.has("test1"))
        self.assertTrue(registry.has("test2"))

        # this should not raise an error because it's the same class
        registry.add("test1")(str)

        with self.assertRaises(ValueError):
            # different class; should raise an error
            registry.add("test1")(int)

    def test_remove(self) -> None:
        registry = self.make_registry()
        registry.add("test1")(str)
        self.assertTrue(registry.has("test1"))

        registry.remove("test1")
        self.assertFalse(registry.has("test1"))

    def test_get(self) -> None:
        registry = self.make_registry()
        registry.add("test1")(str)

        self.assertEqual(registry.get("test1"), str)

        with self.assertRaises(ValueError):
            registry.get("test2")

    def test_list(self) -> None:
        registry = self.make_registry()
        registry.add("test1")(str)
        registry.add("test2")(str)

        self.assertEqual(list(registry.items()), [("test1", str), ("test2", str)])

    def test_no_overlap(self) -> None:
        registry1 = self.make_registry()
        registry2 = self.make_registry()

        registry1.add("test1")(str)
        registry2.add("test2")(str)

        self.assertTrue(registry1.has("test1"))
        self.assertFalse(registry1.has("test2"))

        self.assertTrue(registry2.has("test2"))
        self.assertFalse(registry2.has("test1"))

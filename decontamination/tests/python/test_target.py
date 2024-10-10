from dolma_decontamination.tasks.base import Target

from unittest import TestCase



class TestTargetSelection(TestCase):
    def setUp(self):
        self.data = [
            {"id": 1, "text": "Hello, world!"},
            {"id": 2, "text": "This is a test."},
            {"id": 3, "text": "JQ is fun!"},
        ]

    def test_target_selection(self):
        target = Target(expression=".text", label="text")
        breakpoint()

from unittest import TestCase

from dolma.models.data import _make_selector, make_fasttext_data


class TestSelector(TestCase):
    def test_selector(self):
        d = {"a": [{"b": 1}, {"c": [2, {"d": 3}], "e": 4}, {"f": 5}], "g": 6}
        self.assertEqual(_make_selector("$.a")(d), d["a"])
        self.assertEqual(_make_selector("$.a[1].c")(d), d["a"][1]["c"])
        self.assertEqual(_make_selector("$.a[1].c[1]")(d), d["a"][1]["c"][1])
        self.assertEqual(_make_selector("$.a[1].c[1].d")(d), d["a"][1]["c"][1]["d"])
        self.assertEqual(_make_selector("$.a[1].e")(d), d["a"][1]["e"])
        self.assertEqual(_make_selector("$.g")(d), d["g"])

        with self.assertRaises(TypeError):
            _make_selector("$.a[1].c[1].d[0]")(d)

        with self.assertRaises(KeyError):
            _make_selector("$.z")(d)

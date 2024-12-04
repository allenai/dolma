import unittest

from dolma_classifiers.label.templates import JqTemplate


class TestJqTemplate(unittest.TestCase):
    """Test cases for the JqTemplate class."""

    def setUp(self):
        """Set up test data that will be used across multiple tests."""
        self.test_data = {
            "name": "John",
            "age": 30,
            "address": {"street": "123 Main St", "city": "Springfield"},
            "hobbies": ["reading", "hiking", "coding"],
        }

    def test_basic_expression(self):
        """Test basic template expression."""
        template = JqTemplate("Hello, {.name}!")
        self.assertEqual(template.render(self.test_data), "Hello, John!")

    def test_nested_object_access(self):
        """Test accessing nested object properties."""
        template = JqTemplate("Address: {.address.street}, {.address.city}")
        self.assertEqual(template.render(self.test_data), "Address: 123 Main St, Springfield")

    def test_array_access(self):
        """Test accessing array elements."""
        template = JqTemplate("First hobby: {.hobbies[0]}")
        self.assertEqual(template.render(self.test_data), "First hobby: reading")

    def test_complex_jq_expression(self):
        """Test more complex JQ expressions."""
        template = JqTemplate('Hobbies: {.hobbies | join(", ")}')
        self.assertEqual(template.render(self.test_data), "Hobbies: reading, hiking, coding")

    def test_escaped_braces(self):
        """Test that escaped braces are handled correctly."""
        template = JqTemplate("User {{.name}} is {.age} years old")
        self.assertEqual(template.render(self.test_data), "User {.name} is 30 years old")

    def test_multiple_expressions(self):
        """Test multiple expressions in the same template."""
        template = JqTemplate("{.name} lives at {.address.street}")
        self.assertEqual(template.render(self.test_data), "John lives at 123 Main St")

    def test_missing_field(self):
        """Test behavior when accessing a non-existent field."""
        template = JqTemplate("Name: {.missing_field}")
        self.assertEqual(template.render(self.test_data), "Name: ")

    def test_unmatched_brace(self):
        """Test that unmatched braces raise an error."""
        with self.assertRaises(ValueError):
            JqTemplate("Hello {.name")

    def test_invalid_jq_expression(self):
        """Test that invalid JQ expressions raise an error."""
        with self.assertRaises(ValueError):
            JqTemplate("Hello {invalid!}")

    def test_empty_template(self):
        """Test handling of empty template strings."""
        template = JqTemplate("")
        self.assertEqual(template.render(self.test_data), "")

    def test_template_without_expressions(self):
        """Test template string without any expressions."""
        template = JqTemplate("Hello, world!")
        self.assertEqual(template.render(self.test_data), "Hello, world!")

    def test_adjacent_expressions(self):
        """Test handling of adjacent expressions."""
        template = JqTemplate("{.name}{.age}")
        self.assertEqual(template.render(self.test_data), "John30")

    def test_whitespace_handling(self):
        """Test that whitespace in expressions is handled correctly."""
        template = JqTemplate("Hello, {  .name   }!")
        self.assertEqual(template.render(self.test_data), "Hello, John!")

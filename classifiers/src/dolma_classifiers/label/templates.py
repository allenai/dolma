from typing import Any, Dict, Optional

import jq


class JqTemplate:
    """
    A template engine that processes strings containing JQ expressions in {expression} syntax.
    Supports escaping curly braces with {{ and }}.
    """

    def __init__(self, template_string: str):
        """
        Initialize the template with a template string.

        Args:
            template_string: The template string containing JQ expressions in {expression} syntax
        """
        self.template_string = template_string
        self._compiled = self._compile_template(template_string)

    @staticmethod
    def _compile_template(template_string: str) -> list[tuple[str, Optional[jq.jq]]]:
        """
        Compile the template string into a list of (text, expression) tuples.

        Args:
            template_string: The template string to compile

        Returns:
            List of tuples containing (text, compiled_jq_expression)

        Raises:
            ValueError: If there are unmatched braces or invalid JQ expressions
        """
        parts = []
        current_pos = 0

        # Handle escaped braces first
        template_string = template_string.replace("{{", "\0LEFT_BRACE\0").replace("}}", "\0RIGHT_BRACE\0")

        while current_pos < len(template_string):
            # Find next unescaped opening brace
            start = template_string.find("{", current_pos)

            if start == -1:
                # No more expressions, add remaining text
                text = template_string[current_pos:]
                text = text.replace("\0LEFT_BRACE\0", "{").replace("\0RIGHT_BRACE\0", "}")
                parts.append((text, None))
                break

            # Add text before the expression
            if start > current_pos:
                text = template_string[current_pos:start]
                text = text.replace("\0LEFT_BRACE\0", "{").replace("\0RIGHT_BRACE\0", "}")
                parts.append((text, None))

            # Find matching closing brace
            end = template_string.find("}", start)
            if end == -1:
                raise ValueError(f"Unmatched opening brace at position {start}")

            # Extract and compile JQ expression
            expr = template_string[start + 1:end].strip()
            try:
                compiled_expr = jq.compile(expr)
            except ValueError as e:
                raise ValueError(f"Invalid JQ expression '{expr}': {str(e)}")

            parts.append(("", compiled_expr))
            current_pos = end + 1

        return parts

    def render(self, data: Dict[str, Any]) -> str:
        """
        Render the template by evaluating all JQ expressions against the provided data.

        Args:
            data: Dictionary containing the data to evaluate expressions against

        Returns:
            The rendered template string

        Raises:
            ValueError: If any JQ expression fails to evaluate
        """
        result = []

        for text, expr in self._compiled:
            result.append(text)
            if expr is None:
                continue

            try:
                # Evaluate expression and get first result
                evaluated = expr.input(data).first()
                # append the evaluated result to the result list
                result.append(str(evaluated or ""))
            except StopIteration:
                # No results from JQ expression
                result.append("")
            except Exception as e:
                raise ValueError(f"Error evaluating expression: {str(e)}")

        return "".join(result)

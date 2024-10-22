import shutil
from enum import Enum
from pathlib import Path

from tantivy import Index, SchemaBuilder


class IndexFields(Enum):
    TEXT = "text"
    ID = "id"
    SOURCE = "source"


def create_index(path: str | Path | None = None, reuse: bool = False) -> Index:
    # Declaring our schema.
    schema_builder = SchemaBuilder()
    schema_builder.add_text_field(IndexFields.TEXT.value, stored=True)
    schema_builder.add_text_field(IndexFields.ID.value, stored=True)
    schema_builder.add_text_field(IndexFields.SOURCE.value, stored=True)
    schema = schema_builder.build()

    if path:
        path = Path(path)
        if not reuse and path.exists():
            shutil.rmtree(path)

        path.mkdir(parents=True, exist_ok=True)

    # Creating our index (in memory)
    index = Index(schema, path=str(path), reuse=reuse)
    return index

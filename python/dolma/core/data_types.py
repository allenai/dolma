"""

Data types assumed by Filters.

@kylel, @soldni

"""

import functools
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from msgspec import Struct
from typing_extensions import Self, TypeAlias

TaggerOutputValueType: TypeAlias = Tuple[int, int, float]
TaggerOutputType: TypeAlias = List[TaggerOutputValueType]
TaggerOutputDictType: TypeAlias = Dict[str, TaggerOutputType]


class InputSpec(Struct):
    id: str
    text: str
    source: str = ""
    # created: str = ""
    # added: str = ""
    version: Optional[str] = None


class InputSpecWithMetadata(InputSpec):
    metadata: Optional[Dict[str, Any]] = None


class InputSpecWithMetadataAndAttributes(InputSpecWithMetadata):
    attributes: Optional[Dict[str, List[Tuple[int, int, float]]]] = None


class OutputSpec(Struct):
    id: str
    attributes: Dict[str, List[TaggerOutputValueType]]
    source: Optional[str] = None


class Document:
    __slots__ = "source", "version", "id", "text", "added", "created"
    spec_cls: Type[InputSpec] = InputSpec

    def __init__(
        self,
        source: str,
        id: str,
        text: str,
        version: Optional[str] = None,
        added: Optional[str] = None,
        created: Optional[str] = None,
    ) -> None:
        self.source = source
        self.version = version
        self.id = id
        self.text = text
        self.added = added
        self.created = created

    @classmethod
    def from_spec(cls, spec: InputSpec) -> Self:
        return cls(**{k: v for k in cls.__slots__ if (v := getattr(spec, k)) is not None})

    def to_spec(self) -> InputSpec:
        return self.spec_cls(**{k: v for k in self.__slots__ if (v := getattr(self, k)) is not None})

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> Self:
        return cls(**{k: v for k in cls.__slots__ if (v := d.get(k)) is not None})

    def to_json(self) -> Dict[str, Any]:
        return {k: v for k in self.__slots__ if (v := getattr(self, k)) is not None}

    def __str__(self) -> str:
        attributes_string = ",".join([f"{k}:{repr(v)}" for k, v in self.to_json().items()])
        return f"{self.__class__.__name__}({attributes_string})"


class DocumentWithMetadata(Document):
    __slots__ = Document.__slots__ + ("metadata",)
    spec_cls = InputSpecWithMetadata

    def __init__(self, *args, metadata: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.metadata = metadata or {}

    def __str__(self) -> str:
        repr_ = super().__str__()
        return repr_.rstrip(")") + f",metadata={'...' if self.metadata else 'none'})"


class DocumentWithMetadataAndAttributes(DocumentWithMetadata):
    __slots__ = DocumentWithMetadata.__slots__ + ("attributes",)
    spec_cls = InputSpecWithMetadataAndAttributes

    def __init__(
        self, *args, attributes: Optional[Dict[str, List[Tuple[int, int, float]]]] = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.attributes = attributes or {}

    def __str__(self) -> str:
        return super().__str__().rstrip(")") + f",attributes={'...' if self.attributes else 'none'})"


class Span:
    __slots__ = "start", "end", "type", "score", "experiment", "tagger", "location"

    __selectors_cache__: Dict[str, Callable[["Document"], str]] = {}

    def __init__(
        self,
        start: int,
        end: int,
        type: str,
        score: float = 1.0,
        experiment: Optional[str] = None,
        tagger: Optional[str] = None,
        location: str = "text",
    ):
        self.start = start
        self.end = end
        self.type = type
        self.score = float(score)
        self.experiment = experiment
        self.tagger = tagger
        self.location = location

    def _make_selector(self) -> Callable[["Document"], str]:
        if self.location not in self.__selectors_cache__:

            def _nested_selector(
                doc: Any,
                index: Optional[int] = None,
                key: Optional[str] = None,
                previous: Optional[Callable] = None,
                dict_like: bool = True,
            ) -> Any:
                prev = previous(doc) if previous is not None else doc
                if dict_like or index is not None:
                    assert (key or index) is not None, "Either key or index must be set"
                    return prev[key or index]
                elif key is not None:
                    return getattr(prev, key)
                else:
                    raise ValueError("Either key or index must be set")

            matches = list(
                re.finditer(r"((^|\.)(?P<key>[a-zA-Z][a-zA-Z0-9]*))|(\[(?P<index>[0-9]+)\])", self.location)
            )
            assert len(matches) > 0, f"Invalid location: `{self.location}`"
            init_match, *rest_matches = matches

            fn = functools.partial(
                _nested_selector,
                index=int(init_match.group("index")) if init_match.group("index") is not None else None,
                key=init_match.group("key"),
                dict_like=False,
            )
            for match in rest_matches[::-1]:
                fn = functools.partial(
                    _nested_selector,
                    index=int(match.group("index")) if match.group("index") is not None else None,
                    key=match.group("key"),
                    previous=fn,
                )
            self.__selectors_cache__[self.location] = fn

        return self.__selectors_cache__[self.location]

    def mention(self, text: str, window: int = 0) -> str:
        return text[max(0, self.start - window) : min(len(text), self.end + window)]

    def select(self, doc: Document, left: int = 0, right: int = 0) -> str:
        return self._make_selector()(doc)[self.start - left : self.end + right]

    @classmethod
    def from_spec(cls, attribute_name: str, attribute_value: TaggerOutputValueType) -> Self:
        if "__" in attribute_name:
            # bff tagger has different name
            exp_name, tgr_name, attr_type = attribute_name.split("__", 2)
        else:
            exp_name = tgr_name = attr_type = attribute_name

        start, end, score = attribute_value
        return cls(
            start=int(start),
            end=int(end),
            type=attr_type,
            score=float(score),
            experiment=exp_name,
            tagger=tgr_name,
        )

    def to_spec(self) -> Tuple[str, TaggerOutputValueType]:
        from .utils import format_span_key, format_span_output

        assert self.experiment is not None, "Experiment name must be set to convert to spec"
        assert self.tagger is not None, "Tagger name must be set to convert to spec"
        return format_span_key(self.experiment, self.tagger, self), format_span_output(self)

    def __len__(self) -> int:
        return self.end - self.start

    @classmethod
    def from_json(cls, di: Dict) -> Self:
        return cls(**{k: v for k, v in di.items() if k in cls.__slots__})

    def to_json(self, text: Optional[str] = None, window: int = 0) -> dict:
        span_repr = {"start": self.start, "end": self.end, "type": self.type, "score": self.score}
        if text is not None:
            span_repr["mention"] = self.mention(text=text, window=window)
        return span_repr

    def __str__(self) -> str:
        cls_name = self.__class__.__name__
        return f"{cls_name}(start={self.start},end={self.end},type={repr(self.type)},score={self.score:.5f})"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return (
            self.start == other.start
            and self.end == other.end
            and self.type == other.type
            and self.score == other.score
        )


class DocResult:
    __slots__ = "doc", "spans"

    def __init__(self, doc: Document, spans: List[Span]) -> None:
        self.doc = doc
        self.spans = spans

    @classmethod
    def from_spec(cls, doc: InputSpec, *attrs_groups: OutputSpec) -> "DocResult":
        spans: List[Span] = []
        for attrs in attrs_groups:
            assert doc.id == attrs.id, f"doc.id={doc.id} != attrs.id={attrs.id}"
            spans.extend(
                [
                    Span.from_spec(attribute_name=attr_name, attribute_value=attr_value)
                    for attr_name, attr_values in attrs.attributes.items()
                    for attr_value in attr_values
                ]
            )
        return DocResult(doc=Document.from_spec(doc), spans=spans)

    def to_spec(self) -> Tuple[InputSpec, OutputSpec]:
        doc_spec = self.doc.to_spec()
        attributes: Dict[str, List[TaggerOutputValueType]] = {}

        for span in self.spans:
            attr_name, attr_value = span.to_spec()
            attributes.setdefault(attr_name, []).append(attr_value)

        return doc_spec, OutputSpec(source=self.doc.source, id=self.doc.id, attributes=attributes)

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "DocResult":
        return DocResult(
            doc=Document.from_json(d["doc"]),
            spans=[Span.from_json(span) for span in d["spans"]],
        )

    def to_json(self, with_doc: bool = False, window: int = 0) -> Dict[str, Any]:
        d: Dict[str, Any] = {"spans": [span.to_json(text=self.doc.text, window=window) for span in self.spans]}
        if with_doc:
            d["doc"] = self.doc.to_json()
        return d

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(doc={self.doc},spans=[{','.join(str(s) for s in self.spans)}])"


class TextSlice:
    """A slice of text from a document."""

    __slots__ = "doc", "start", "end"

    def __init__(self, doc: str, start: int, end: int):
        self.doc = doc
        self.start = start
        self.end = end

    @property
    def text(self) -> str:
        return self.doc[self.start : self.end]

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(text={repr(self.text)},start={self.start},end={self.end})"

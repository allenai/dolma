# Writing Your Own Parallel Processor

Many functions in the Dolma toolkit are built on top of `dolma.core.parallel.BaseParallelProcessor`. This class provides a simple interface for parallelizing a function over a list of inputs, as well keeping track of status using one or more progress bars. In this tutorial, we will walk through the process of writing a parallel processor to remove empty documents from a dataset.

At its core, a parallel processor requires implementing two class methods, `process_single` and `increment_progressbar`:

```python
from dolma.core.parallel import BaseParallelProcessor
from queue import Queue


class CustomParallelProcessor(BaseParallelProcessor):
    @classmethod
    def increment_progressbar(
        cls,
        queue: Queue,
        /,
        files: int = 0,
        documents: int = 0,
        ...
    ):
        """
        This method is called in the process_single
        to increment the progress bar.
        You can create as many progress bars as are
        the numbers of arguments after the '/' separator.
        In this example, I have created two progress
        bars, one for files and one for documents.
        The increment progressbar method should call
        the super method with the same arguments.
        """
        super().increment_progressbar(
            queue,
            files=files,
            documents=documents,
            ...
        )

    @classmethod
    def process_single(
        cls,
        source_path: str,
        destination_path: str,
        queue: Queue,
        **kwargs: Any,
    ):
        """
        This method is to process a single input file.
        The method broadly opens source_path file,
        processes it and writes the output to
        destination_path. Every now and then, it
        should call the increment_progressbar method
        to update the progress bar.
        """
        ...
```

Let's dive a bit deeper into one might implement the `process_single` method in the case of removing empty documents.
We assume `source_path` is a path to a either local or remote JSONL gzip'ed file, and use `smart_open` to deal with that.

```python
from contextlib import ExitStack
from typing import Any
from queue import Queue
import json

import smart_open
from dolma.core.parallel import BaseParallelProcessor


class RemoveEmptyDocumentsProcessor(BaseParallelProcessor):
    @classmethod
    def increment_progressbar(
        cls,
        queue: Queue,
        /,
        files: int = 0,
        read_docs: int = 0,
        written_docs: int = 0
    ):
        """
        This method is to update the progress bar. We keep
        track of three things:
        - files: the number of files processed
        - read_docs: the number of documents read in
        - written_docs: the number of documents written out
            (i.e., the number of documents that are not empty)
        """
        super().increment_progressbar(
            queue,
            files=files,
            read_docs=read_docs,
            written_docs=written_docs
        )

    @classmethod
    def process_single(
        cls,
        source_path: str,
        destination_path: str,
        queue: Queue,
        **kwargs: Any,
    ):
        """
        This method is called for each file. It reads the file
        line by line, and writes to the destination file only
        if the document is not empty.
        """

        update_every_n_lines = 10_000
        read_docs = written_docs = 0

        with ExitStack() as stack:
            # open source and destination files
            source_file = stack.enter_context(
                smart_open.open(source_path, "rt")
            )
            destination_file = stack.enter_context(
                smart_open.open(destination_path, "wt")
            )
            for ln in source_file:
                # we first load the json document
                document = json.loads(ln)
                read_docs += 1

                # we check if the document is
                # empty, and if it is, we skip it
                if document['text'].strip() == '':
                    continue

                # if the document is not empty,
                # we write it to output
                destination_file.write(ln)
                written_docs += 1

                # we update the progress bar every
                # update_every_n_lines
                if read_docs % update_every_n_lines == 0:
                    cls.increment_progressbar(
                        queue,
                        read_docs=read_docs,
                        written_docs=written_docs,
                    )

            # we update the progress bar one last time
            cls.increment_progressbar(
                queue,
                files=1,
                read_docs=read_docs,
                written_docs=written_docs,
            )
```

To use this processor, we invoke it as follows:

```python
from tempfile import TemporaryDirectory

with TemporaryDirectory() as tmpdir:
    # create the processor
    processor = RemoveEmptyDocumentsProcessor(
        source_prefix="path/to/source/files/*.gz",
        destination_prefix="path/to/destination/files",
        metadata_prefix=tmpdir
    )

    # run the processor
    processor()
```

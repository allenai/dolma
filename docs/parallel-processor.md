# Writing Your Own Parallel Processor

Many functions in the Dolma toolkit are built on top of `dolma.core.parallel.BaseParallelProcessor`. This class provides a simple interface for parallelizing a function over a list of inputs, as well keeping track of status using one or more progress bars. In this tutorial, we will walk through the process of writing a parallel processor to remove empty documents from a dataset.


At its core, a parallel processor requires implementing two class methods, `process_single` and `increment_progressbar`:

```python
from dolma.core.parallel import BaseParallelProcessor
from queue import Queue

class RemoveEmptyDocumentsProcessor(BaseParallelProcessor):

    @classmethod
    def process_single(
        cls,
        source_path: str,
        destination_path: str,
        queue: Queue,
        **kwargs: Any,
    ):
        """


        """
        ...

```

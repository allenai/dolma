class DolmaError(Exception):
    """Base class for all errors"""


class DolmaFatalError(DolmaError):
    """Fatal error. Abort the entire process"""


class DolmaShardError(DolmaError):
    """Fail the shard and continue"""


class DolmaRetryableFailure(DolmaError):
    """Retry if a shard throws this error"""


class DolmaRustPipelineError(DolmaError):
    """Error raised by the rust pipeline"""


class DolmaConfigError(DolmaError):
    """Error raised while parsing config"""

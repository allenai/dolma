class DolmaFilterError(Exception):
    """Base class for all errors"""


class DolmaFatalError(DolmaFilterError):
    """Fatal error. Abort the entire process"""


class DolmaShardError(DolmaFilterError):
    """Fail the shard and continue"""


class DolmaRetryableFailure(DolmaFilterError):
    """Retry if a shard throws this error"""

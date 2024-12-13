import hashlib

from diskcache import Cache

diskcache = Cache('.cache_dir')


def cache(avoid_fields=None, skip_load=False, fn_name=None):
    """Cache the output of a function call"""

    def _hash(*args, **kwargs):
        """Hash the input arguments"""
        return hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()

    def _cache_key(func_name, *args, **kwargs):
        """Return cache key"""
        return func_name + "_" + _hash(*args, **kwargs)

    def _cache(func_name, **kwargs):
        """Return the cached output if it exists, otherwise return None"""
        cache_file = _cache_key(func_name, **kwargs)
        return diskcache.get(cache_file)

    def _save_cache(func_name, *args, **kwargs):
        """Save the output to the cache file"""
        output = kwargs.pop("output")
        cache_key = _cache_key(func_name, *args, **kwargs)
        diskcache.set(cache_key, output)

    def _decorator(func):
        def _wrapper(*args, **kwargs):
            # convert args and kwargs to a dictionary
            all_kwargs = dict(zip(func.__code__.co_varnames, args))
            all_kwargs.update(kwargs)
            all_kwargs = {k: v for k, v in all_kwargs.items() if k not in avoid_fields} if avoid_fields else all_kwargs

            # Check if the output is cached
            try:
                cached = _cache(func_name=func.__name__, **all_kwargs)
                if cached is not None and not skip_load:
                    return cached
            except Exception as e:
                pass

            # Call the function
            output = func(*args, **kwargs)

            func_name = fn_name or func.__name__
            # Save the output to the cache
            _save_cache(func_name=func_name, **all_kwargs, output=output)

            return output

        return _wrapper

    return _decorator

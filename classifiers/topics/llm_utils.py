import hashlib
import os
import pickle
from dataclasses import dataclass

from dotenv import load_dotenv
from litellm import completion, completion_cost
from diskcache import Cache

load_dotenv()
diskcache = Cache('.cache_dir')


def cache(avoid_fields=None, skip_load=False):
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

            # Save the output to the cache
            _save_cache(func_name=func.__name__, **all_kwargs, output=output)

            return output

        return _wrapper

    return _decorator

@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cost: float


@cache()
def generate_response(model_engine, prompt, stop_tokens=None, max_output_tokens=600, temperature=0.2, top_p=0.5, seed=0) -> LLMResponse:
    """
    seed is implicitly used to allow calling the function more than once without the cache being used
    """
    response = completion(
        model=model_engine,
        messages=prompt,
        max_tokens=max_output_tokens,
        stop=stop_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    text = response.choices[0].message.content.strip()
    cost = completion_cost(completion_response=response, model=model_engine, messages=prompt)

    return LLMResponse(
        text=text,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        cost=cost,
    )

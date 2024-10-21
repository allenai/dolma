# env_handler.py
import os
import re
from dotenv import load_dotenv
from utils import vprint

def load_env_variables():
    load_dotenv()

def expand_custom_env_vars(value):
    """Expand environment variables with ${oc.env:VAR_NAME} syntax."""
    pattern = r'\${oc\.env:([^}]+)}'
    
    def replace_env_var(match):
        env_var_name = match.group(1)
        env_var_value = os.getenv(env_var_name)
        if env_var_value is None:
            print(f"Warning: Environment variable {env_var_name} not found")
            return match.group(0)  # Return the original string if env var not found
        return env_var_value

    return re.sub(pattern, replace_env_var, value)

def expand_env_vars_in_config(config):
    """Expand environment variables in 'documents' and 'output' sections of the config."""
    if 'streams' in config:
        for stream in config['streams']:
            if 'documents' in stream:
                stream['documents'] = [expand_custom_env_vars(doc) for doc in stream['documents']]
                vprint(f"Expanded documents: {stream['documents']}")  # Debug print
            if 'output' in stream and 'path' in stream['output']:
                stream['output']['path'] = expand_custom_env_vars(stream['output']['path'])
                vprint(f"Expanded output path: {stream['output']['path']}")  # Debug print
    return config
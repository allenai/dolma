import os
import warnings

# warning raised by pkg_resources used in a lot of google packages
warnings.filterwarnings("ignore", message=r".*declare_namespace\(\'.*google.*", category=DeprecationWarning)
# base warning raised when warning above are raised
warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated.*", category=DeprecationWarning)

# ignore warning from packages that have not updated to use utcfromtimestamp
for module in ("botocore", "tqdm", "dateutil"):
    warnings.filterwarnings("ignore", module=module, message=r".*utcfromtimestamp\(\) is deprecated.*")
    warnings.filterwarnings("ignore", module=module, message=r".*utcnow\(\) is deprecated.*")

# ignore type annotation errors in this package
warnings.filterwarnings("ignore", message=r".*google\._upb\._message.*", category=DeprecationWarning)

# prefer ipdb over pdb in tests
os.environ.setdefault("PYTHONBREAKPOINT", "ipdb.set_trace")

"""
Imports all the generators in the generators in the generators folder (so new generators can be dynamically added)
"""

import os
import glob
import importlib


package_dir = os.path.dirname(__file__)

# Find all .py files in the package directory (excluding __init__.py)
modules = [
    os.path.basename(f)[:-3]
    for f in glob.glob(os.path.join(package_dir, "*.py"))
    if os.path.isfile(f) and not f.endswith("__init__.py") and f.endswith(".py")
]

__all__ = []

# Import each module and pull its public names into the package namespace
for module in modules:
    imported_module = importlib.import_module(f".{module}", package=__name__)
    for name in imported_module.__dict__:
        if not name.startswith("_"):
            globals()[name] = imported_module.__dict__[name]
            __all__.append(name)

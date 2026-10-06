"""Former import name of :mod:`blt_utils`, kept so older notebooks keep importing.

The utilities now live in ``src/blt_utils/`` as an installed (editable) package, so
``import blt_utils`` resolves from any working directory and editors can follow it
for tooltips and go-to-definition.

Mechanism: rather than re-exporting with ``from blt_utils import *``, this replaces
its own entry in ``sys.modules`` with the real module object. The difference
matters because several notebooks and ``validation/harmonization_utils.py``
reassign ``blt``'s module-level path constants (``FIGURE_DATA_DIR``,
``RAMS_OUTPUT_DIRS``, ...) to redirect it. A star re-export would give them a
separate namespace, so those reassignments would never reach the functions in
``blt_utils`` that read the constants -- ``save_figure_data`` would keep writing to
the original directory without any error. Aliasing the module object makes
``import blt_utils_limited`` and ``import blt_utils`` the very same module.
"""

import sys

import blt_utils

# Make this import name an alias for the real module object (see docstring).
sys.modules[__name__] = blt_utils

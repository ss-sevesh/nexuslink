"""Root conftest: ensures 'nexuslink' is importable when running pytest directly.

Because pyproject.toml lives inside the nexuslink/ package directory,
the parent directory must be on sys.path for `import nexuslink` to resolve.
`uv run pytest` handles this via the editable install; this file covers
plain `pytest` invocations.
"""

import sys
from pathlib import Path

# Parent of this file = Nexus-link/, which contains the nexuslink/ package dir
_project_parent = Path(__file__).parent.parent
if str(_project_parent) not in sys.path:
    sys.path.insert(0, str(_project_parent))

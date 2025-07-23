"""Import-linter contracts for this codebase.

At the moment we only enforce a single rule that prevents bringing back
legacy modules.  By placing contracts inside ``lint_contracts`` instead of
the project root we keep the repository tidy while still allowing
Import-Linter to discover them via the ``[tool.importlinter]`` configuration
in ``pyproject.toml``.
"""

from importlinter import Contract as ImportContract
from importlinter.domain.contract import ForbiddenContract


Contract = ImportContract(
    name="No legacy imports",
    session_options={"root_package": "jules"},
    contract=ForbiddenContract(
        for_layers=["jules"],
        forbidden_modules=["jules.legacy", "oldlib"],
    ),
)


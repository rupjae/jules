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

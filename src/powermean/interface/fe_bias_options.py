"""Normalize distinct BBP and published WZ coefficient requests."""

from typing import Literal

from powermean.errors import SpecificationError

BiasRequest = Literal["auto", "short_group", "cell", "wz"] | None


def resolve_bias_correction(value: object = False) -> BiasRequest:
    """Validate an existing request; WZ and undeclared methods are not BBP aliases."""
    if value is False:
        return None
    if value is True:
        return "auto"
    if isinstance(value, str):
        if value == "short_group":
            return "short_group"
        if value == "cell":
            return "cell"
        if value == "wz":
            return "wz"
    raise SpecificationError(
        "specification.invalid_value",
        {
            "argument": "bias_correction",
            "value": value,
            "valid": "False, True, 'short_group', 'cell', or 'wz'",
        },
    )

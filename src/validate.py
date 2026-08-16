"""Data contract for a single insurance claim record.

Uses only the standard library + pandas (no hard pydantic dependency required
for the lightweight dataframe validator), but also exposes a pydantic model
for use in the optional API layer.
"""
from __future__ import annotations

VALID_STATES = {"IN", "MH", "KA", "DL", "GJ", "TN", "UP", "RJ", "WB", "AP"}
VALID_CHANNELS = {"agent", "online", "branch", "broker", "direct"}

REQUIRED_COLUMNS = {"age", "vehicle_age", "state", "channel"}


def validate_dataframe(df, require_target: bool = False, target: str = "loss") -> list:
    """Validate a claims DataFrame against the data contract.

    Returns a list of human-readable error strings. Empty list means the
    DataFrame passed validation.
    """
    errors: list = []
    required = set(REQUIRED_COLUMNS)
    if require_target:
        required.add(target)

    missing = required - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
        return errors

    if (df["age"] < 18).any() or (df["age"] > 100).any():
        errors.append("Column 'age' must be between 18 and 100")
    if (df["vehicle_age"] < 0).any() or (df["vehicle_age"] > 40).any():
        errors.append("Column 'vehicle_age' must be between 0 and 40")
    bad_states = set(df["state"].astype(str).str.upper()) - VALID_STATES
    if bad_states:
        errors.append(f"Unknown state code(s): {sorted(bad_states)}")
    bad_channels = set(df["channel"].astype(str).str.lower()) - VALID_CHANNELS
    if bad_channels:
        errors.append(f"Unknown channel(s): {sorted(bad_channels)}")
    if require_target:
        if (df[target] <= 0).any():
            errors.append(f"Target column '{target}' must contain only positive values")

    return errors


try:
    from pydantic import BaseModel, Field, field_validator

    class ClaimRecord(BaseModel):
        """Pydantic schema for a single claim (used by optional API layer)."""

        age: int = Field(..., ge=18, le=100)
        vehicle_age: int = Field(..., ge=0, le=40)
        state: str
        channel: str

        @field_validator("state")
        @classmethod
        def _check_state(cls, v: str) -> str:
            if v.upper() not in VALID_STATES:
                raise ValueError(f"Unknown state code '{v}'")
            return v

        @field_validator("channel")
        @classmethod
        def _check_channel(cls, v: str) -> str:
            if v.lower() not in VALID_CHANNELS:
                raise ValueError(f"Unknown channel '{v}'")
            return v

except ImportError:  # pragma: no cover - pydantic optional at runtime
    ClaimRecord = None

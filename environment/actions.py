"""
Shared action-index mapping for all Phase 1 components.

Import this module instead of hard-coding integer constants so that
the Gym environment, the demo recorder, and any policy all agree on
the same action ↔ integer mapping.
"""

# Action indices
NOOP           = 0
PARRY          = 1
DODGE          = 2
JUMP           = 3
GRADIENT_PARRY = 4
ATTACK         = 5
JUMP_ATTACK    = 6

NUM_ACTIONS = 7

# Human-readable names (index → name)
ACTION_NAMES = {
    NOOP:           "NOOP",
    PARRY:          "PARRY",
    DODGE:          "DODGE",
    JUMP:           "JUMP",
    GRADIENT_PARRY: "GRADIENT_PARRY",
    ATTACK:         "ATTACK",
    JUMP_ATTACK:    "JUMP_ATTACK",
}

# Reverse mapping (name → index), useful for the demo recorder
ACTION_INDEX = {v: k for k, v in ACTION_NAMES.items()}

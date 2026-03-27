"""
Unit tests for environment/actions.py.
"""

import pytest
from environment.actions import (
    NOOP, PARRY, DODGE, JUMP, GRADIENT_PARRY, ATTACK, JUMP_ATTACK,
    NUM_ACTIONS, ACTION_NAMES, ACTION_INDEX,
)


class TestActionConstants:
    def test_all_constants_have_unique_values(self):
        values = [NOOP, PARRY, DODGE, JUMP, GRADIENT_PARRY, ATTACK, JUMP_ATTACK]

        assert len(set(values)) == len(values)

    def test_constants_are_zero_indexed_integers(self):
        assert NOOP == 0
        assert PARRY == 1
        assert DODGE == 2
        assert JUMP == 3
        assert GRADIENT_PARRY == 4
        assert ATTACK == 5
        assert JUMP_ATTACK == 6

    def test_num_actions_matches_constant_count(self):
        assert NUM_ACTIONS == 7


class TestActionNames:
    def test_action_names_covers_all_constants(self):
        assert set(ACTION_NAMES.keys()) == {
            NOOP, PARRY, DODGE, JUMP, GRADIENT_PARRY, ATTACK, JUMP_ATTACK
        }

    def test_action_names_values_are_strings(self):
        assert all(isinstance(v, str) for v in ACTION_NAMES.values())

    def test_action_names_specific_mappings(self):
        assert ACTION_NAMES[NOOP]           == "NOOP"
        assert ACTION_NAMES[PARRY]          == "PARRY"
        assert ACTION_NAMES[DODGE]          == "DODGE"
        assert ACTION_NAMES[JUMP]           == "JUMP"
        assert ACTION_NAMES[GRADIENT_PARRY] == "GRADIENT_PARRY"
        assert ACTION_NAMES[ATTACK]         == "ATTACK"
        assert ACTION_NAMES[JUMP_ATTACK]    == "JUMP_ATTACK"


class TestActionIndex:
    def test_action_index_is_inverse_of_action_names(self):
        for idx, name in ACTION_NAMES.items():
            assert ACTION_INDEX[name] == idx

    def test_action_index_covers_all_names(self):
        assert set(ACTION_INDEX.keys()) == set(ACTION_NAMES.values())

    def test_round_trip_index_to_name_and_back(self):
        for idx in range(NUM_ACTIONS):
            assert ACTION_INDEX[ACTION_NAMES[idx]] == idx

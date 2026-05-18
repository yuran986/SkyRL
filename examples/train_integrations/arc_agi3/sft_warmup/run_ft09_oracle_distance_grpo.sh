#!/bin/bash
set -x

# Oracle-distance GRPO warm-up for ARC-AGI-3 ft09.
#
# This wraps the standard ARC-AGI-3 GRPO launcher and enables ft09 oracle
# potential shaping in the environment. Existing GRPO knobs can still be
# overridden with environment variables or trailing CLI overrides.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${RUN_NAME:=arc_agi3_ft09_oracle_distance}"
: "${ARC_AGI3_ORACLE_DISTANCE_REWARD_ENABLED:=true}"
: "${ARC_AGI3_ORACLE_DISTANCE_REWARD:=0.05}"
: "${ARC_AGI3_ORACLE_DISTANCE_VALID_ACTION_REWARD:=0.0}"
: "${ARC_AGI3_ORACLE_ACTION_MATCH_REWARD:=0.05}"
: "${ARC_AGI3_ORACLE_ACTION_MATCH_RADIUS:=0}"
: "${ARC_AGI3_ORACLE_DISTANCE_UNRECOVERABLE_PENALTY:=-1.0}"
: "${ARC_AGI3_ORACLE_DISTANCE_MAX_NEXT_ACTIONS:=8}"

export RUN_NAME
export ARC_AGI3_ORACLE_DISTANCE_REWARD_ENABLED
export ARC_AGI3_ORACLE_DISTANCE_REWARD
export ARC_AGI3_ORACLE_DISTANCE_VALID_ACTION_REWARD
export ARC_AGI3_ORACLE_ACTION_MATCH_REWARD
export ARC_AGI3_ORACLE_ACTION_MATCH_RADIUS
export ARC_AGI3_ORACLE_DISTANCE_UNRECOVERABLE_PENALTY
export ARC_AGI3_ORACLE_DISTANCE_MAX_NEXT_ACTIONS

echo "ARC-AGI-3 oracle-distance reward enabled: $ARC_AGI3_ORACLE_DISTANCE_REWARD_ENABLED"
echo "ARC-AGI-3 oracle-distance reward coef: $ARC_AGI3_ORACLE_DISTANCE_REWARD"
echo "ARC-AGI-3 oracle action match reward: $ARC_AGI3_ORACLE_ACTION_MATCH_REWARD"
echo "ARC-AGI-3 oracle action match radius: $ARC_AGI3_ORACLE_ACTION_MATCH_RADIUS"
echo "ARC-AGI-3 oracle unrecoverable penalty: $ARC_AGI3_ORACLE_DISTANCE_UNRECOVERABLE_PENALTY"

bash "$SCRIPT_DIR/../run_arc_agi3_grpo.sh" "$@"

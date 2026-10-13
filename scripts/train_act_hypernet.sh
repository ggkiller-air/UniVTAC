#!/usr/bin/env bash
set -euo pipefail

export ACT_VARIANT=hypernet
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_act.sh" "$@"

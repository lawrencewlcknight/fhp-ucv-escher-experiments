#!/usr/bin/env bash
set -Eeuo pipefail
export FEATURE_NUMBER=33
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_strategic_features.sh" "$@"

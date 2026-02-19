#!/usr/bin/env bash
set -euo pipefail

bash scripts/bootstrap.sh
bash scripts/smoke.sh

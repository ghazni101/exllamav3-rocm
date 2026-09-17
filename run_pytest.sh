#!/bin/bash
cd /opt/exllamav3
/opt/venv/bin/pip install -q pytest pytest-timeout
exec /opt/venv/bin/python3 -m pytest tests/test_qgemm.py tests/test_quant_fn.py tests/test_cache_rotate.py tests/test_gated_delta_rule.py -q --timeout=300 "$@"

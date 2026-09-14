#!/usr/bin/env python3
"""Aggregate only an explicit versioned run manifest, paired by seed.

Use: python3 scripts/e7_aggregate_tost.py --stage formal --manifest PATH
Historical glob/latest selection and unpaired Welch inference are intentionally disabled.
"""
import sys
from e7_restart_checks import main

if __name__ == '__main__':
    sys.argv.insert(1, 'aggregate')
    main()

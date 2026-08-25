#!/usr/bin/env python
"""Entry point: python linkage_dedup.py <input.csv> [more.csv ...] [options]

See README.md for usage. Run with -h for the full option list.
"""

import sys

from record_linkage.cli import main

if __name__ == "__main__":
    sys.exit(main())

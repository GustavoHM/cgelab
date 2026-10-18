"""``python -m cgelab`` runs the command line entry point."""

import sys

from .cli import main

sys.exit(main())

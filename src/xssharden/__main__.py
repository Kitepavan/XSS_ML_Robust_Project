"""Enable running xssharden as a module: python -m xssharden."""
import sys
from xssharden.cli import main

sys.exit(main())
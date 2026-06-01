"""``python -m mesofield`` — entry point for the Mesofield CLI.

The CLI itself lives in :mod:`mesofield.cli`; this module just exposes the
root group so both ``python -m mesofield`` and the ``mesofield`` console
script resolve to the same place.
"""

from mesofield.cli import cli

if __name__ == "__main__":
    cli()

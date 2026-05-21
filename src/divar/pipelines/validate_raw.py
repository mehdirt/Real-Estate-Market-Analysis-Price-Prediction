"""DVC stage: validate raw Divar CSV."""

from __future__ import annotations

import argparse
import logging
import sys

from divar.data.load import load_divar
from divar.validation import validate_raw_divar
from divar.validation.run import DataValidationError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate raw Divar CSV schema.")
    parser.parse_args(argv)

    try:
        df = load_divar()
        validate_raw_divar(df)
    except (FileNotFoundError, DataValidationError) as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

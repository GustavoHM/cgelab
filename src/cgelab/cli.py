"""Command line entry point: run the diagnostics on a SAM."""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="cgelab", description=__doc__)
    parser.add_argument("sam", nargs="?", help="path to a long-form SAM (.parquet or .csv); "
                                               "omitted, the bundled toy economy is used")
    parser.add_argument("--sigma-armington", type=float, default=2.0)
    parser.add_argument("--sigma-va", type=float, default=0.8)
    args = parser.parse_args(argv)

    import cgelab as cge

    if args.sam:
        sam = cge.SAM.read(args.sam)
        spec = cge.ModelSpec(sigma_armington=args.sigma_armington, sigma_va=args.sigma_va)
        econ = cge.calibrate(sam, spec)
    else:
        from cgelab.examples import toy
        sam, econ = toy.build()
        print("(no SAM given -- running the bundled toy economy)\n")

    print(sam)
    print(cge.report(econ))
    return 0


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())

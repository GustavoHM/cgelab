"""Command line entry point: run the diagnostics on a SAM."""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="cgekit", description=__doc__)
    parser.add_argument("sam", nargs="?", help="path to a long-form SAM (.parquet or .csv); "
                                               "omitted, the bundled toy economy is used")
    parser.add_argument("--sigma-armington", type=float, default=2.0)
    parser.add_argument("--sigma-va", type=float, default=0.8)
    args = parser.parse_args(argv)

    import cgekit as ck

    if args.sam:
        sam = ck.SAM.read(args.sam)
        spec = ck.ModelSpec(sigma_armington=args.sigma_armington, sigma_va=args.sigma_va)
        econ = ck.calibrate(sam, spec)
    else:
        from cgekit.examples import toy
        sam, econ = toy.build()
        print("(no SAM given -- running the bundled toy economy)\n")

    print(sam)
    print(ck.report(econ))
    return 0


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())

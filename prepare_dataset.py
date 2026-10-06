"""
prepare_dataset.py

Run this script ONCE before starting Assignment #3 to verify that your working
directory contains all the files you need to develop and test your agent
against the sample query.

Usage:
    python prepare_dataset.py

Expected output:
    All 6 required gist files found (out of 8 — README.md and this script are
    not checked). You are ready to start working on hw3.py.
"""

import os
import sys

REQUIRED_FILES = [
    "input.json",
    "receipt_analysis.txt",
    "receipt_analysis_validate.py",
    "receipt.png",
    "orders.db",
    "customers.db",
]


def main():
    missing = [f for f in REQUIRED_FILES if not os.path.isfile(f)]
    if missing:
        print("ERROR: The following required files are missing from the current directory:")
        for f in missing:
            print(f"  - {f}")
        print()
        print("Make sure you downloaded ALL files from the gist and that you are running")
        print("this script from the same directory where they live.")
        sys.exit(1)

    print("All 6 required gist files found (out of 8 — README.md and this script are not checked). You are ready to start working on hw3.py.")
    print()
    print("Next steps:")
    print("  1. Implement your agent in hw3.py (raw OpenAI tool-calling loop).")
    print("  2. Run it with:  python hw3.py")
    print("  3. Verify it produces: receipt_analysis.log and receipt_analysis_result.json")


if __name__ == "__main__":
    main()

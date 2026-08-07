#!/usr/bin/env python3
"""
OpenMix — entry point. Delegates to cli module.

Usage:
    python openmix.py /path/to/music
    python openmix.py /path/to/music -o mix.wav
"""


def main():
    from cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()

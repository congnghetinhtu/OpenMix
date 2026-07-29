#!/usr/bin/env python3
"""
OpenMix — backward-compatible entry point.
Delegates to modular implementation.


Usage:
    python openmix.py /path/to/music
    python openmix.py /path/to/music -o mix.wav -c 10.0

For the new modular API:
    from cli import run
    run("/path/to/music")
"""

import sys

# Re-export the OpenMixer class for backward compatibility
# New code should use: from cli import run


class OpenMixer:
    """Legacy wrapper — delegates to cli.run()."""

    def __init__(self, input_folder, output_file="openmix_output.wav",
                 crossfade_duration=15.0, sample_rate=44100):
        self.input_folder = input_folder
        self.output_file = output_file
        self.crossfade_duration = crossfade_duration
        self.sample_rate = sample_rate

    def create_mix(self):
        from cli import run
        return run(
            self.input_folder,
            self.output_file,
            self.crossfade_duration,
            self.sample_rate,
        )


def main():
    from cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    sys.exit(main())

"""Compatibility entry point for the Pac-Man demo.

The implementation now lives in ``envs.pacman.run``. Keeping this thin wrapper
lets existing commands such as ``py -3 run.py`` continue to work.
"""

from envs.pacman.run import main


if __name__ == "__main__":
    main()

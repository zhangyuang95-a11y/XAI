"""Compatibility entry point for Pac-Man explanation validation.

The implementation now lives in ``envs.pacman.validate_explanations``. Keeping
this wrapper lets existing commands continue to work during the package split.
"""

from envs.pacman.validate_explanations import main


if __name__ == "__main__":
    main()

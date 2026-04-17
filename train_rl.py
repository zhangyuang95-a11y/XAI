"""Compatibility entry point for Pac-Man RL training.

The implementation now lives in ``envs.pacman.train_rl``. Keeping this wrapper
lets existing commands such as ``py -3 train_rl.py`` continue to work.
"""

from envs.pacman.train_rl import main


if __name__ == "__main__":
    main()

"""Compatibility entry point for the current kitchen calibration.

Prior v4 evidence is archived; this entry no longer runs the old protocol.
"""
import json
from .calibrate_v5 import build, rollout

if __name__ == '__main__':
    print(json.dumps(build(),indent=2))

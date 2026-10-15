"""Compatibility entry point for the current two-recipe Kitchen validation.

The former soup-controller proxies no longer describe this engine. Existing
validation_results.json / validation_replays.json are historical v3.1 artifacts
and are deliberately not overwritten. Current results are versioned inside
configs/study_v3_kitchen.json by calibrate_v5.
"""
import json
from .calibrate_v5 import build


if __name__ == '__main__':
    print('Kitchen v5 feasibility proxy; historical soup/v4 results are not current evidence.')
    print(json.dumps(build(), indent=2))

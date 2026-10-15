"""The archived pure-NN reader keeps its original, hash-pinned artifact."""

from hashlib import sha256
import json
from pathlib import Path

from domains.warehouse import neural_turnbased as warehouse
from env.warehouse.pure_nn import FrozenPureNN


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/study_v3_warehouse_neural.json"
V6_CATALOG = ROOT / "releases/policylens-three-domain-20261016.v6.0/catalog.json"


def test_archived_warehouse_actor_remains_loadable_without_promotion() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    path = ROOT / config["actor_path"]
    assert config["model_version"] == "warehouse-pure-nn.v1"
    assert config["actor_sha256"] == sha256(path.read_bytes()).hexdigest()
    actor = FrozenPureNN.load(path, expected_sha256=config["actor_sha256"])
    assert actor.artifact_sha256 == config["actor_sha256"]
    assert warehouse._actor(warehouse.LEGACY_POLICY_ID).artifact_sha256 == actor.artifact_sha256

    catalog_text = V6_CATALOG.read_text(encoding="utf-8")
    catalog = json.loads(catalog_text)
    if catalog["warehouse"]["status"] == "pending":
        assert "warehouse" not in catalog["policies"]
    assert config["actor_sha256"] not in catalog_text

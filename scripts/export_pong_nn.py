#!/usr/bin/env python3
"""Export a frozen Pong Actor to a browser-readable float32 package."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from domains.pong.adapters.core_adapter import feature_mapping_from_vector, feature_vector
from domains.pong.training.browser_compat import verify_browser_export
from domains.pong.training.runner import PongPPOTrainer, _device, _signature, load_config
from domains.pong.policies.hybrid import VERSION as HYBRID_VERSION
from domains.pong.policies.coordinated import VERSION as COORDINATED_VERSION

def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--run",required=True); parser.add_argument("--checkpoint",required=True); parser.add_argument("--output",required=True); parser.add_argument("--device",default="cpu")
    parser.add_argument("--controller-mode", choices=("pure_nn", "hybrid", "coordinated", "rule_only"), default="pure_nn")
    parser.add_argument("--allow-candidate", action="store_true")
    parser.add_argument("--controller-config", help="Versioned controller settings for coordinated mode")
    args=parser.parse_args(); run=Path(args.run); config=load_config(run/"config.yaml"); trainer=PongPPOTrainer(config,device=_device(args.device)); payload=torch.load(run/args.checkpoint,map_location="cpu",weights_only=False); trainer.load_state_dict(payload)
    if config.get("curriculum", {}).get("v22") and not args.allow_candidate:
        raise ValueError("v2.2 candidate export requires --allow-candidate; it is not a qualified release")
    target=Path(args.output)
    if config.get("curriculum", {}).get("v22") and target.exists() and any(target.iterdir()):
        raise FileExistsError("v2.2 export requires a new empty output directory")
    target.mkdir(parents=True,exist_ok=True); state=trainer.model.actor.state_dict()
    arrays={key:value.detach().cpu().numpy().astype(np.float32) for key,value in state.items()}
    actor_digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        actor_digest.update(name.encode()); actor_digest.update(value.tobytes())
    digest = actor_digest.hexdigest()
    coordinated_settings = {}
    if args.controller_mode == "coordinated":
        if not args.controller_config:
            raise ValueError("coordinated export requires --controller-config")
        manifest = json.loads(Path(args.controller_config).read_text(encoding="utf-8"))
        if manifest.get("version") != COORDINATED_VERSION or manifest.get("actor_sha256") != digest:
            raise ValueError("coordinated controller version or frozen Actor hash mismatch")
        if manifest.get("source_checkpoint") != args.checkpoint:
            raise ValueError("coordinated source checkpoint mismatch")
        coordinated_settings = dict(manifest["controller"])
    model={"format":"pong-browser-float32.v2","model_sha256":digest,"actions":["left","right","stay"],"signature":_signature(trainer.env_config,trainer.names),"layers":[{"weight":arrays["0.weight"].tolist(),"bias":arrays["0.bias"].tolist(),"activation":"tanh"},{"weight":arrays["2.weight"].tolist(),"bias":arrays["2.bias"].tolist(),"activation":"tanh"},{"weight":arrays["4.weight"].tolist(),"bias":arrays["4.bias"].tolist(),"activation":"linear"}]}
    # The browser receives this exact float32 representation; verify its
    # features, probabilities and deterministic action before writing it.
    row = feature_vector(trainer.envs[0], "ai")
    features = feature_mapping_from_vector(trainer.names, row)
    with torch.no_grad():
        python_probabilities = torch.softmax(
            trainer.model.actor_logits(torch.as_tensor(row[None, :], dtype=torch.float32, device=trainer.device)), -1,
        )[0].cpu().numpy()
    compatibility = verify_browser_export(model, features, python_probabilities)
    (target/"nn_model.json").write_text(json.dumps(model,separators=(",",":")),encoding="utf-8")
    stored_program=payload.get("rcpd",{}).get("program")
    program=(stored_program if stored_program and stored_program.get("metadata", {}).get("actor_parameters_sha256") == digest
             else None) if config.get("curriculum", {}).get("v22") else stored_program
    program_status = ("current_actor" if program else "stale_or_missing") if config.get("curriculum", {}).get("v22") else ("available" if program else "missing")
    if program: (target/"program.json").write_text(json.dumps(program,ensure_ascii=False,indent=2),encoding="utf-8")
    program_hash = hashlib.sha256(json.dumps(program,sort_keys=True,separators=(",",":")).encode()).hexdigest() if program else None
    controller = {"controller_mode": args.controller_mode, "rule_version": COORDINATED_VERSION if args.controller_mode == "coordinated" else HYBRID_VERSION if args.controller_mode == "hybrid" else None,
                  "urgent_window_seconds": float(config.get("hybrid", {}).get("urgent_window_seconds", .6)),
                  "tradeoff_window_seconds": float(config.get("hybrid", {}).get("tradeoff_window_seconds", 1.2)),
                  "minimum_recovery_margin_seconds": float(config.get("hybrid", {}).get("minimum_recovery_margin_seconds", .1)),
                  "model_sha256": digest, "program_sha256": program_hash, "candidate_only": True,
                  "source_checkpoint": args.checkpoint, "source_joint_steps": trainer.joint_steps,
                  **coordinated_settings}
    (target/"controller_config.json").write_text(json.dumps(controller,ensure_ascii=False,indent=2),encoding="utf-8")
    (target/"export_report.json").write_text(json.dumps({"model_sha256":digest,"program_available":bool(program),"program_status":program_status,"program_sha256":program_hash,"controller":controller,"browser_compatibility":compatibility,"qualification":{"performed":False,"status":"candidate_only"},"note":"Float32 browser export; candidate status is preserved. JavaScript parity is verified separately."},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"event":"export_complete","output":str(target),"model_sha256":digest,"program_available":bool(program)},ensure_ascii=False));return 0
if __name__=="__main__":raise SystemExit(main())

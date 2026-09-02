#!/usr/bin/env python
"""
Publish a trained checkpoint to the Hub, so the worker can pick it up.

Why this exists
---------------
The notebook publishes from inside train(), at the end of a run. There is no way
to re-publish weights that already exist: re-running the cell retrains from
scratch, which takes twenty minutes and -- because GPU work is not bit-for-bit
deterministic -- produces a DIFFERENT model from the one that was evaluated. You
would be shipping something you had not measured.

This publishes an existing checkpoint, so the artefact that was scored by
evaluate.py is the artefact that goes live.

It also writes the preprocessing into the published config. timm carries the
base model's pretrained_cfg forward, which for swin_s3_base_224.ms_in1k says
ImageNet mean/std -- not what this model was trained with. Anything deriving a
transform from the hub config would silently mis-normalise every image.

    python publish.py --weights ./swin_s3_base_224-xblockm-timm --dry-run
    python publish.py --weights ./swin_s3_base_224-xblockm-timm --yes
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "processor"))

from manifest import CLASSES  # noqa: E402

DEFAULT_REPO = "swin_s3_base_224-xblockm-timm"
ARCH = "swin_s3_base_224"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="checkpoint dir or model.safetensors")
    p.add_argument("--repo", default=DEFAULT_REPO,
                   help=f"hub repo name under your account (default: {DEFAULT_REPO})")
    p.add_argument("--eval", help="an evaluate.py --json result, recorded in the model card")
    p.add_argument("--dry-run", action="store_true", help="show what would be pushed")
    p.add_argument("--yes", action="store_true",
                   help="actually publish -- this replaces what the live worker loads")
    args = p.parse_args()

    # Check write access first. Publishing fails at the very end otherwise, with
    # a 403 from repos/create that reads like the repo is missing rather than
    # like the token is read-scoped.
    if args.yes:
        from huggingface_hub import HfApi
        try:
            me = HfApi().whoami()
        except Exception as e:
            sys.exit(f"not logged in to Hugging Face: {e}\n"
                     "  huggingface-cli login")
        at = (me.get("auth", {}) or {}).get("accessToken", {}) or {}
        role = at.get("role")
        # A fine-grained token reports role='fineGrained' and carries its real
        # permissions nested, so the role alone says nothing about write access.
        perms = set()
        fg = at.get("fineGrained") or {}
        for scope in list(fg.get("scoped") or []) + [{"permissions": fg.get("global") or []}]:
            perms.update(scope.get("permissions") or [])
        can_write = role == "write" or any("write" in p for p in perms)
        if not can_write:
            sys.exit(
                f"logged in as {me.get('name')}, token role '{role}' -- no write "
                f"permission found.\n"
                "  Create a token at https://huggingface.co/settings/tokens with write\n"
                "  access, then: huggingface-cli login")
        print(f"authenticated as {me.get('name')} (role: {role}"
              + (f", perms: {', '.join(sorted(perms))}" if perms else "") + ")")

    import torch  # noqa: F401
    from safetensors.torch import load_file
    from timm import create_model

    path = Path(args.weights)
    f = path / "model.safetensors" if path.is_dir() else path
    if not f.is_file():
        sys.exit(f"no model.safetensors at {f}")

    state = load_file(str(f))
    head = state.get("head.fc.weight")
    if head is None:
        sys.exit("checkpoint has no head.fc.weight -- is this a timm swin checkpoint?")
    if head.shape[0] != len(CLASSES):
        sys.exit(f"checkpoint has {head.shape[0]} classes, the manifest has "
                 f"{len(CLASSES)}. Publishing this would put a model with the wrong "
                 f"class list in front of the worker.")

    model = create_model(ARCH, num_classes=len(CLASSES), pretrained=False)
    model.load_state_dict(state)
    model.eval()

    # What the model was ACTUALLY trained with. See processor/preprocessing.py.
    # Note the resize is an aspect-ratio squash, which timm's data config cannot
    # express -- crop_pct 1.0 is the nearest approximation, and serving code
    # should mirror preprocessing.py rather than derive a transform from this.
    model.pretrained_cfg.update({
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "input_size": (3, 224, 224),
        "interpolation": "bilinear",
        "crop_pct": 1.0,
        "num_classes": len(CLASSES),
        "tag": "xblockm",
    })

    print(f"checkpoint   {f}")
    print(f"architecture {ARCH}   classes {len(CLASSES)}")
    print(f"classes      {', '.join(CLASSES)}")
    print(f"normalize    mean/std 0.5  (NOT the ImageNet values timm would infer)")
    print(f"target repo  howdyaendra/{args.repo}")

    metrics = None
    if args.eval:
        metrics = json.loads(Path(args.eval).read_text())
        print(f"\nevaluated    macro AP {metrics['macro_ap']:.4f}   "
              f"micro AUROC {metrics['micro_roc_auc']:.4f}   "
              f"FP {metrics['fp_rate_on_negatives']*100:.2f}%  "
              f"on {metrics['images']:,} {metrics['split']} images")

    print(f"\nThis repo is what processor/worker.py downloads. Publishing replaces")
    print(f"the model the live labeller uses on its next restart.")

    if not args.yes:
        print("\ndry run -- re-run with --yes to publish")
        return

    card = None
    if metrics:
        rows = "\n".join(
            f"| {k} | {v['n']} | {v['ap']:.3f} | {v['precision']:.3f} | {v['recall']:.3f} |"
            for k, v in metrics["per_class"].items() if v["ap"] is not None)
        card = {"description":
                f"Multi-label screenshot classifier. Sigmoid head; pass logits "
                f"through sigmoid, not softmax.\n\n"
                f"Preprocessing: resize to 224x224 (aspect-ratio squash), "
                f"normalize mean/std 0.5. The inherited pretrained_cfg from "
                f"ms_in1k does NOT describe this model.\n\n"
                f"Evaluated on {metrics['images']:,} held-out images "
                f"({datetime.now(timezone.utc).date()}): macro AP "
                f"{metrics['macro_ap']:.3f}, false positives on negatives "
                f"{metrics['fp_rate_on_negatives']*100:.2f}%.\n\n"
                f"| class | n | AP | precision | recall |\n|---|---|---|---|---|\n{rows}\n"}

    from huggingface_hub import HfApi
    from timm.models._hub import save_for_hf

    api = HfApi()
    repo_id = f"{api.whoami()['name']}/{args.repo}"

    # push_to_hf_hub calls create_repo(exist_ok=True) unconditionally, which a
    # fine-grained token without repo-creation rights rejects with a 403 -- even
    # when the repo already exists and the token can write to it. Only create
    # when it is genuinely missing.
    if api.repo_exists(repo_id):
        print(f"repo exists, uploading without create")
    else:
        print(f"repo does not exist, creating it")
        api.create_repo(repo_id, exist_ok=True)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        save_for_hf(model, tmp, model_config=dict(label_names=list(CLASSES)),
                    safe_serialization=True)
        if card:
            (Path(tmp) / "README.md").write_text(card["description"])
        api.upload_folder(repo_id=repo_id, folder_path=tmp,
                          commit_message=f"xblock model, macro AP "
                                         f"{metrics['macro_ap']:.3f}" if metrics
                                         else "xblock model")
    print(f"\npublished to {repo_id}")
    print("Restart the worker to pick it up:")
    print("  supervisorctl -c /home/aendra/xblock-docker/supervisord.conf "
          "restart xblock-worker")


if __name__ == "__main__":
    main()

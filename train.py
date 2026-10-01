"""Entry point for AffectDrop cold-start SFT and joint RL."""
import argparse
import glob
import os
from pathlib import Path
import runpy


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for name, steps in (("sft", 2400), ("rl", 1000)):
        stage = stages.add_parser(name)
        stage.add_argument("--model", type=Path, required=True, help="Local base checkpoint and processor directory")
        stage.add_argument("--train", type=Path, required=True, help="Training JSONL")
        stage.add_argument("--output", type=Path, required=True, help="New or empty output directory")
        stage.add_argument("--seed", type=int, default=18)
        stage.add_argument("--steps", type=positive, default=steps)
        stage.add_argument("--save-every", type=positive, default=200 if name == "sft" else 20)
        if name == "sft":
            stage.add_argument("--traces", required=True, help="Quoted glob for teacher JSONL files")
            stage.add_argument("--lr", type=float, default=1e-4)
        else:
            stage.add_argument("--sft-adapter", type=Path, required=True)
            stage.add_argument("--dev-file", type=Path, required=True, help="Frozen 200-row TRAIN-internal development JSONL")
            stage.add_argument("--resume", type=Path, help="Full RL resume checkpoint; use a new output directory")
            stage.add_argument("--actor-lr", type=float, default=2e-5)
            stage.add_argument("--selector-lr", type=float, default=3e-4)
            stage.add_argument("--max-tokens", type=positive, default=384)
    args = parser.parse_args()
    if args.seed < 0 or args.seed >= 2**32:
        parser.error("--seed must be in [0, 2**32)")
    import math
    for key in ("lr", "actor_lr", "selector_lr"):
        value = getattr(args, key, None)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")
    args.model = args.model.expanduser().resolve()
    args.train = args.train.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.model.is_dir():
        parser.error(f"Missing model directory: {args.model}")
    if not args.train.is_file():
        parser.error(f"Missing training JSONL: {args.train}")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("--output must be new or empty")
    for key in ("sft_adapter", "dev_file", "resume"):
        value = getattr(args, key, None)
        if value is not None:
            value = value.expanduser().resolve()
            if not value.is_file():
                parser.error(f"Missing --{key.replace('_', '-')}: {value}")
            setattr(args, key, value)
    if args.stage == "sft" and not glob.glob(os.path.expanduser(args.traces)):
        parser.error("--traces matched no files")
    if args.stage == "rl" and args.max_tokens < 8:
        parser.error("--max-tokens must be at least 8")

    # Keep the CLI configuration independent of stale experiment variables.
    for key in list(os.environ):
        if key.startswith(("AFFECT_SFT_", "AFFECT_JOINT_")):
            del os.environ[key]
    prefix = "AFFECT_SFT_" if args.stage == "sft" else "AFFECT_JOINT_"
    settings = {
        "BASE_MODEL": args.model, "OUTPUT": args.output,
        "SEED": args.seed, "STEPS": args.steps, "SAVE_EVERY": args.save_every,
    }
    if args.stage == "sft":
        settings.update(TRACE_GLOB=os.path.expanduser(args.traces), LR=args.lr,
                        EXPECTED_TRACE_FILES=len(glob.glob(os.path.expanduser(args.traces))),
                        GRAD_ACCUM=1, LORA_RANK=16, CLASS_BALANCE_ALPHA=0.5,
                        SCHEDULE="cosine", DROP_RATIOS="0,0.1,0.2,0.3,0.4")
    else:
        settings.update(SFT_ADAPTER=args.sft_adapter, DEV_FILE=args.dev_file,
                        GROUP_SIZE=2, MAX_TOKENS=args.max_tokens, REFLECT_TOKENS=32,
                        ACTOR_LR=args.actor_lr, SELECTOR_LR=args.selector_lr,
                        ALIGN_WEIGHT=0.15, DROP_WEIGHT=0.05, KL_COEF=0.03,
                        FULL_SAVE_EVERY=200)
        if args.resume is not None:
            settings["RESUME_CHECKPOINT"] = args.resume
    os.environ["AFFECT_TRAIN_FILE"] = str(args.train)
    os.environ.update({prefix + key: str(value) for key, value in settings.items()})
    runpy.run_path(str(Path(__file__).resolve().parent / f"_{args.stage}.py"), run_name="__main__")


if __name__ == "__main__":
    main()

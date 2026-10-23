"""Command-line entry point for the complete TinyGPT experiment."""

import argparse
from pathlib import Path

from checks import check_data, check_model
from config import Config, load_config
from pipeline import build_artifacts, run_pipeline
from runtime import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and report the TinyGPT assignment end to end")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Prepare → check → overfit → train → evaluate → sample → plot → report")
    run.add_argument("--config", type=Path, default=Path("config.json"), help="Settings file (default: config.json)")
    run.add_argument("--run-dir", type=Path, help="Fresh output directory, or a matching completed experiment to reuse")
    run.add_argument("--data-dir", type=Path)
    run.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    run.add_argument("--gate", type=Path, help="Reuse matching fixed-batch overfit evidence")
    run.add_argument("--notes", type=Path, help="Qualitative observations to include in the report")
    run.add_argument("--revision", help="Dataset revision for preparing new data")
    report = subparsers.add_parser("report", help="Verify and build a results packet from an existing completed run")
    report.add_argument("--run-dir", type=Path, default=Config().output_dir)
    report.add_argument("--data-dir", type=Path, help="Override data location after moving a run")
    report.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    report.add_argument("--notes", type=Path)
    check = subparsers.add_parser("check", help="Run data and model checks without training")
    check.add_argument("--config", type=Path, default=Path("config.json"))
    check.add_argument("--run-dir", type=Path)
    check.add_argument("--data-dir", type=Path)
    check.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    try:
        device = select_device(args.device)
        if args.command == "report":
            output = build_artifacts(args.run_dir, device, args.data_dir, args.notes)
        else:
            config = load_config(args.config, data_dir=args.data_dir, output_dir=args.run_dir)
            if args.command == "check":
                check_data(config, config.output_dir / "checks/data_contract.json")
                check_model(config, device, config.output_dir / "checks/model_check.json")
                print(f"Checks passed: {config.output_dir / 'checks'}")
                return
            output = run_pipeline(config, device, args.gate, args.notes, args.revision)
        print(f"Results packet: {output / 'report.pdf'}", flush=True)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

"""Ordered, recorded stages for fresh experiments and existing-run reporting."""

from contextlib import redirect_stdout
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

import torch

from checkpoint import write_json
from checks import check_data, check_model
from config import Config
from evaluation import verify_run
from generate import generate_comparison
from overfit import run_overfit
from plot_loss import plot_losses
from prepare_data import prepare, sha256_file
from report import build_report
from train import run_training, verify_overfit_gate


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class StageRecorder:
    def __init__(self, path: Path, command: str):
        self.path = path
        self.state = {"status": "running", "command": command,
                      "started_utc": datetime.now(timezone.utc).isoformat(), "stages": []}
        write_json(path, self.state)

    def run(self, name, operation):
        entry = {"name": name, "status": "running"}
        self.state["stages"].append(entry)
        write_json(self.path, self.state)
        print(f"[{name}] starting", flush=True)
        start = time.perf_counter()
        try:
            result = operation()
        except BaseException as exc:
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            self.state["status"] = "failed"
            raise
        else:
            entry["status"] = "complete"
            print(f"[{name}] complete", flush=True)
            return result
        finally:
            entry["seconds"] = time.perf_counter() - start
            write_json(self.path, self.state)

    def finish(self):
        self.state.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(self.path, self.state)


def assert_reusable_run(config: Config) -> bool:
    path = config.output_dir / "results.json"
    if not path.exists():
        if any(config.output_dir.glob("checkpoints/*.pt")):
            raise ValueError("Checkpoint files exist without completed results; choose a fresh run directory")
        return False
    results = json.loads(path.read_text())
    if results["status"] != "complete":
        raise ValueError("Existing training is incomplete. Exact training resume is unsupported; choose a fresh run directory")
    stored = Config.from_dict(results["config"]).to_dict()
    differences = [key for key, value in config.to_dict().items()
                   if key not in ("data_dir", "output_dir") and value != stored[key]]
    if differences:
        raise ValueError(f"Existing experiment settings differ: {', '.join(differences)}. Use report for that run or choose a fresh directory")
    return True


def build_artifacts(run_dir: Path, device: torch.device, data_dir: Path | None = None,
                    notes: Path | None = None, recorder: StageRecorder | None = None) -> Path:
    output = run_dir / "report"
    output.mkdir(parents=True, exist_ok=True)
    own_recorder = recorder is None
    recorder = recorder or StageRecorder(output / "pipeline.json", "report")
    recorder.run("verify_checkpoints_and_evaluate", lambda: verify_run(run_dir, device, output / "verification.json", data_dir))
    recorder.run("generate_nine_continuations", lambda: generate_comparison(run_dir, device, output))
    recorder.run("plot_losses", lambda: plot_losses(run_dir, output))
    recorder.run("assemble_report", lambda: build_report(run_dir, output, notes))
    # A packet is ready only after every artifact was successfully generated.
    artifacts = ["report.md", "report.pdf", "requirements.json", "verification.json", "samples.json",
                 "samples.md", "loss_curve.png", "loss_curve.svg", "loss_points.csv", "figure_manifest.json"]
    write_json(output / "manifest.json", {
        "source_results": str(run_dir / "results.json"),
        "source_results_sha256": sha256_file(run_dir / "results.json"),
        "artifacts": {name: sha256_file(output / name) for name in artifacts},
    })
    if own_recorder:
        recorder.finish()
    return output


def run_pipeline(config: Config, device: torch.device, gate_path: Path | None = None,
                 notes: Path | None = None, revision: str | None = None) -> Path:
    reusable = assert_reusable_run(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    recorder = StageRecorder(config.output_dir / "metadata/pipeline.json", "run")
    with (config.output_dir / "metadata/pipeline.log").open("a") as log, redirect_stdout(_Tee(sys.stdout, log)):
        if not (config.data_dir / "manifest.json").exists():
            recorder.run("prepare_data", lambda: prepare(config, revision))
        else:
            print("Using existing prepared data; hashes and recipe settings will be checked.")
        checks_dir = config.output_dir / "checks"
        recorder.run("check_data", lambda: check_data(config, checks_dir / "data_contract.json"))
        recorder.run("check_model", lambda: check_model(config, device, checks_dir / "model_check.json"))
        manifest = json.loads((config.data_dir / "manifest.json").read_text())
        gate = config.output_dir / "checks/overfit.json"

        def ensure_gate():
            if gate_path is not None:
                verify_overfit_gate(gate_path, config, manifest)
                if gate_path.resolve() != gate.resolve():
                    if gate.exists() and gate.read_bytes() != gate_path.read_bytes():
                        raise ValueError("A different overfit gate already exists in this run")
                    shutil.copyfile(gate_path, gate)
            elif not gate.exists():
                if reusable:
                    raise ValueError("Completed run has no overfit evidence; provide its original --gate")
                run_overfit(config, device)
            verify_overfit_gate(gate, config, manifest)

        recorder.run("verify_or_run_overfit_gate", ensure_gate)
        if reusable:
            print("Reusing completed training; saved metrics and weights will not be changed.")
        else:
            write_json(config.output_dir / "metadata/config.json", config.to_dict())
            source_dir = Path(__file__).parent
            write_json(config.output_dir / "metadata/code_manifest.json", {
                "source_sha256": {path.name: sha256_file(path) for path in sorted(source_dir.glob("*.py"))},
                "torch_version": torch.__version__,
            })
            recorder.run("train", lambda: run_training(config, device, gate))
        output = build_artifacts(config.output_dir, device, config.data_dir, notes, recorder)
        recorder.finish()
    return output

"""Offline checks for evaluation sampling and model selection."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    calls = []
    eos_token_id = 0

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        return messages[0]["content"]


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeLLM:
    engines = []
    calls = []

    def __init__(self, **kwargs):
        self.engines.append(kwargs)

    def generate(self, prompts, sampling, **kwargs):
        self.calls.append((sampling, kwargs))
        answer = "A" if sampling.n == 4 else "42"
        return [types.SimpleNamespace(outputs=[
            types.SimpleNamespace(text=f"\\boxed{{{answer}}}", token_ids=[1, 2], finish_reason="stop", stop_reason=0)
            for _ in range(sampling.n)
        ]) for _ in prompts]


class FakeLoRARequest:
    def __init__(self, name, identifier, path):
        self.path = path


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        FakeLLM.engines.clear()
        FakeLLM.calls.clear()
        FakeTokenizer.calls.clear()
        modules = {
            "vllm": types.SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams, __version__="test"),
            "vllm.lora.request": types.SimpleNamespace(LoRARequest=FakeLoRARequest),
            "transformers": types.SimpleNamespace(AutoTokenizer=FakeTokenizer),
            "torch": types.SimpleNamespace(__version__="test"),
        }
        self.modules = patch.dict(sys.modules, modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.math = load_module("math_evaluation_test", "eval/math_avg12.py")

    def run_math(self, directory, *extra):
        data = directory / "math.jsonl"
        if not data.exists():
            data.write_text(''.join(json.dumps({"id": str(i), "problem": "What is 17 + 25?", "answer": "42"}) + "\n" for i in range(30)))
        argv = ["math_avg12.py", "--base_model", "base", "--dataset", "aime24",
                "--dataset_jsonl", str(data), "--output_dir", str(directory / "out"), *extra]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.math.main(), 0)
        return json.loads((directory / "out/run_summary.json").read_text())

    def test_math_sampling_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = self.run_math(Path(tmp))
        sampling, kwargs = FakeLLM.calls[0]
        self.assertEqual((sampling.seed, sampling.n, sampling.max_tokens), (2026, 12, 38912))
        self.assertEqual((sampling.temperature, sampling.top_p, sampling.top_k), (1.0, 0.95, -1))
        self.assertEqual(FakeLLM.engines[0]["seed"], 2026)
        self.assertEqual(FakeLLM.engines[0]["max_model_len"], 40960)
        self.assertIsNone(kwargs["lora_request"])
        self.assertTrue(all(call["enable_thinking"] for call in FakeTokenizer.calls))
        self.assertEqual(summary["total_solutions"], 360)
        self.assertEqual(summary["average_at_n_pct"], 100.0)
        self.assertEqual({g["seed"] for row in summary["results"] for g in row["generations"]}, {2026})

    def test_math_resume_skips_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.run_math(Path(tmp))
            self.run_math(Path(tmp))
        self.assertEqual(len(FakeLLM.calls), 1)

    def test_math_resume_rejects_changed_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.run_math(Path(tmp))
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                self.run_math(Path(tmp), "--seed", "7")
        self.assertEqual(len(FakeLLM.calls), 1)

    def test_math_resume_rejects_duplicate_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.run_math(directory)
            path = directory / "out/generations.json"
            payload = json.loads(path.read_text())
            payload["results"][0]["generations"][1]["sample_index"] = 0
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "indices mismatch"):
                self.run_math(directory)

    def test_math_adapter_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            adapter = directory / "adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"test")
            self.run_math(directory, "--checkpoint_dir", str(adapter))
        self.assertTrue(FakeLLM.engines[0]["enable_lora"])
        self.assertEqual(FakeLLM.calls[0][1]["lora_request"].path, str(adapter))

    def test_math_rejects_incomplete_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "math.jsonl").write_text(json.dumps({"id": "0", "problem": "What is 17 + 25?", "answer": "42"}) + "\n")
            with self.assertRaisesRegex(ValueError, "expected 30 problems"):
                self.run_math(directory)
        self.assertEqual(len(FakeLLM.calls), 0)

    def test_gpqa_base_and_adapter(self):
        gpqa = load_module("gpqa_evaluation_test", "eval/ood/gpqa_avg4.py")
        for adapter in (None, "adapter"):
            with self.subTest(adapter=adapter), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                data = directory / "gpqa.jsonl"
                data.write_text(json.dumps({"id": "question", "question": "Choose A", "options": {"A": "A", "B": "B", "C": "C", "D": "D"}, "answer": "A"}) + "\n")
                argv = ["gpqa_avg4.py", "--base", "base", "--data", str(data), "--out", str(directory / "out"), "--stride_index", "0", "--stride_count", "1"]
                if adapter:
                    argv.extend(["--adapter", adapter])
                with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                    gpqa.main()
                summary = json.loads((directory / "out/run_summary.json").read_text())
                self.assertEqual((summary["generations"], summary["accuracy_avg4"]), (4, 1.0))
                self.assertEqual(FakeLLM.engines[-1]["enable_lora"], bool(adapter))
                sampling, kwargs = FakeLLM.calls[-1]
                self.assertEqual((sampling.seed, sampling.n), (2026071201, 4))
                self.assertEqual(sampling.max_tokens, 38912)
                self.assertEqual(FakeLLM.engines[-1]["max_model_len"], 40960)
                self.assertEqual(kwargs["lora_request"].path if adapter else kwargs["lora_request"], adapter)

    def test_mmlu_base_and_adapter(self):
        mmlu = load_module("mmlu_evaluation_test", "eval/ood/mmlu_pro_avg4.py")
        for adapter in (None, "adapter"):
            with self.subTest(adapter=adapter), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                data = directory / "mmlu.jsonl"
                data.write_text(json.dumps({"question_id": "question", "question": "Choose A", "options": ["A", "B", "C", "D"], "answer": "A", "category": "math"}) + "\n")
                argv = ["mmlu_pro_avg4.py", "--base", "base", "--data", str(data), "--out", str(directory / "out")]
                argv.extend(["--adapter", adapter] if adapter else ["--no_lora"])
                with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                    mmlu.main()
                summary = json.loads((directory / "out/run_summary.json").read_text())
                self.assertEqual((summary["generations"], summary["accuracy_avg4"]), (4, 1.0))
                self.assertEqual(FakeLLM.engines[-1]["enable_lora"], bool(adapter))
                sampling, kwargs = FakeLLM.calls[-1]
                self.assertEqual((sampling.seed, sampling.n, sampling.max_tokens), (2026071201, 4, 8192))
                self.assertEqual(FakeLLM.engines[-1]["max_model_len"], 12288)
                self.assertEqual(kwargs["lora_request"].path if adapter else kwargs["lora_request"], adapter)
                self.assertTrue(FakeTokenizer.calls[-1]["enable_thinking"])


if __name__ == "__main__":
    unittest.main()

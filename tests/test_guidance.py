"""Offline checks for repair prompts, validation, and launcher configuration."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from air_opd.core import hint_client


class GuidanceTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"SOCRATIC_REPAIR_GUIDANCE_PROMPTS": str(ROOT / "configs/prompts/repair_guidance_prompts.json")})
        env.start()
        self.addCleanup(env.stop)
        hint_client._REFERENCE_ADAPTIVE_REPAIR_PROMPTS = None

    def item(self, round_number=1):
        return {
            "problem": "What is 17 + 25?",
            "gold": "42",
            "last_attempt": "17 + 25 = 41",
            "teacher_context": "Add 17 and 25 to obtain 42.",
            "round": round_number,
            "hints": [] if round_number == 1 else ["Add place values separately."],
        }

    def test_prompt_bundle_and_all_rounds(self):
        for round_number in (1, 2, 3):
            with self.subTest(round=round_number):
                messages = hint_client.build_reference_multiround_messages(self.item(round_number))
                self.assertEqual([m["role"] for m in messages], ["system", "user"])
                self.assertIn("17 + 25 = 41", messages[1]["content"])
                self.assertIn("Add 17 and 25 to obtain 42.", messages[1]["content"])
                self.assertIn("<repair_guidance>", messages[1]["content"])
                self.assertEqual("[Current Repair Guidance]" in messages[1]["content"], round_number > 1)

    def test_valid_repair_anchor(self):
        text = "<decision>repair</decision><error_quote>17 + 25 = 41</error_quote><repair_guidance>Add place values separately, then combine the carry.</repair_guidance>"
        guidance, meta = hint_client.sanitize_adaptive_repair_repair_hint_with_meta(text, "42", task_type="math", quote_source="17 + 25 = 41", round_number=1)
        self.assertTrue(guidance)
        self.assertIsNone(meta["fallback_reason"])

    def test_answer_leak_is_rejected(self):
        text = "<decision>repair</decision><error_quote>17 + 25 = 41</error_quote><repair_guidance>The final answer is \\boxed{42}.</repair_guidance>"
        guidance, meta = hint_client.sanitize_adaptive_repair_repair_hint_with_meta(text, "42", task_type="math", quote_source="17 + 25 = 41", round_number=1)
        self.assertFalse(guidance)
        self.assertEqual(meta["fallback_reason"], "answer_leak")

    def test_unlocatable_anchor_is_rejected(self):
        text = "<decision>repair</decision><error_quote>unrelated claim</error_quote><repair_guidance>Add place values separately.</repair_guidance>"
        guidance, meta = hint_client.sanitize_adaptive_repair_repair_hint_with_meta(text, "42", task_type="math", quote_source="17 + 25 = 41", round_number=1)
        self.assertFalse(guidance)
        self.assertTrue(meta["fallback_reason"])

    def test_no_repair_stops_supervision(self):
        guidance, meta = hint_client.sanitize_adaptive_repair_repair_hint_with_meta("<decision>no_repair</decision>", "42", task_type="math", quote_source="17 + 25 = 41", round_number=1)
        self.assertFalse(guidance)
        self.assertEqual(meta["fallback_reason"], "decision_no_repair")


class LauncherTests(unittest.TestCase):
    def test_both_guidance_recipes(self):
        for backend in ("self", "external"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                model = directory / "model"
                model.mkdir()
                (model / "config.json").write_text("{}")
                data = directory / "train.jsonl"
                data.write_text('{"problem":"17+25","answer":"42","solution":"Add."}\n')
                profile = directory / "profile.env"
                profile.write_text((ROOT / f"configs/qwen3_4b_{backend}_guidance.env").read_text() + f"\nMODEL_PATH={model}\nTRAIN_JSONL={data}\nOUTPUT_DIR={directory / 'output'}\n" + (f"SOCRATIC_REFLECTOR_MODEL_PATH={model}\n" if backend == "external" else ""))
                stub = directory / "python3"
                stub.write_text(f"#!{sys.executable}\nimport json,os,sys\nfrom pathlib import Path\nPath(os.environ['CAPTURE_PATH']).write_text(json.dumps({{'argv':sys.argv[1:],'prompt':os.environ['SOCRATIC_REPAIR_GUIDANCE_PROMPTS']}}))\n")
                stub.chmod(0o755)
                capture = directory / "capture.json"
                env = dict(os.environ, PATH=str(directory) + os.pathsep + os.environ.get("PATH", ""), CAPTURE_PATH=str(capture))
                result = subprocess.run(["bash", str(ROOT / "scripts/train.sh"), str(profile)], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
                self.assertTrue(capture.is_file(), result.stdout + result.stderr)
                payload = json.loads(capture.read_text())
                argv = payload["argv"]
                expected = {
                    "--num_processes": "8", "--per_device_train_batch_size": "1", "--gradient_accumulation_steps": "4",
                    "--max_steps": "100", "--save_steps": "20", "--save_total_limit": "5",
                    "--learning_rate": "5e-6", "--lora_r": "64", "--lora_alpha": "128",
                    "--max_completion_length": "8192", "--max_length": "19168",
                    "--temperature": "1.1", "--top_p": "0.95", "--top_k": "20",
                    "--vllm_sync_frequency": "1", "--socratic_max_rounds": "3",
                    "--socratic_privilege_mode": "repair_guidance", "--socratic_repair_selection_mode": "reference_adaptive_repair_replacement",
                    "--socratic_reflection_prompt_mode": "adaptive_repair_self_xml_v1",
                    "--socratic_round_weight_alpha": "0.8", "--socratic_round_weight_rho": "1.5",
                    "--socratic_endpoint_epsilon": "0.10", "--hint_max_new_tokens": "8192",
                    "--hint_temperature": "0.7", "--hint_top_p": "0.9", "--socratic_self_hint_thinking": "1",
                    "--hinter_backend": backend + "_model",
                }
                for option, value in expected.items():
                    self.assertEqual(argv[argv.index(option) + 1], value, option)
                self.assertTrue(Path(payload["prompt"]).is_file())


if __name__ == "__main__":
    unittest.main()

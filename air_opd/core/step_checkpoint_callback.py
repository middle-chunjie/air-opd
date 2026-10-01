"""Transformers callback for sealing optimizer-step checkpoints."""
from __future__ import annotations

import os
from pathlib import Path

from transformers import TrainerCallback

from air_opd.core.step_checkpoint import finalize_checkpoint, rotate_checkpoints


class StepCheckpointCallback(TrainerCallback):
    """Seal every Trainer checkpoint, then apply milestone-aware retention."""

    def __init__(
        self,
        trainer,
        *,
        milestone_interval: int = 25,
        rolling_keep: int = 2,
    ) -> None:
        self.trainer = trainer
        self.milestone_interval = int(milestone_interval)
        self.rolling_keep = int(rolling_keep)

    def on_train_begin(self, args, state, control, **kwargs):
        del kwargs
        if float(args.save_steps) != 1.0:
            raise RuntimeError(
                f"step checkpointing requires save_steps=1, got {args.save_steps!r}"
            )
        # Transformers restores this scheduling field from trainer_state.json.
        # Rebind it after resume so an older checkpoint cadence cannot suppress
        # the first post-resume save.
        state.save_steps = 1
        return control

    def on_save(self, args, state, control, **kwargs):
        del kwargs
        self.trainer.accelerator.wait_for_everyone()
        if self.trainer.accelerator.is_main_process:
            checkpoint = Path(args.output_dir) / f"checkpoint-{int(state.global_step)}"
            finalize_checkpoint(
                checkpoint,
                world_size=int(self.trainer.accelerator.num_processes),
                milestone_interval=self.milestone_interval,
            )
            rotate_checkpoints(
                args.output_dir,
                world_size=int(self.trainer.accelerator.num_processes),
                milestone_interval=self.milestone_interval,
                rolling_keep=self.rolling_keep,
            )
            latest = Path(args.output_dir) / "checkpoint-latest"
            tmp = latest.with_name(f".{latest.name}.tmp.{os.getpid()}")
            tmp.symlink_to(checkpoint.name)
            os.replace(tmp, latest)
        self.trainer.accelerator.wait_for_everyone()
        return control

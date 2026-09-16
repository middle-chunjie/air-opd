"""Socratic-OPSD runtime overlay.

Multi-round external-hint repair on top of OPSD-style on-policy self-distillation.
The hint loop is a trajectory-generation mechanism only; training is pure OPSD
JSD distillation with an optional fixed round-weight policy (no RL reward).

Modules:
- sci_verifier : SciKnowEval MCQ verifier (ported from SocraticPO sci_eval.py).
- math_verifier: boxed-answer verifier for OPSD/OpenThoughts math.
- render       : student / teacher / guided prompt renderers (MCQ and math).
- repair       : engine-agnostic multi-round repair state machine.
- hint_client  : external OpenAI-compatible hinter + answer-leak guard.
"""

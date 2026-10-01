#!/usr/bin/env bash
# Portable AIR-OPD training launcher.
#
#   bash scripts/train.sh configs/qwen3_4b_self_guidance.env
#
# See README.md for required model and data paths.
set -euo pipefail

PROFILE=${1:?usage: scripts/train.sh <config.env>}
test -s "${PROFILE}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a
# shellcheck disable=SC1090
source "${PROFILE}"
set +a

: "${MODEL_PATH:?set MODEL_PATH to the base model directory}"
: "${TRAIN_JSONL:?set TRAIN_JSONL to the annotated training corpus}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to the run output directory}"
test -s "${MODEL_PATH}/config.json"
test -s "${TRAIN_JSONL}"

NUM_PROCESSES=${NUM_PROCESSES:-8}
PER_DEVICE_BS=${PER_DEVICE_BS:-1}
GRAD_ACCUM=${GRAD_ACCUM:-4}
EFFECTIVE_BATCH=$((NUM_PROCESSES * PER_DEVICE_BS * GRAD_ACCUM))
echo "[air-opd] effective batch size = ${EFFECTIVE_BATCH} (paper: 32)"

ACCEL_CONFIG="$(mktemp -t air_opd_accelerate.XXXXXX.yaml)"
trap 'rm -f "${ACCEL_CONFIG}"' EXIT
cat > "${ACCEL_CONFIG}" <<YAML
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
machine_rank: 0
main_process_ip: null
main_process_port: ${MASTER_PORT:-29500}
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: ${NUM_PROCESSES}
rdzv_backend: static
same_network: true
use_cpu: false
YAML

JSD_MEMORY_ARGS=()
if [ "${JSD_SEQUENCE_CHUNK_SIZE:-0}" -gt 0 ]; then
  JSD_MEMORY_ARGS+=(--jsd_sequence_chunk_size "${JSD_SEQUENCE_CHUNK_SIZE}")
fi
VLLM_SLEEP_ARGS=()
if [ "${VLLM_ENABLE_SLEEP:-0}" = "1" ]; then
  VLLM_SLEEP_ARGS+=(--vllm_enable_sleep_mode)
fi
REFLECTOR_ARGS=()
if [ -n "${SOCRATIC_REFLECTOR_MODEL_PATH:-}" ]; then
  REFLECTOR_ARGS+=(
    --socratic_reflector_model_path "${SOCRATIC_REFLECTOR_MODEL_PATH}"
    --socratic_reflector_gpu_memory_utilization "${SOCRATIC_REFLECTOR_GPU_MEMORY_UTILIZATION:-0.18}"
    --socratic_reflector_max_model_len "${SOCRATIC_REFLECTOR_MAX_MODEL_LEN:-16384}"
    --socratic_reflector_max_num_seqs "${SOCRATIC_REFLECTOR_MAX_NUM_SEQS:-16}"
    --socratic_reflector_enforce_eager "${SOCRATIC_REFLECTOR_ENFORCE_EAGER:-True}"
  )
fi

# Resolve a repo-relative prompt bundle against the repository root.
if [ -n "${SOCRATIC_REPAIR_GUIDANCE_PROMPTS:-}" ] && [ "${SOCRATIC_REPAIR_GUIDANCE_PROMPTS#/}" = "${SOCRATIC_REPAIR_GUIDANCE_PROMPTS}" ]; then
  SOCRATIC_REPAIR_GUIDANCE_PROMPTS="${REPO_ROOT}/${SOCRATIC_REPAIR_GUIDANCE_PROMPTS}"
  export SOCRATIC_REPAIR_GUIDANCE_PROMPTS
fi
test -s "${SOCRATIC_REPAIR_GUIDANCE_PROMPTS:?set SOCRATIC_REPAIR_GUIDANCE_PROMPTS}"

# External guidance needs the separate generator model.
if [ "${HINTER_BACKEND:-self_model}" = "external_model" ] && [ -z "${SOCRATIC_REFLECTOR_MODEL_PATH:-}" ]; then
  echo "[air-opd] HINTER_BACKEND=external_model requires SOCRATIC_REFLECTOR_MODEL_PATH" >&2
  exit 2
fi
if [ -n "${SOCRATIC_REFLECTOR_MODEL_PATH:-}" ]; then
  test -s "${SOCRATIC_REFLECTOR_MODEL_PATH}/config.json"
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

exec python3 -m accelerate.commands.launch \
  --config_file "${ACCEL_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" --main_process_port "${MASTER_PORT:-29500}" \
  "${REPO_ROOT}/air_opd/train.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_kind openthoughts_math --openthoughts_jsonl "${TRAIN_JSONL}" \
  --max_train_rows "${MAX_TRAIN_ROWS:--1}" \
  --output_dir "${OUTPUT_DIR}" --run_config "${RUN_ID:-air_opd_run}" \
  --num_train_epochs 1 --max_steps "${MAX_STEPS:-100}" --save_strategy steps \
  --save_steps "${SAVE_STEPS:-20}" --save_only_model False \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-5}" --logging_steps 1 \
  --per_device_train_batch_size "${PER_DEVICE_BS}" --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --steps_per_generation "${STEPS_PER_GENERATION:-4}" --shuffle_dataset "${SHUFFLE_DATASET:-True}" \
  --optim adamw_torch_fused --learning_rate "${LEARNING_RATE:-5e-6}" \
  --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8 --weight_decay 0 \
  --lr_scheduler_type linear --warmup_ratio 0 --warmup_steps 0 \
  --seed 42 --data_seed "${DATA_SEED:-42}" \
  --max_grad_norm 0.1 --gradient_checkpointing --ddp_find_unused_parameters False \
  --max_completion_length "${MAX_COMPLETION_LENGTH:-8192}" --max_length "${MAX_LENGTH:-19168}" \
  --attn_implementation "${ATTN_IMPL:-flash_attention_2}" --no_remove_unused_columns \
  --beta 0 --jsd_token_clip 0.05 --lmbda 1 \
  ${JSD_MEMORY_ARGS[@]+"${JSD_MEMORY_ARGS[@]}"} \
  --temperature "${STUDENT_TEMPERATURE:-1.1}" --top_p "${STUDENT_TOP_P:-0.95}" --top_k "${STUDENT_TOP_K:-20}" \
  --use_vllm --vllm_mode colocate --vllm_tensor_parallel_size 1 \
  ${VLLM_SLEEP_ARGS[@]+"${VLLM_SLEEP_ARGS[@]}"} \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEM:-0.35}" --vllm_sync_frequency 1 \
  --use_peft --lora_r "${LORA_R:-64}" --lora_alpha "${LORA_ALPHA:-128}" \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --fixed_teacher --student_thinking False --teacher_thinking True \
  --socratic_self_hint_thinking "${SOCRATIC_SELF_HINT_THINKING:-1}" \
  --socratic_max_rounds "${SOCRATIC_MAX_ROUNDS:-3}" --socratic_distill_mode "${DISTILL_MODE:-guided}" \
  --socratic_privilege_mode "${SOCRATIC_PRIVILEGE_MODE:-repair_guidance}" \
  --socratic_leak_fallback_mode "${SOCRATIC_LEAK_FALLBACK_MODE}" \
  --socratic_supervision_scope "${SOCRATIC_SUPERVISION_SCOPE:-initial_wrong_only}" \
  --socratic_solution_privilege_mode "${SOCRATIC_SOLUTION_PRIVILEGE_MODE:-exclude}" \
  --math_verifier_profile "${SOCRATIC_MATH_VERIFIER_PROFILE:-legacy}" \
  --socratic_key_idea_privilege_context "${SOCRATIC_KEY_IDEA_PRIVILEGE_CONTEXT:-problem_only}" \
  --socratic_loss_mask_mode "${SOCRATIC_LOSS_MASK_MODE:-after_error_quote}" \
  --socratic_repair_selection_mode "${SOCRATIC_REPAIR_SELECTION_MODE}" \
  --socratic_round_weight_mode "${SOCRATIC_ROUND_WEIGHT_MODE}" \
  --socratic_round_weight_alpha "${SOCRATIC_ROUND_WEIGHT_ALPHA:-0.8}" \
  --socratic_round_weight_rho "${SOCRATIC_ROUND_WEIGHT_RHO:-1.5}" \
  --socratic_endpoint_mode "${SOCRATIC_ENDPOINT_MODE:-cumulative_correction_coverage}" \
  --socratic_endpoint_epsilon "${SOCRATIC_ENDPOINT_EPSILON:-0.10}" \
  --socratic_terminal_retry "${SOCRATIC_TERMINAL_RETRY:-False}" \
  --socratic_max_repair_attempts "${SOCRATIC_MAX_REPAIR_ATTEMPTS:-1}" \
  --socratic_reflection_context_mode "${SOCRATIC_REFLECTION_CONTEXT_MODE:-last_wrong}" \
  --socratic_reflection_prompt_mode "${SOCRATIC_REFLECTION_PROMPT_MODE}" \
  --scope_classifier_max_new_tokens "${SCOPE_CLASSIFIER_MAX_NEW_TOKENS:-32}" \
  --scope_classifier_temperature "${SCOPE_CLASSIFIER_TEMPERATURE:-0.0}" \
  --socratic_error_route_mode "${SOCRATIC_ERROR_ROUTE_MODE:-legacy}" \
  --socratic_error_quote_locator_mode "${SOCRATIC_ERROR_QUOTE_LOCATOR_MODE:-locatable}" \
  --socratic_error_quote_fallback_mode "${SOCRATIC_ERROR_QUOTE_FALLBACK_MODE:-quote_source_prefix}" \
  --socratic_wrong_pair_filter_mode "${SOCRATIC_WRONG_PAIR_FILTER_MODE:-none}" \
  ${REFLECTOR_ARGS[@]+"${REFLECTOR_ARGS[@]}"} \
  --hinter_backend "${HINTER_BACKEND:-self_model}" --hint_max_new_tokens "${HINT_MAX_NEW_TOKENS:-8192}" \
  --hint_concurrency "${HINT_CONCURRENCY:-8}" \
  --hint_temperature "${HINT_TEMPERATURE:-0.7}" --hint_top_p "${HINT_TOP_P:-0.9}" \
  --hint_log_path "${OUTPUT_DIR}/hint_context_log.jsonl"

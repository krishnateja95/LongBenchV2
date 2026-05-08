#!/usr/bin/env bash
set -euo pipefail

# source /home/krishnateja95/virtual_envs/HIGGS/bin/activate
source /home/krishnateja95/virtual_envs/eval_models/bin/activate

export CUDA_HOME=/usr/local/cuda

RESULTS_DIR="${RESULTS_DIR:-eval_results_longbench2}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8088}"
VLLM_STARTUP_TIMEOUT_SEC="${VLLM_STARTUP_TIMEOUT_SEC:-2700}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-260000}"
LM_EVAL_MAX_LENGTH="${LM_EVAL_MAX_LENGTH:-260000}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-32000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
NUM_CONCURRENT="${NUM_CONCURRENT:-128}"
TIMEOUT="${TIMEOUT:-1200}"
LM_EVAL_BIN="${LM_EVAL_BIN:-lm_eval}"

# Reasoning parser for Qwen3-family thinking models. vLLM will route
# <think>...</think> traces into a separate `reasoning_content` field so
# the chat-completions `content` returned to lm-eval contains only the
# final answer. Set REASONING_PARSER="" to disable.
REASONING_PARSER="${REASONING_PARSER:-qwen3}"

# Skip the vision encoder for text-only benchmarks like LongBench v2.
# Frees memory for KV cache, which matters at 260k context. Set to 0 for
# multimodal evals.
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"

# Mitigates thinking-trace repetition loops under greedy decoding.
# Qwen3.6 README recommends 1.5 for thinking mode.
PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}"

# Set LIMIT=N to cap samples per task (useful for smoke tests).
# Leave empty for full eval.
LIMIT="${LIMIT:-}"

VLLM_PID=""

models=(
  "Qwen/Qwen3.6-35B-A3B"
)

# Format: "suite|task|fewshot|repeats"
LM_EVAL_LONGBENCH2_JOBS=(
#   "longbench2|longbench2_govt_single|0|1"
#   "longbench2|longbench2_legal_single|0|1"
#   "longbench2|longbench2_lit_single|0|1"
#   "longbench2|longbench2_fin_single|0|1"
#   "longbench2|longbench2_academic_single|0|1"
  # "longbench2|longbench2_detective|0|1"
#   "longbench2|longbench2_event_order|0|1"
#   "longbench2|longbench2_govt_multi|0|1"
#   "longbench2|longbench2_academic_multi|0|1"
#   "longbench2|longbench2_fin_multi|0|1"
#   "longbench2|longbench2_legal_multi|0|1"
  # "longbench2|longbench2_news_multi|0|1"
#   "longbench2|longbench2_user_guide|0|1"
#   "longbench2|longbench2_translate|0|1"
  "longbench2|longbench2_many_shot|0|1"
  "longbench2|longbench2_agent_history|0|1"
  "longbench2|longbench2_dialogue_history|0|1"
#   "longbench2|longbench2_code|0|1"
#   "longbench2|longbench2_graph|0|1"
#   "longbench2|longbench2_table|0|1"
)

SEEDS=(111 222 333)

slugify() {
  python3 -c "
import re, sys
s = sys.argv[1].strip().replace('\\\\', '/')
s = re.sub(r'^/+', '', s).replace('/', '__')
print(re.sub(r'[^A-Za-z0-9._-]+', '_', s))
" "$1"
}

start_vllm_server() {
  local model="$1" name="$2" port="$3"

  echo "Starting vLLM for ${model} on port ${port}"
  local extra_args=()
  if [[ -n "$REASONING_PARSER" ]]; then
    extra_args+=(--reasoning-parser "$REASONING_PARSER")
  fi
  if [[ "$LANGUAGE_MODEL_ONLY" == "1" ]]; then
    extra_args+=(--language-model-only)
  fi

  nohup vllm serve "$model" \
    --host "$VLLM_HOST" \
    --port "$port" \
    --served-model-name "$name" \
    --dtype auto \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enable-chunked-prefill \
    --trust-remote-code \
    "${extra_args[@]}" \
    > "${RESULTS_DIR}/vllm_${name}.log" 2>&1 &
  VLLM_PID=$!
}

wait_for_vllm() {
  local port="$1"
  local attempts=$(( (VLLM_STARTUP_TIMEOUT_SEC + 4) / 5 ))
  for _ in $(seq 1 "$attempts"); do
    curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && return 0
    if [[ -n "$VLLM_PID" ]] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "ERROR: vLLM died on port ${port}"
      return 1
    fi
    sleep 5
  done
  echo "ERROR: vLLM did not become ready within ${VLLM_STARTUP_TIMEOUT_SEC}s on port ${port}"
  return 1
}

stop_vllm() {
  if [[ -n "$VLLM_PID" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""
  fi
  pkill -f "vllm serve.*--port ${VLLM_PORT}" 2>/dev/null || true
  sleep 3
}

run_lm_eval_task() {
  local model="$1" name="$2" suite="$3" task="$4" fewshot="$5" seed="$6" port="$7"
  local out="${RESULTS_DIR}/$(slugify "$model")/${suite}/${task}/shot_${fewshot}/seed_${seed}"
  mkdir -p "$out"
  if [[ -e "${out}/results.json" ]]; then
    echo "Skipping existing: ${out}"
    return 0
  fi

  # Chat-completions backend: server applies the chat template, so we do NOT
  # send pre-tokenized requests and we do NOT pass max_length in model_args
  # (the server enforces context via --max-model-len).
  local model_args="model=${name},tokenizer=${model},base_url=http://127.0.0.1:${port}/v1/chat/completions,num_concurrent=${NUM_CONCURRENT},max_retries=3,tokenized_requests=False,timeout=${TIMEOUT}"
  local gen_kwargs="temperature=0.0,max_gen_toks=${MAX_GEN_TOKS},presence_penalty=${PRESENCE_PENALTY}"

  local limit_arg=()
  if [[ -n "$LIMIT" ]]; then
    limit_arg=(--limit "$LIMIT")
  fi

  echo "lm-eval: ${suite}/${task} fewshot=${fewshot} seed=${seed} limit=${LIMIT:-all}"
  "$LM_EVAL_BIN" \
    --model local-chat-completions \
    --tasks "$task" \
    "${limit_arg[@]}" \
    --model_args "$model_args" \
    --num_fewshot "$fewshot" \
    --seed "$seed" \
    --gen_kwargs "$gen_kwargs" \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --log_samples \
    --output_path "$out"
}

run_jobs() {
  local model="$1" name="$2" port="$3"
  for job in "${LM_EVAL_LONGBENCH2_JOBS[@]}"; do
    IFS='|' read -r suite task fewshot reps <<< "$job"
    for i in $(seq 0 $((reps - 1))); do
      run_lm_eval_task "$model" "$name" "$suite" "$task" "$fewshot" "${SEEDS[$i]}" "$port"
    done
  done
}

mkdir -p "$RESULTS_DIR"
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"
echo "RESULTS_DIR=${RESULTS_DIR}"
trap stop_vllm EXIT

for model in "${models[@]}"; do
  echo "==================== ${model} ===================="
  name="$(slugify "$model")"
  mkdir -p "${RESULTS_DIR}/${name}"

  start_vllm_server "$model" "$name" "$VLLM_PORT"
  wait_for_vllm "$VLLM_PORT" || { echo "ERROR: vLLM failed for ${model}"; exit 1; }
  run_jobs "$model" "$name" "$VLLM_PORT"
  stop_vllm

  echo "Done: ${model}"
done
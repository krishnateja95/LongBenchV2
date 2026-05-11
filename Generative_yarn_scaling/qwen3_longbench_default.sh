#!/usr/bin/env bash
set -euo pipefail

export CUDA_HOME=/usr/local/cuda

RESULTS_DIR="${RESULTS_DIR:-eval_results_longbench2}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8088}"
VLLM_STARTUP_TIMEOUT_SEC="${VLLM_STARTUP_TIMEOUT_SEC:-2700}"

# Qwen3.6 native context is 262,144.
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-262144}"

# Leave room for generation inside the native context window:
# 262,144 - 32,768 = 229,376
LM_EVAL_MAX_LENGTH="${LM_EVAL_MAX_LENGTH:-229376}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-32768}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
NUM_CONCURRENT="${NUM_CONCURRENT:-128}"
TIMEOUT="${TIMEOUT:-1200}"
LM_EVAL_BIN="${LM_EVAL_BIN:-lm_eval}"

# Qwen3.6 / vLLM specific knobs
VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-qwen3}"
VLLM_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-1}"

# README-recommended thinking-mode defaults for general tasks.
# Override from env if you want different sampling.
GEN_KWARGS="${GEN_KWARGS:-temperature=1.0,top_p=0.95,presence_penalty=1.5,repetition_penalty=1.0,max_gen_toks=${MAX_GEN_TOKS}}"

VLLM_PID=""

models=(
  "Qwen/Qwen3.6-35B-A3B"
)

# Format: "suite|task|fewshot|repeats"
LM_EVAL_LONGBENCH2_JOBS=(
  "longbench2|longbench2_govt_single|0|1"
  "longbench2|longbench2_legal_single|0|1"
  "longbench2|longbench2_lit_single|0|1"
  "longbench2|longbench2_fin_single|0|1"
  "longbench2|longbench2_academic_single|0|1"
  "longbench2|longbench2_detective|0|1"
  "longbench2|longbench2_event_order|0|1"
  "longbench2|longbench2_govt_multi|0|1"
  "longbench2|longbench2_academic_multi|0|1"
  "longbench2|longbench2_fin_multi|0|1"
  "longbench2|longbench2_legal_multi|0|1"
  "longbench2|longbench2_news_multi|0|1"
  "longbench2|longbench2_user_guide|0|1"
  "longbench2|longbench2_translate|0|1"
  "longbench2|longbench2_many_shot|0|1"
  "longbench2|longbench2_agent_history|0|1"
  "longbench2|longbench2_dialogue_history|0|1"
  "longbench2|longbench2_code|0|1"
  "longbench2|longbench2_graph|0|1"
  "longbench2|longbench2_table|0|1"
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

  local cmd=(
    vllm serve "$model"
    --host "$VLLM_HOST"
    --port "$port"
    --served-model-name "$name"
    --dtype auto
    --max-model-len "$VLLM_MAX_MODEL_LEN"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --enable-chunked-prefill
    --trust-remote-code
    --reasoning-parser "$VLLM_REASONING_PARSER"
  )

  # LongBench2 is text-only, so skip the vision encoder to free memory/KV budget.
  if [[ "$VLLM_LANGUAGE_MODEL_ONLY" == "1" ]]; then
    cmd+=(--language-model-only)
  fi

  nohup "${cmd[@]}" > "${RESULTS_DIR}/vllm_${name}.log" 2>&1 &
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

  local model_args="model=${name},tokenizer=${model},max_length=${LM_EVAL_MAX_LENGTH},base_url=http://127.0.0.1:${port}/v1/completions,num_concurrent=${NUM_CONCURRENT},max_retries=3,tokenized_requests=True,timeout=${TIMEOUT}"

  echo "lm-eval: ${suite}/${task} fewshot=${fewshot} seed=${seed}"
  "$LM_EVAL_BIN" \
    --model local-completions \
    --tasks "$task" \
    --model_args "$model_args" \
    --num_fewshot "$fewshot" \
    --seed "$seed" \
    --gen_kwargs "$GEN_KWARGS" \
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
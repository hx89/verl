#!/usr/bin/env bash
# config_openmath_h100_32g: Colocated GRPO on 32x H100 (4 nodes) over
# OpenMathInstruct-2, Qwen3-30B-A3B-Base, bf16, max seq len 4096.
#
# Single 32-GPU pool, hybrid engine (generation + training time-share the GPUs):
#   - Generation (vLLM):  TP=2  -> 16 replicas (DP=16)
#   - Training (Megatron): TP=1, PP=1, CP=1, EP=8, ETP=1
#       -> dense DP=32; expert grid EP=8 x EDP=4 (the "EP=8, DP=4" mapping)
#
# Batch: 64 prompts/step x 32 generations = 2048 trajectories/step;
#        ppo_mini_batch_size=16 prompts x 32 = 512 trajectories/optimizer step
#        (= train_global_batch_size) -> 4 optimizer steps/step.
#
# Same conventions as config_b2_interactive.sh (env.sh, env-var secrets, ROCR
# fix, absolute hydra.run.dir, USE_DEEPEP toggle). Colocated sync path
# (verl.trainer.main_ppo), NOT the disaggregated async one.
#
# Secrets from the environment:
#   HF_TOKEN=hf_xxx WANDB_API_KEY=yyy ./config_openmath_h100_32gpu_interactive.sh <exp> <none|nsys|torch>

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "${SCRIPT_DIR}/env.sh" ] && source "${SCRIPT_DIR}/env.sh"

{ set +x; } 2>/dev/null
: "${HF_TOKEN:?not set — pass it at invocation, e.g. HF_TOKEN=hf_xxx WANDB_API_KEY=yyy ./config_openmath_h100_32gpu_interactive.sh <exp> <mode>}"
: "${WANDB_API_KEY:?not set — pass it at invocation, e.g. HF_TOKEN=hf_xxx WANDB_API_KEY=yyy ./config_openmath_h100_32gpu_interactive.sh <exp> <mode>}"
set -x

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export PYTHONPATH=${VERL_SRC_DIR:-/opt/workspace/RL/verl}:${PYTHONPATH:-}
export HF_HOME="${HF_HOME:-/opt/hf-cache}"
mkdir -p $HF_HOME

# ===================================== Topology =====================================
NNODES=4
GPUS_PER_NODE=8        # 32 H100 total (colocated gen+train)

# ===================================== Output / metadata =====================================
# ./config_openmath_h100_32gpu_interactive.sh <exp_base> <none|nsys|torch>  (default none)
WANDB_PROJECT_NAME="verl_grpo_openmath_h100"
EXPERIMENT_NAME_BASE="${1:-config_openmath_h100_32g-qwen_3_30b_a3b_base-openmathinstruct2-$(date +%Y_%m_%d)}"
PROFILE_MODE="${2:-${PROFILE_MODE:-none}}"
case "${EXPERIMENT_NAME_BASE}" in
  *_nsys|*_torch|*_none) EXPERIMENT_NAME="${EXPERIMENT_NAME_BASE}" ;;
  *) EXPERIMENT_NAME="${EXPERIMENT_NAME_BASE}_${PROFILE_MODE}" ;;
esac
OUT_DIR="${OUT_DIR:-${PWD}/logs}"
EXP_DIR="${OUT_DIR}/${EXPERIMENT_NAME}"
mkdir -p "${EXP_DIR}"
OUT_FILE="${EXP_DIR}/${EXPERIMENT_NAME}.log"
total_training_steps=15

VERL_SRC_DIR="${VERL_SRC_DIR:-/opt/workspace/RL/verl}"
git config --global --add safe.directory "${VERL_SRC_DIR}"
VERL_COMMIT=$(git -C "${VERL_SRC_DIR}" rev-parse HEAD)
echo "Using VERL Commit: ${VERL_COMMIT}"

JOB_INFO=(
    "+job_info.config_number='openmath-h100-32g'"
    "+job_info.cluster='computelab'"
    "+job_info.gpu='H100-SXM'"
    "+job_info.num_gpus=$((GPUS_PER_NODE * NNODES))"
    "+job_info.verl_container_image='${VERL_CONTAINER_IMAGE:-unknown}'"
    "+job_info.verl_commit=${VERL_COMMIT}"
    "+job_info.environment_name='openmathinstruct2'"
    "+job_info.environment_type='math'"
    "+job_info.architecture_name='Qwen3-30B-A3B-Base'"
    "+job_info.num_params='30B'"
    "+job_info.algorithm_name='grpo'"
    "+job_info.colocated=true"
    "+job_info.async_lag=0"
    "+job_info.log_dir=${EXP_DIR}"
    "+job_info.profile_mode=${PROFILE_MODE}"
)

# ===================================== Dataset / model =====================================
cd ${VERL_SRC_DIR}
DATA_SAVE_DIR=~/data/openmathinstruct2
OPENMATH_SPLIT="${OPENMATH_SPLIT:-train_1M}"   # train | train_1M | train_2M | train_5M
if [[ ! -f ${DATA_SAVE_DIR}/train.parquet || ! -f ${DATA_SAVE_DIR}/test.parquet ]]; then
    python examples/data_preprocess/openmathinstruct2.py \
        --local_save_dir ${DATA_SAVE_DIR} \
        --train_split "${OPENMATH_SPLIT}"
fi
TRAIN_FILE=${DATA_SAVE_DIR}/train.parquet
TEST_FILE=${DATA_SAVE_DIR}/test.parquet

HUGGINGFACE_MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Base}"
hf download "${HUGGINGFACE_MODEL_PATH}"

# ===================================== Algorithm =====================================
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001

actor_lr=1e-6
actor_lr_warmup_steps=10
actor_weight_decay=0.1
actor_clip_grad=1.0
actor_entropy_coeff=0
clip_ratio_low=0.2
clip_ratio_high=0.28
clip_ratio_c=10.0
loss_agg_mode="token-mean"
temperature=1.0
top_p=1.0
top_k=-1
val_temperature=1.0
val_top_p=0.7
val_top_k=-1

# ===================================== Data / batch =====================================
# max seq len 4096 = max_prompt_length + max_response_length.
max_prompt_length=1024
max_response_length=3072

train_prompt_bsz=64        # num_prompts_per_step
n_resp_per_prompt=32       # num_generations_per_prompt
train_prompt_mini_bsz=16   # 16 prompts x 32 = 512 trajectories = train_global_batch_size
n_resp_per_prompt_val=1

# Dynamic (token-budget) micro-batching — better than a fixed micro-batch for
# variable-length math responses; supported on the Megatron main_ppo path.
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))

# ===================================== Trainer Megatron parallelism =====================================
# 32 GPU: TP=1, PP=1, CP=1 -> dense DP=32. EP=8, ETP=1 -> expert grid EP=8 x EDP=4
# (matches the requested "training: EP=8, DP=4"). Colocated -> offload to free GPU
# memory for vLLM during generation.
OFFLOAD=True
TP_SIZE=1
CP_SIZE=1
PP_SIZE=1
VPP_SIZE=null
EP_SIZE=8
ETP_SIZE=1

# MoE token dispatcher. H100/Hopper has working DeepEP kernels; env.sh sets
# USE_DEEPEP=1. Falls back to Megatron alltoall when USE_DEEPEP=0.
USE_DEEPEP="${USE_DEEPEP:-0}"
if [[ "${USE_DEEPEP}" == "1" ]]; then
    MOE_DISPATCHER_CONFIG="+actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=flex"
else
    MOE_DISPATCHER_CONFIG="+actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall"
fi

ACTOR_MEGATRON_CONFIG="
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.actor.megatron.context_parallel_size=$CP_SIZE \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP_SIZE \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=$VPP_SIZE \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP_SIZE \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP_SIZE \
    actor_rollout_ref.actor.megatron.param_offload=$OFFLOAD \
    actor_rollout_ref.actor.megatron.grad_offload=$OFFLOAD \
    actor_rollout_ref.actor.megatron.optimizer_offload=$OFFLOAD \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    ${MOE_DISPATCHER_CONFIG} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True"

ACTOR_CONFIG="
    actor_rollout_ref.model.path=$HUGGINGFACE_MODEL_PATH \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=$actor_lr_warmup_steps \
    actor_rollout_ref.actor.optim.weight_decay=$actor_weight_decay \
    actor_rollout_ref.actor.optim.clip_grad=$actor_clip_grad \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=$clip_ratio_c \
    actor_rollout_ref.actor.entropy_coeff=$actor_entropy_coeff \
    actor_rollout_ref.actor.loss_agg_mode=$loss_agg_mode \
    actor_rollout_ref.actor.ppo_mini_batch_size=$train_prompt_mini_bsz \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_ppo_max_token_len"

REF_CONFIG="
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP_SIZE \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=$EP_SIZE \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=$ETP_SIZE \
    actor_rollout_ref.ref.megatron.param_offload=$OFFLOAD \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$infer_ppo_max_token_len"

# ===================================== Inference (vLLM) =====================================
# TP=2 -> 16 vLLM replicas (DP=16) across the 32 colocated GPUs.
rollout_name=vllm
infer_tp=2
gpu_memory_utilization=0.5
ROLLOUT_CONFIG="
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$(((max_prompt_length + max_response_length) * 4)) \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=$top_k \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$infer_ppo_max_token_len \
    actor_rollout_ref.rollout.val_kwargs.temperature=$val_temperature \
    actor_rollout_ref.rollout.val_kwargs.top_p=$val_top_p \
    actor_rollout_ref.rollout.val_kwargs.top_k=$val_top_k \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val"

# ===================================== Profiling =====================================
NSYS_DIR="${EXP_DIR}/nsys_profiles"
TORCH_PROFILE_DIR="${EXP_DIR}/torch_profiles"
PROFILER_STEPS='[5]'
mkdir -p "${NSYS_DIR}" "${TORCH_PROFILE_DIR}"
case "${PROFILE_MODE}" in
  nsys)
    PROFILER_CONFIG="
    global_profiler.tool=nsys \
    global_profiler.save_path=$EXP_DIR \
    global_profiler.steps=${PROFILER_STEPS} \
    global_profiler.profile_continuous_steps=False \
    global_profiler.global_tool_config.nsys.discrete=False \
    actor_rollout_ref.actor.profiler.enable=True \
    actor_rollout_ref.ref.profiler.enable=True \
    actor_rollout_ref.rollout.profiler.enable=False \
    actor_rollout_ref.actor.profiler.all_ranks=True \
    actor_rollout_ref.ref.profiler.all_ranks=True \
    actor_rollout_ref.actor.profiler.save_path=${NSYS_DIR} \
    actor_rollout_ref.ref.profiler.save_path=${NSYS_DIR}"
    ;;
  torch)
    PROFILER_CONFIG="
    global_profiler.tool=torch \
    global_profiler.save_path=$EXP_DIR \
    global_profiler.steps=${PROFILER_STEPS} \
    global_profiler.profile_continuous_steps=False \
    actor_rollout_ref.actor.profiler.enable=False \
    actor_rollout_ref.ref.profiler.enable=False \
    actor_rollout_ref.rollout.profiler.enable=True \
    actor_rollout_ref.rollout.profiler.all_ranks=True \
    actor_rollout_ref.rollout.profiler.tool=torch \
    actor_rollout_ref.rollout.profiler.save_path=${TORCH_PROFILE_DIR} \
    actor_rollout_ref.rollout.profiler.tool_config.torch.discrete=True \
    actor_rollout_ref.rollout.profiler.tool_config.torch.contents=[cuda,cpu,stack]"
    ;;
  none)
    PROFILER_CONFIG="
    global_profiler.steps=null \
    actor_rollout_ref.actor.profiler.enable=False \
    actor_rollout_ref.ref.profiler.enable=False \
    actor_rollout_ref.rollout.profiler.enable=False"
    ;;
  *)
    echo "Unknown PROFILE_MODE=${PROFILE_MODE}; use none, nsys, or torch" >&2
    exit 1
    ;;
esac

# ===================================== Ray runtime env =====================================
RAY_KWARGS="+ray_kwargs.ray_init.runtime_env={env_vars:{CUDA_DEVICE_MAX_CONNECTIONS:'1',NCCL_NVLS_ENABLE:'0',HF_HOME:'${HF_HOME}',ROCR_VISIBLE_DEVICES:''}}"

# ===================================== Run =====================================
cd "${EXP_DIR}"

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_megatron_trainer.yaml \
    hydra.run.dir="${EXP_DIR}/hydra/$(date +%Y-%m-%d/%H-%M-%S)" \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.train_batch_size=$train_prompt_bsz \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    $ACTOR_CONFIG \
    $ACTOR_MEGATRON_CONFIG \
    $REF_CONFIG \
    $ROLLOUT_CONFIG \
    $PROFILER_CONFIG \
    $RAY_KWARGS \
    reward_model.reward_manager=naive \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$WANDB_PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.val_before_train=False \
    trainer.test_freq=10 \
    trainer.save_freq=-1 \
    trainer.total_training_steps=$total_training_steps \
    trainer.total_epochs=10 \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    "${JOB_INFO[@]}" 2>&1 | tee ${OUT_FILE}

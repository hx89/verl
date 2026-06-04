# Codebase Architecture & Dev Commands

> Orientation map for agents working in `verl`. Read this before diving into an
> unfamiliar subsystem. For contribution policy and `uv` env setup see
> [`AGENTS.md`](../../AGENTS.md); for editing agent instructions see
> [`editing-agent-instructions.md`](editing-agent-instructions.md).
>
> Detailed reference docs live at <https://verl.readthedocs.io>. This guide
> captures the cross-file control/data flow and "where do I edit X" knowledge
> that the code alone doesn't make obvious.

## Big picture: the HybridFlow single-controller

verl runs RL post-training (PPO/GRPO/…) as a **single driver process** (the
"controller") that orchestrates many **worker groups** (actor, critic, ref,
reward, rollout) living on Ray actors across the GPU cluster.

- **`verl/protocol.py` — `DataProto`** is the one data container passed
  everywhere: a `TensorDict` (`.batch`) + numpy `non_tensor_batch` + `meta_info`.
  `DataProtoFuture` keeps results as Ray refs so the driver never materializes
  intermediate worker outputs until needed.
- **`verl/single_controller/`** — `WorkerGroup`/`RayWorkerGroup` +
  `ResourcePool`. Worker methods decorated with `@register(dispatch_mode=...)`
  become callable on the group: a driver call `wg.foo(data)` runs
  **dispatch** (partition `data` across N workers) → **execute** (`foo.remote()`
  on each) → **collect** (reassemble into one `DataProto`). See
  `single_controller/base/decorator.py` for the `Dispatch` modes.

**The PPO loop** (`verl/trainer/ppo/ray_trainer.py`, `RayPPOTrainer.fit()`):
`generate (rollout) → compute reward → old/ref log-probs → values (critic) →
compute_advantage (on the driver) → update critic → update actor → sync actor
weights to rollout`, with periodic `_validate()` / `_save_checkpoint()`.

## Subsystem map

| Area | Path | What lives here |
| --- | --- | --- |
| Entry points | `verl/trainer/main_ppo.py`, `main_ppo_sync.py`, `sft_trainer.py`, `main_eval.py`, `main_generation_server.py` | `@hydra.main` CLI entry points. **`main_ppo.py` is `@deprecated`** (to be replaced by `main_ppo_sync.py` in v0.8.0); the sync trainer uses `TransferQueue` + `ReplayBuffer`. |
| Config | `verl/trainer/config/**.yaml`, `verl/workers/config/*.py`, `verl/trainer/config/*.py`, `verl/base_config.py` | Hydra YAML tree + `BaseConfig` dataclasses. See *Configuration* below. |
| Controller core | `verl/single_controller/`, `verl/protocol.py` | Driver↔worker dispatch, `DataProto`. |
| Trainer & algos | `verl/trainer/ppo/ray_trainer.py`, `core_algos.py`, `reward.py`, `utils.py` (`Role` enum) | The fit loop; advantage estimators + policy losses. |
| Workers & engines | `verl/workers/engine_workers.py`, `verl/workers/engine/` (`base.py`, `fsdp/`, `megatron/`), `verl/workers/reward_manager/` | `TrainingWorker`/`ActorRolloutRefWorker`; the `BaseEngine` abstraction over FSDP/FSDP2/Megatron/VeOmni/TorchTitan; reward managers. |
| Rollout & agents | `verl/workers/rollout/` (`base.py`, `vllm_rollout/`, `sglang_rollout/`, `hf_rollout.py`), `verl/experimental/agent_loop/`, `verl/tools/` | Generation backends, multi-turn agent loop, tool calling. |
| Models | `verl/models/` (`registry.py`, `weight_loader_registry.py`, `transformers/`, `mcore/`) | Megatron model registry + HF↔parallel weight converters; packed-input HF model patches. |
| Weight sync | `verl/checkpoint_engine/` (`base.py` + nccl/hccl/nixl/mooncake), `verl/model_merger/` | Trainer→rollout weight transfer; merge sharded checkpoints back to HF. |
| Platform | `verl/plugin/platform/` | Hardware abstraction (CUDA/NPU/XPU/MetaX) replacing direct `torch.cuda.*`. |
| Experimental | `verl/experimental/` (`agent_loop`, `one_step_off_policy`, `fully_async_policy`, `transfer_queue`, `vla`, `separation`) | Async/agentic paradigms slated to merge into the core. |
| Examples / recipes | `examples/` (`run_<model>_<backend>.sh`), `recipe/` | Runnable scripts; **`recipe/` is a git submodule** — `git submodule update --init --recursive recipe`. |

## Configuration (Hydra + dataclasses)

YAML defines composition + defaults; each YAML carries a `_target_` pointing at
a `BaseConfig` dataclass that `omega_conf_to_dataclass` instantiates
(`verl/utils/config.py`). Root config: `verl/trainer/config/ppo_trainer.yaml`.
The `model_engine` variable selects strategy-specific sub-configs (e.g.
`dp_actor.yaml` for FSDP vs Megatron). Override on the CLI with dotted paths,
e.g. `actor_rollout_ref.actor.strategy=fsdp2 algorithm.gamma=0.99`.

- `BaseConfig` is **frozen** except fields listed in `_mutable_fields`; a
  subclass must OR-in the parent set: `_mutable_fields = BaseConfig._mutable_fields | {...}`.
- `verl/trainer/config/_generated_*.yaml` are **auto-generated** — never hand
  edit. Regenerate via the `autogen-trainer-cfg` pre-commit hook
  (`scripts/generate_trainer_config.sh`). YAML formatting (comments above
  fields, blank lines between fields) is CI-enforced.

## Extension points — "to add X, edit Y"

- **New RL algorithm** → `verl/trainer/ppo/core_algos.py`: register an advantage
  estimator with `@register_adv_est(...)` and/or a loss with
  `@register_policy_loss(...)`; select via `algorithm.adv_estimator` /
  `actor_rollout_ref.actor.policy_loss`. `compute_advantage()` runs on the driver.
- **Custom reward** → function-based reward loaded via `reward.*` config
  (`verl/trainer/ppo/reward.py`), or a new manager in
  `verl/workers/reward_manager/` registered in its registry.
- **New training backend** → implement `BaseEngine` under
  `verl/workers/engine/<backend>/`, register with `EngineRegistry`, add an
  `EngineConfig` subclass in `verl/workers/config/engine.py`; select via `strategy=`.
- **New rollout backend** → implement a `BaseRollout` adapter under
  `verl/workers/rollout/<backend>/` and add it to `_ROLLOUT_REGISTRY`
  (`verl/workers/rollout/base.py`); select via `actor_rollout_ref.rollout.name`.
- **New tool / tool-call format** → a `@function_tool` or `BaseTool` subclass in
  `verl/tools/`; for parsing model output, register a `ToolParser`
  (`verl/experimental/agent_loop/tool_parser.py`).
- **New model (Megatron)** → `verl/models/registry.py` + converters in
  `weight_loader_registry.py`. For FSDP/HF, add packed-input patches under
  `verl/models/transformers/`.
- **New config field** → edit the dataclass in `verl/{workers,trainer}/config/`
  *and* the matching YAML, then regenerate `_generated_*.yaml` (see above).
- **New hardware platform** → implement `PlatformBase` and register via
  `@PlatformRegistry.register` / setuptools entry points in `verl/plugin/platform/`.

## Dev commands

Env setup uses `uv` — see [`AGENTS.md`](../../AGENTS.md). Python ≥ 3.10.

```bash
# Install (editable, pick an inference backend)
pip install -e .[test,vllm]      # or .[test,sglang]

# Lint / format (ruff + mypy + sanity hooks, all via pre-commit)
pip install pre-commit hydra-core && pre-commit install
pre-commit run                                   # staged changes
pre-commit run --all-files                       # whole repo
pre-commit run --all-files ruff                  # one hook
pre-commit run --all-files autogen-trainer-cfg   # regenerate _generated_*.yaml

# Tests (default target is GPU; CPU-only tests end in *_on_cpu.py)
pytest -s tests/path/to/test_file_on_cpu.py                     # a CPU test file
pytest -s tests/path/to/test_file.py::test_name                # a single test
```

Test layout (`tests/README.md`): folders mirror `verl/` namespaces. `special_*`
dirs are excluded from the default unit-test workflows — `special_distributed`
(multi-GPU), `special_e2e` (training scripts), `special_npu`, `special_sanity`
(pre-commit sanity hooks), `special_standalone`. CI splits into
`cpu_unit_tests.yml` (`*_on_cpu.py`), `gpu_unit_tests.yml` (everything else
minus `special_*`/`vllm`/`sglang`), heavy `model.yml`/`vllm.yml`/`sgl.yml`, and
`e2e_*.yml`. PR-gating sanity (always-on): pre-commit, PR-title check, secrets
scan, docs.

## High-value gotchas

- **`compute_advantage` runs on the driver**, not workers. GRPO-family
  estimators need `non_tensor_batch['uid']` for group normalization, and the
  loop must call `compute_response_mask(batch)` before advantage/loss.
- **`DataProto.batch` is a lazy `TensorDict`** — call `.consolidate()` /
  `contiguous()` before Ray serialization; `non_tensor_batch` arrays must share
  the batch dim (`check_consistency()`).
- A worker method is only callable from the driver if decorated
  `@register(dispatch_mode=...)`; `mesh_name` routes to the `actor`/`ref`/`rollout` mesh.
- **Backend = `strategy`** (`fsdp`/`fsdp2`/`megatron`/…). Engine and actor/critic
  config classes are strategy-specific; FSDP2 needs torch ≥ 2.4 and a different
  offload config than FSDP1. Router replay (MoE) is Megatron/VeOmni only.
- Rollout weight sync is **generator-based, rank-by-rank** — every rank must
  iterate or FSDP collectives deadlock. `HYBRID` mode is HF-rollout only;
  vLLM/SGLang run as async servers.
- Naming is CI-checked: always spell the project **`verl`** (lowercase) and the
  engine **`SGLang`/`sglang`** — other casings fail pre-commit. Example scripts
  follow `run_<model>_<backend>.sh`.

Last updated: 06/04/2026

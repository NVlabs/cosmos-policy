"""TTA DPO model: LoRA on the real Cosmos Policy + shared-noise DPO forward.

Design (v1, decided 2026-09-09):
- pair-as-item: the forward batch contains BOTH sides of each pair ([B, 2, ...]
  flattened to [2B, ...]).
- shared noise: the stock draw gives every batch item INDEPENDENT noise, so
  share_noise_across_pairs() explicitly copies each pair's chosen-slot
  sigma/epsilon onto the rejected slot before the forward (review finding,
  2026-09-09 — the old "shared by construction" claim was wrong).
- the whole graph is one forward: loss = -logsigmoid(beta * (E_rej - E_chosen))
  on deltas E_policy - E_reference; reference = adapters disabled, no_grad.
- bf16: sigma and the clean latent must enter in model precision (fp32 sigma
  promotes the whole graph and crashes the bf16 linears — GPU-validated).

Trainer assumptions (A1/A2/A3 from the earlier plan) are OBSOLETE: training
uses the custom single-GPU loop in dpo_train.py.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import torch
import torch.nn.functional as F

CHOSEN, REJECTED = 0, 1
DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2"


def setup_offline_hf_cache() -> None:
    """Point HF at a private offline cache whose snapshot symlink resolves the
    WAN tokenizer to the LOCAL base-model dir (the shared cache lacks the
    snapshot). Must run before anything imports huggingface_hub."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    revision = "f50c09f5d8ab133a90cac3f4886a6471e9ba3f18"
    repo_root = Path(__file__).resolve().parents[2]
    base = repo_root.parents[1] / "checkpoints/Cosmos-Predict2-2B-Video2World"
    cache = Path("/tmp/tta_hf_cache/huggingface-hub")
    repo = cache / "models--nvidia--Cosmos-Predict2-2B-Video2World"
    snap = repo / "snapshots" / revision
    snap.mkdir(parents=True, exist_ok=True)
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    (repo / "refs" / "main").write_text(revision)
    link = snap / "tokenizer"
    if not link.exists():
        link.symlink_to(base / "tokenizer")
    os.environ["HF_HUB_CACHE"] = str(cache)


def dpo_loss_from_errors(e_chosen: torch.Tensor, e_rejected: torch.Tensor, beta: float):
    """Trajectory scores are plain sums over the flattened pair dimension.

    Returns (loss, margin); loss = log(2) iff e_chosen == e_rejected, and
    swapping the sides flips the margin sign.
    """
    margin = beta * (e_rejected.sum() - e_chosen.sum())
    return -F.logsigmoid(margin), margin


def share_noise_across_pairs(epsilon: torch.Tensor, sigma: torch.Tensor):
    """Copy each pair's CHOSEN-slot noise onto its REJECTED slot.

    draw_training_sigma_and_epsilon draws independent values for every batch
    item, so a [2B, ...] batch gives chosen/rejected DIFFERENT noise. DPO v1
    requires both sides of a pair to share one draw (margin-variance control):
    this overwrites the rejected slot (odd indices) with the chosen slot's
    (even indices) noise, in place.
    """
    n = epsilon.shape[0]
    if n % 2 != 0 or sigma.shape[0] != n:
        raise ValueError(f"epsilon/sigma must have an even leading dim: {epsilon.shape} / {sigma.shape}")
    eps = epsilon.view(n // 2, 2, *epsilon.shape[1:])
    eps[:, 1] = eps[:, 0]
    sig = sigma.view(n // 2, 2, *sigma.shape[1:])
    sig[:, 1] = sig[:, 0]
    return eps.view_as(epsilon), sig.view_as(sigma)


def load_policy_model(ckpt_path: str, experiment_name: str = "cosmos_predict2_2b_480p_libero__inference_only"):
    """Load the released policy through the predict2 wheel (same as eval).

    config_file 必须与 get_model() 一致(cosmos_policy/config/config.py):
    该 policy 的 experiment 只注册在那条配置链上
    (review finding 1, 2026-09-09 — 旧路径连 experiment 都找不到).
    """
    from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint

    model, _config = load_model_from_checkpoint(
        experiment_name=experiment_name,
        s3_checkpoint_dir=str(ckpt_path),
        config_file="cosmos_policy/config/config.py",
        load_ema_to_reg=False,
    )
    return model.to("cuda")


def attach_adapter(model, adapter_path: str) -> None:
    """Eval-side interface (review decision 2026-09-09, option 2): inject LoRA
    according to the saved metadata and load its weights, so a trained adapter
    can be exercised by the existing run_libero_eval flow."""
    blob = torch.load(adapter_path, map_location="cpu")
    meta = blob["meta"]
    model.add_lora(
        model.net,
        lora_rank=meta["lora_rank"],
        lora_alpha=meta["lora_alpha"],
        lora_target_modules=meta["target_modules"],
        init_lora_weights=True,
    )
    state = {k.replace("_checkpoint_wrapped_module.", ""): v
             for k, v in blob["lora_state"].items()}
    result = model.net.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            f"[attach_adapter] {len(result.unexpected_keys)} unexpected keys after "
            f"normalization, e.g. {result.unexpected_keys[:3]}")
    lora_missing = [k for k in result.missing_keys if "lora_" in k]
    if lora_missing:
        raise RuntimeError(f"[attach_adapter] {len(lora_missing)} lora keys missing after load")
    print(f"[attach_adapter] LoRA injected + loaded <- {adapter_path}")


class TTADPOModel:
    """Wraps the loaded policy; owns LoRA injection, the DPO forward, the
    adapter disable/enable toggle and adapter save/load."""

    def __init__(self, model, beta: float = 0.1, lora_rank: int = 8, lora_alpha: int = 16,
                 target_modules: str = DEFAULT_TARGET_MODULES, base_checkpoint: str = ""):
        self.model = model
        self.beta = beta
        self.base_checkpoint = base_checkpoint
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules
        self.model.add_lora(self.model.net, lora_rank=lora_rank, lora_alpha=lora_alpha,
                            lora_target_modules=target_modules, init_lora_weights=True)
        self._lora_names = {n for n, _ in self._iter_params() if "lora_" in n}
        if not self._lora_names:
            raise RuntimeError("LoRA injection produced no trainable parameters — refusing to train the base.")
        n_train, n_frozen = 0, 0
        for name, p in self._iter_params():
            p.requires_grad_(name in self._lora_names)
            if name in self._lora_names:
                n_train += p.numel()
            else:
                n_frozen += p.numel()
        print(f"[TTADPOModel] LoRA params: {n_train:,} trainable / {n_frozen:,} frozen")

    def _iter_params(self):
        return self.model.net.named_parameters()

    def lora_parameters(self):
        return [p for p in self.model.net.parameters() if p.requires_grad]

    @staticmethod
    def batch_to_device(batch: dict, device: str = "cuda") -> dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if v.dtype.is_floating_point and k != "video":
                    out[k] = v.to(device, dtype=torch.bfloat16)
                else:
                    out[k] = v.to(device)
            elif isinstance(v, (list, tuple)) and len(v) > 0 and \
                    all(isinstance(x, torch.Tensor) for x in v):
                # Per-side fields arrive as list-of-tensors after _stack_pair +
                # default_collate (one entry per pair side, each with a leading
                # batch dim). Stack to [B, 2, ...] so prepare_inputs' flatten
                # sees the same layout as the tensor fields. Without this the
                # conditioner/loss hit raw lists (found by the preflight probe,
                # 2026-09-13 — dpo_train would crash on its first batch).
                out[k] = torch.stack(list(v), dim=1).to(device)
            else:
                out[k] = v
        return out

    def prepare_inputs(self, batch: dict):
        """Flatten the pair batch, encode, and draw ONE shared-noise
        sigma/epsilon. Returns (flat, x0, condition, sigma, eps) so callers may
        reuse the exact draw across forwards (tests); real training draws fresh
        per step via dpo_forward."""
        flat = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[1] == 2:
                flat[k] = v.flatten(0, 1)  # [B, 2, ...] -> [2B, ...]
            else:
                flat[k] = v
        _, x0, condition = self.model.get_data_and_condition(flat)
        sigma, eps = self.model.draw_training_sigma_and_epsilon(x0.size(), condition)
        x0 = x0.to(**self.model.tensor_kwargs)
        eps, sigma = share_noise_across_pairs(
            eps.to(**self.model.tensor_kwargs), sigma.to(**self.model.tensor_kwargs)
        )
        return flat, x0, condition, sigma, eps

    def errors(self, batch: dict, prepared=None):
        """Policy and reference action-frame errors for a pair batch, computed
        with ONE shared sigma/epsilon draw (shared across each pair's two
        sides). Returns (e_policy, e_reference), each [2B]; the reference pass
        runs no_grad with adapters disabled. Pass `prepared` (from
        prepare_inputs) to reuse the same noise across calls (tests)."""
        flat, x0, condition, sigma, eps = prepared if prepared is not None else self.prepare_inputs(batch)
        common = dict(
            action_chunk=flat["actions"],
            action_indices=flat["action_latent_idx"],
            proprio=flat["proprio"],
            current_proprio_indices=flat["current_proprio_latent_idx"],
            future_proprio=flat["future_proprio"],
            future_proprio_indices=flat["future_proprio_latent_idx"],
            future_wrist_image_indices=flat["future_wrist_image_latent_idx"],
            future_wrist_image2_indices=None,
            future_image_indices=flat["future_image_latent_idx"],
            future_image2_indices=None,
            rollout_data_mask=flat["rollout_data_mask"],
            world_model_sample_mask=flat["world_model_sample_mask"],
            value_function_sample_mask=flat["value_function_sample_mask"],
            value_function_return=flat["value_function_return"],
            value_indices=flat["value_latent_idx"],
        )
        out, _, _, _ = self.model.compute_loss_with_epsilon_and_sigma(x0, condition, eps, sigma, **common)
        bidx = torch.arange(out["edm_loss_per_frame"].shape[0], device=x0.device)
        e_policy = out["edm_loss_per_frame"][bidx, flat["action_latent_idx"]].float()
        with torch.no_grad(), self.adapters_disabled():
            ref_out, _, _, _ = self.model.compute_loss_with_epsilon_and_sigma(x0, condition, eps, sigma, **common)
            e_reference = ref_out["edm_loss_per_frame"][bidx, flat["action_latent_idx"]].float()
        return e_policy, e_reference

    def dpo_forward(self, batch: dict, prepared=None):
        """Real-training entry: draws fresh noise each call unless `prepared`
        is given. Returns (loss, margin, per-item deltas e_policy - e_ref)."""
        e_policy, e_reference = self.errors(batch, prepared=prepared)
        delta_pairs = (e_policy - e_reference).view(-1, 2)
        loss, margin = dpo_loss_from_errors(delta_pairs[:, CHOSEN], delta_pairs[:, REJECTED], self.beta)
        return loss, margin, delta_pairs

    @contextlib.contextmanager
    def adapters_disabled(self):
        """Reference mode: disable every injected adapter, restore on exit
        (including exceptions). peft's inject_adapter_in_model leaves a plain
        module, so we drive the tuner layers directly."""
        from peft.tuners.tuners_utils import BaseTunerLayer

        layers = [(n, m) for n, m in self.model.net.named_modules() if isinstance(m, BaseTunerLayer)]
        if not layers:
            raise RuntimeError("no LoRA tuner layers found — reference would equal policy")
        saved = [(m, bool(m.disable_adapters)) for _n, m in layers]
        for _n, m in layers:
            m.enable_adapters(False)
        try:
            yield
        finally:
            for m, was_disabled in saved:
                if not was_disabled:
                    m.enable_adapters(True)

    @staticmethod
    def _canonical(name: str) -> str:
        """Strip gradient-checkpointing wrapper prefixes: training-time
        named_parameters carry them, a freshly-injected net does not. Adapter
        files store CANONICAL names; loaders normalize defensively."""
        return name.replace("_checkpoint_wrapped_module.", "")

    def save_adapter(self, path: str, extra_meta: dict | None = None) -> None:
        state = {self._canonical(n): p.detach().cpu()
                 for n, p in self._iter_params() if p.requires_grad}
        meta = {
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "base_checkpoint": self.base_checkpoint,
            "beta": self.beta,
            **(extra_meta or {}),
        }
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing adapter: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"meta": meta, "lora_state": state}, path)
        print(f"[TTADPOModel] adapter saved -> {path}")

    def load_adapter(self, path: str) -> dict:
        blob = torch.load(path, map_location="cpu")
        meta = blob.get("meta", {})
        for key, expected in (("lora_rank", self.lora_rank), ("lora_alpha", self.lora_alpha),
                              ("target_modules", self.target_modules)):
            if key in meta and meta[key] != expected:
                raise ValueError(f"adapter {path}: {key}={meta[key]!r} does not match model ({expected!r})")
        state = {self._canonical(k): v for k, v in blob["lora_state"].items()}
        result = self.model.net.load_state_dict(state, strict=False)
        lora_missing = [k for k in result.missing_keys if "lora_" in k]
        if result.unexpected_keys or lora_missing:
            raise RuntimeError(
                f"adapter {path}: unexpected={len(result.unexpected_keys)} lora_missing={len(lora_missing)}")
        print(f"[TTADPOModel] adapter loaded <- {path}")
        return meta

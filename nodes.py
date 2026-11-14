import os
import shutil
import urllib.parse
import comfy.utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import folder_paths
from safetensors.torch import load_file
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import logging
logger = logging.getLogger(__name__)

A_END, B_END, N_BLOCKS = 20, 40, 60


def _rope(q, k, pe):
    if pe is not None:
        from comfy.ldm.flux.math import apply_rope1
        q, k = apply_rope1(q, pe), apply_rope1(k, pe)
    return q, k


def _sdpa(q, k, v, mask=None):
    from comfy.ldm.modules.attention import (
        AttentionTensorContainer,
        optimized_attention,
        optimized_attention_masked,
    )

    # Get the head count BEFORE wrapping the tensors.
    heads = q.shape[1]

    # Let ComfyUI manage ownership and dispatch.
    q = AttentionTensorContainer(q)
    k = AttentionTensorContainer(k)
    v = AttentionTensorContainer(v)

    attention_fn = (
        optimized_attention
        if mask is None
        else optimized_attention_masked
    )

    return attention_fn(
        q,
        k,
        v,
        heads,
        mask=mask,
        skip_reshape=True,
        skip_output_reshape=True,
    )


def _segments(opts, n_img, mode):
    lengths = opts.get("reference_image_num_tokens")
    if not lengths:
        raise RuntimeError("OrionEdit requires Qwen ref_latents; reference_image_num_tokens is missing.")
    lengths = [int(x) for x in lengths]
    if any(x <= 0 for x in lengths):
        raise RuntimeError(f"Invalid reference token lengths: {lengths}")
    if mode == "edit":
        if len(lengths) != 2:
            raise RuntimeError(f"Edit mode requires [reference, source] token lengths; got {lengths}.")
        nr, ns = lengths
    else:
        nr, ns = sum(lengths), 0
    nz = n_img - nr - ns
    if nz <= 0:
        raise RuntimeError(f"Invalid stream sizes: image tokens={n_img}, reference/source={lengths}.")
    return slice(0, nz), slice(nz, nz + nr), slice(nz + nr, nz + nr + ns)


def _text_mask(x, text_mask, nt, total):
    if text_mask is None:
        return None
    m = torch.zeros((x.shape[0], 1, 1, total), device=x.device, dtype=x.dtype)
    m[..., :nt] = text_mask.reshape(x.shape[0], 1, 1, nt).to(device=x.device, dtype=x.dtype)
    return m


def _flow_mask(base, batch, total, nt, syn, ref, src, device, dtype, hard, beta):
    # A batch-independent flow mask can broadcast across CFG batches.
    # Keep a batch dimension only when the text mask itself is batch-specific.
    m = torch.zeros((1, 1, total, total), device=device, dtype=dtype) if base is None else base.expand(batch, 1, total, total).clone()
    syn, ref, src = (slice(nt + s.start, nt + s.stop) for s in (syn, ref, src))
    txt = slice(0, nt)
    blocked = torch.finfo(dtype).min
    def block(q, k):
        if q.start == q.stop or k.start == k.stop:
            return
        if hard:
            m[:, :, q, k] = blocked
        else:
            m[:, :, q, k] -= beta
    block(ref, txt); block(ref, syn)
    if src.stop > src.start:
        block(ref, src)
    block(src, txt); block(src, syn)
    return m


class _Projector(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.A1 = nn.Parameter(torch.empty(d, r), requires_grad=False)
        self.A2 = nn.Parameter(torch.empty(d, r), requires_grad=False)
        self.A3 = nn.Parameter(torch.empty(d, r), requires_grad=False)
        self.B1 = nn.Parameter(torch.zeros(r, r))
        self.B2 = nn.Parameter(torch.zeros(r, r))
        self.B3 = nn.Parameter(torch.zeros(r, r))
        self.alpha = 1.0

    @staticmethod
    def apply(x, a, b, alpha):
        a = a.to(device=x.device, dtype=x.dtype)
        b = b.to(device=x.device, dtype=x.dtype)
        return x + alpha * ((x @ a) @ b @ a.T)


class _StageC(nn.Module):
    def __init__(self, d, rank):
        super().__init__()
        self.proj = _Projector(d, rank)
        self.synthesis_fusion_logits = nn.Parameter(torch.zeros(3))

    def forward(self, attn, x, text, text_mask, pe, opts, mode, q, k, v, tv):
        if next(self.parameters()).device != x.device or next(self.parameters()).dtype != x.dtype:
            self.to(device=x.device, dtype=x.dtype)
        b, ni, _ = x.shape
        nt, h = text.shape[1], attn.heads
        total = nt + ni
        base_mask = _text_mask(x, text_mask, nt, total)
        base = _sdpa(q, k, v, base_mask).transpose(1, 2).reshape(b, total, -1).to(q.dtype)
        # Clone the small text result so the full joint output can be released
        # before the second (branch) attention operation.
        txt_out = base[:, :nt].clone()
        img_out = base[:, nt:].clone()
        del base
        syn, ref, src = _segments(opts, ni, mode)
        qm = q[:, :, nt:].transpose(1, 2).reshape(b, ni, -1)
        km = k[:, :, nt:].transpose(1, 2).reshape(b, ni, -1)
        vm = v[:, :, nt:].transpose(1, 2).reshape(b, ni, -1)
        tkm = k[:, :, :nt].transpose(1, 2).reshape(b, nt, -1)
        tvm = tv.transpose(1, 2).reshape(b, nt, -1)
        p = self.proj
        qz, kz, vz = qm[:, syn], km[:, syn], vm[:, syn]
        kz1 = p.apply(kz, p.A1, p.B1, p.alpha); vz1 = p.apply(vz, p.A1, p.B1, p.alpha)
        kr = p.apply(km[:, ref], p.A2, p.B2, p.alpha)
        vr = p.apply(vm[:, ref], p.A2, p.B2, p.alpha)
        qr = p.apply(qz, p.A2, p.B2, p.alpha)
        if src.stop > src.start:
            ks = p.apply(km[:, src], p.A3, p.B3, p.alpha)
            vs = p.apply(vm[:, src], p.A3, p.B3, p.alpha)
            qs = p.apply(qz, p.A3, p.B3, p.alpha)
            bk, bv, bq = torch.cat((tkm, kz1, kr, ks), 1), torch.cat((tvm, vz1, vr, vs), 1), torch.cat((qr, qs), 1)
        else:
            qa = p.apply(qz, p.A1, p.B1, p.alpha)
            bk, bv, bq = torch.cat((tkm, kz1, kr), 1), torch.cat((tvm, vz1, vr), 1), torch.cat((qr, qa), 1)

        # These flattened/projected intermediates are no longer needed once
        # branch Q/K/V have been assembled.
        del qm, km, vm, tkm, tvm, qz, kz, vz, kz1, vz1, kr, vr, qr
        if src.stop > src.start:
            del ks, vs, qs
        else:
            del qa

        bq = bq.view(b, bq.shape[1], h, -1).transpose(1, 2).contiguous()
        bk = bk.view(b, bk.shape[1], h, -1).transpose(1, 2).contiguous()
        bv = bv.view(b, bv.shape[1], h, -1).transpose(1, 2).contiguous()
        branch = _sdpa(bq, bk, bv).transpose(1, 2).reshape(b, bq.shape[2], -1).to(bq.dtype)
        nz = syn.stop - syn.start
        w = torch.softmax(self.synthesis_fusion_logits, dim=0)
        img_out[:, syn] = w[0] * img_out[:, syn] + w[1] * branch[:, :nz] + w[2] * branch[:, nz:2*nz]
        img_out = attn.to_out[0](img_out)
        if len(attn.to_out) > 1:
            img_out = attn.to_out[1](img_out)
        return img_out, attn.to_add_out(txt_out)


def _attention(processor, attn, x, text, text_mask, pe, opts, mode, index):
    if text is None:
        raise RuntimeError("OrionEdit requires the Qwen text stream.")
    b, ni, _ = x.shape; nt = text.shape[1]; h = attn.heads
    iq = attn.to_q(x).view(b, ni, h, -1).transpose(1, 2).contiguous()
    ik = attn.to_k(x).view(b, ni, h, -1).transpose(1, 2).contiguous()
    iv = attn.to_v(x).view(b, ni, h, -1).transpose(1, 2)
    tq = attn.add_q_proj(text).view(b, nt, h, -1).transpose(1, 2).contiguous()
    tk = attn.add_k_proj(text).view(b, nt, h, -1).transpose(1, 2).contiguous()
    tv = attn.add_v_proj(text).view(b, nt, h, -1).transpose(1, 2)
    iq, ik, tq, tk = attn.norm_q(iq), attn.norm_k(ik), attn.norm_added_q(tq), attn.norm_added_k(tk)
    q, k, v = torch.cat((tq, iq), 2), torch.cat((tk, ik), 2), torch.cat((tv, iv), 2)
    # The concatenated tensors own their storage; release the original image
    # projections and the text Q/K projections. Stage C only needs text V.
    del iq, ik, iv, tq, tk
    from comfy.ldm.flux.math import apply_rope1
    if pe is not None:
        q, k = apply_rope1(q, pe), apply_rope1(k, pe)
    if index >= B_END:
        return processor(attn, x, text, text_mask, pe, opts, mode, q, k, v, tv)
    syn, ref, src = _segments(opts, ni, mode)
    base = _text_mask(x, text_mask, nt, nt + ni)
    if index < A_END:
        mask = _flow_mask(base, b, nt + ni, nt, syn, ref, src, x.device, x.dtype, True, 0.0)
    else:
        beta = 8.0 - 6.0 * ((index - A_END) / (B_END - A_END - 1))
        mask = _flow_mask(base, b, nt + ni, nt, syn, ref, src, x.device, x.dtype, False, beta)
    out = _sdpa(q, k, v, mask).transpose(1, 2).reshape(b, nt + ni, -1).to(q.dtype)
    txt_out, img_out = out[:, :nt], out[:, nt:]
    img_out = attn.to_out[0](img_out)
    if len(attn.to_out) > 1:
        img_out = attn.to_out[1](img_out)
    return img_out, attn.to_add_out(txt_out)


class _BlockPatch:
    def __init__(self, block, processor, mode, index, previous=None):
        self.block, self.processor, self.mode, self.index, self.previous = block, processor, mode, index, previous

    def __call__(self, args, extra):
        attn = self.block.attn
        original = attn.forward
        def patched_forward(hidden_states, encoder_hidden_states=None, encoder_hidden_states_mask=None,
                            attention_mask=None, image_rotary_emb=None, transformer_options=None, **kwargs):
            return _attention(self.processor, attn, hidden_states, encoder_hidden_states,
                              encoder_hidden_states_mask, image_rotary_emb, transformer_options or {},
                              self.mode, self.index)
        attn.forward = patched_forward
        try:
            return self.previous(args, extra) if self.previous is not None else extra["original_block"](args)
        finally:
            attn.forward = original

    def to(self, device_or_dtype):
        if isinstance(self.processor, nn.Module):
            self.processor.to(device_or_dtype)
        if self.previous is not None and hasattr(self.previous, "to"):
            self.previous.to(device_or_dtype)
        return self

    def cleanup(self):
        # ComfyUI calls cleanup when model patches are being released/offloaded.
        if isinstance(self.processor, nn.Module):
            self.processor.to(device="cpu")
        if self.previous is not None and hasattr(self.previous, "cleanup"):
            self.previous.cleanup()

    def models(self):
        return self.previous.models() if self.previous is not None and hasattr(self.previous, "models") else []


class ApplyOrionEditAttention(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ApplyOrionEditAttention",
            display_name="Apply OrionEdit Attention",
            category="OrionEdit",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("mode", options=["edit", "fusion"], default="edit"),
                io.String.Input("processor_weights", default="orionedit/pytorch_processor_weights.safetensors"),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, model, mode, processor_weights) -> io.NodeOutput:
        if not hasattr(model, "clone") or not hasattr(model, "set_model_patch_replace"):
             raise TypeError("Expected a ComfyUI MODEL patcher input.")
        raw_path = processor_weights.strip()
        if not raw_path:
            raise ValueError("Please provide the OrionEdit processor weights path.")

        candidates = [raw_path] if os.path.isabs(raw_path) else [
            os.path.join(folder_paths.models_dir, raw_path),
            os.path.join(folder_paths.models_dir, "model_patches", raw_path),
            os.path.join(os.path.dirname(folder_paths.models_dir), raw_path),
        ]
        if not os.path.isabs(raw_path):
            try:
                registered_path = folder_paths.get_full_path("model_patches", raw_path)
                if registered_path:
                    candidates.insert(0, registered_path)
            except (AttributeError, KeyError, TypeError):
                pass

        path = next((p for p in candidates if os.path.isfile(p)), None)
        if path is None:
            searched = "\\n  ".join(candidates)
            raise FileNotFoundError(
                f"OrionEdit processor weights not found. Searched:\\n  {searched}"
            )
        state = load_file(path, device="cpu")
        patched = model.clone()
        diffusion_model = patched.get_model_object("diffusion_model")
        blocks = getattr(diffusion_model, "transformer_blocks", None)
        if blocks is None or len(blocks) != N_BLOCKS:
            raise RuntimeError(f"Expected the 60-block Qwen Image Edit transformer; found {len(blocks) if blocks is not None else 0} blocks.")
        replacements = patched.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
        for i, block in enumerate(blocks):
            attn = getattr(block, "attn", None)
            if attn is None or not hasattr(attn, "to_q") or not hasattr(attn, "add_q_proj"):
                raise RuntimeError(f"Block {i} is not the expected Qwen double-stream attention block.")
            processor = None
            if i >= B_END:
                prefix = f"transformer.transformer_blocks.{i}.attn.processor."
                sd = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
                required = {"proj.A1", "proj.A2", "proj.A3", "proj.B1", "proj.B2", "proj.B3", "synthesis_fusion_logits"}
                missing = required - set(sd)
                if missing:
                    raise RuntimeError(f"Missing Stage C weights for block {i}: {sorted(missing)}")
                rank, d = int(sd["proj.B1"].shape[0]), int(sd["proj.A1"].shape[0])
                expected_shapes = {
                    "proj.A1": (d, rank), "proj.A2": (d, rank), "proj.A3": (d, rank),
                    "proj.B1": (rank, rank), "proj.B2": (rank, rank), "proj.B3": (rank, rank),
                    "synthesis_fusion_logits": (3,),
                }
                bad_shapes = {
                    name: (tuple(sd[name].shape), shape)
                    for name, shape in expected_shapes.items()
                    if tuple(sd[name].shape) != shape
                }
                if bad_shapes:
                    raise RuntimeError(f"Invalid Stage C tensor shapes in block {i}: {bad_shapes}")
                model_d = int(getattr(attn, "inner_dim", attn.heads * attn.dim_head))
                if d != model_d:
                    raise RuntimeError(f"Stage C dimension mismatch at block {i}: checkpoint {d}, model {model_d}.")
                processor = _StageC(d, rank)
                processor.load_state_dict(sd, strict=True)
            key = ("double_block", i)
            previous = replacements.get(key)
            if isinstance(previous, _BlockPatch):
                raise RuntimeError("OrionEdit attention is already applied to this model.")
            patched.set_model_patch_replace(_BlockPatch(block, processor, mode, i, previous), "dit", "double_block", i)
        return io.NodeOutput(patched)

####################################################
# 2. OrionEditImagePrep node
####################################################
import math
from typing import List, Optional
import numpy as np
from PIL import Image
import comfy.model_management


def calculate_dimensions(target_area: int, ratio: float):
    """Aligns dimensions to 1MP area and strictly rounds to multiples of 32 (from pipeline_orion_edit.py)."""
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = max(32, round(width / 32) * 32)
    height = max(32, round(height / 32) * 32)
    return width, height


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim == 4:
        tensor = tensor[0]
    np_img = (tensor.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(np_img).convert("RGB")


def _pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    np_img = np.array(pil_img).astype(np.float32) / 255.0
    return torch.from_numpy(np_img).unsqueeze(0)


def _build_composited_reference_canvas(
    refs: List[Image.Image], source: Optional[Image.Image]
) -> Image.Image:
    """1:1 implementation of OrionEdit reference canvas compositor."""
    if len(refs) == 1:
        return refs[0]

    strip_w = sum(r.width for r in refs)
    strip_h = max(r.height for r in refs)
    strip = Image.new("RGB", (strip_w, strip_h), (0, 0, 0))
    x = 0
    for ref in refs:
        strip.paste(ref, (x, 0))
        x += ref.width

    if source is not None:
        canvas_w, canvas_h = source.size
    else:
        canvas_w, canvas_h = refs[0].size

    scale = min(canvas_w / strip_w, canvas_h / strip_h)
    new_w = max(1, int(round(strip_w * scale)))
    new_h = max(1, int(round(strip_h * scale)))
    resized_strip = strip.resize((new_w, new_h), Image.BICUBIC)

    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    paste_x = (canvas_w - new_w) // 2
    paste_y = (canvas_h - new_h) // 2
    canvas.paste(resized_strip, (paste_x, paste_y))
    return canvas


class OrionEditImagePrep(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OrionEditImagePrep",
            display_name="OrionEdit Image Prep",
            category="OrionEdit",
            inputs=[
                io.Image.Input("reference1"),
                io.Image.Input("reference2", optional=True),
                io.Image.Input("reference3", optional=True),
                io.Image.Input("source_image", optional=True),
                io.Int.Input("target_area", default=1024 * 1024, min=262144, max=4194304, step=65536, optional=True),
                io.Int.Input("batch_size", default=1, min=1, max=64, optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="reference_image"),
                io.Image.Output(display_name="source_image"),
                io.Latent.Output(display_name="empty_latent"),
            ],
        )
    
    @classmethod
    def execute(
        cls,
        reference1: torch.Tensor,
        reference2: Optional[torch.Tensor] = None,
        reference3: Optional[torch.Tensor] = None,
        source_image: Optional[torch.Tensor] = None,
        target_area: int = 1024 * 1024,
        batch_size: int = 1,
    ) -> io.NodeOutput:
        refs_tensors = [reference1]
        if reference2 is not None:
            refs_tensors.append(reference2)
        if reference3 is not None:
            refs_tensors.append(reference3)

        refs_pil = [_tensor_to_pil(r) for r in refs_tensors]
        has_source = source_image is not None

        if has_source:
            # --- Editing Mode ---
            source_pil = _tensor_to_pil(source_image)
            composed_ref_pil = _build_composited_reference_canvas(refs_pil, source_pil)
            final_ref_pil = composed_ref_pil
            final_src_pil = source_pil
            # Output canvas size derived from source image
            canvas_w, canvas_h = calculate_dimensions(target_area, source_pil.width / source_pil.height)
        else:
            # --- Fusion Mode ---
            if len(refs_pil) < 2:
                raise ValueError(
                    "Fusion mode requires at least two reference images (reference1 and reference2) when source_image is omitted."
                )
            final_ref_pil = refs_pil[0]
            final_src_pil = refs_pil[1]
            # Output canvas size derived from ref2 (image[-1] in OrionEdit)
            canvas_w, canvas_h = calculate_dimensions(target_area, final_src_pil.width / final_src_pil.height)

        # Pre-align each image to multiple-of-32 so downstream TextEncodeQwenImageEditPlus
        # maintains integer 2x2 DiT patch divisibility without odd-dimension artifacts
        ref_w, ref_h = calculate_dimensions(target_area, final_ref_pil.width / final_ref_pil.height)
        src_w, src_h = calculate_dimensions(target_area, final_src_pil.width / final_src_pil.height)

        out_ref_tensor = _pil_to_tensor(final_ref_pil.resize((ref_w, ref_h), Image.BICUBIC))
        out_src_tensor = _pil_to_tensor(final_src_pil.resize((src_w, src_h), Image.BICUBIC))

        # Generate zeroed latent: 16 channels, downsampled by 8
        latent_channels = 16
        empty_latent = torch.zeros(
            [batch_size, latent_channels, canvas_h // 8, canvas_w // 8],
            device=comfy.model_management.intermediate_device(),
        )

        return io.NodeOutput(out_ref_tensor, out_src_tensor, {"samples": empty_latent})

########################################################
# ShellCommandNode
########################################################
import subprocess
import sys
import time

class ShellCommandNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ShellCommandRunner",
            display_name="Shell Command Runner",
            category="utils",
            description="Execute shell commands on the host system and log output.",
            is_output_node=True,
            not_idempotent=True,
            inputs=[
                io.String.Input(
                    "command",
                    default="",
                    multiline=True,
                    tooltip="Shell command(s) to execute.",
                ),
                io.String.Input(
                    "cwd",
                    default="",
                    multiline=False,
                    optional=True,
                    tooltip="Working directory (defaults to custom_nodes when empty).",
                ),
            ],
            outputs=[],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> int:
        return time.monotonic_ns()

    @classmethod
    def execute(cls, command: str, cwd: str = "") -> io.NodeOutput:
        cmd = (command or "").strip()
        if not cmd:
            raise ValueError("Command cannot be empty.")

        work_dir = (cwd or "").strip() or folder_paths.get_folder_paths("custom_nodes")[0]
        if not os.path.isdir(work_dir):
            raise FileNotFoundError(f"Working directory does not exist: {work_dir}")

        env = os.environ.copy()
        py_bin = os.path.dirname(sys.executable)
        env["PATH"] = f"{py_bin}{os.pathsep}{env.get('PATH', '')}"

        logger.info("Executing command in %s: %s", work_dir, cmd)
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=work_dir,
            env=env,
            capture_output=True,
            text=True,
        )

        if result.stdout:
            logger.info("[stdout]\n%s", result.stdout.rstrip())
        if result.stderr:
            logger.warning("[stderr]\n%s", result.stderr.rstrip())
        logger.info("Command exited with code %d", result.returncode)

        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed (exit {result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )

        return io.NodeOutput()

# =====================================================================
# 3. RestartSignal
# =====================================================================
class RestartSignal(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RestartSignal",
            category="utils",
            is_output_node=True,
            inputs=[],
            outputs=[],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        flag = "/tmp/comfyui_restart.request"
        with open(flag, "w") as f:
            f.write("restart")

        logger.warning("[RestartSignal] Restart requested.")
        return io.NodeOutput()

# =====================================================================
# 4. DataInspector log
# =====================================================================
"""Data Inspector: a pass-through node that logs details of whatever flows through it."""
MAX_VALUE_ELEMS = 32  # larger values are not logged; also caps listed container items
MAX_DEPTH = 2         # container levels expanded below the input

_DEVICE_ATTRS = ("device", "load_device", "offload_device", "current_device")
_WRAPPED_ATTRS = ("model", "patcher")  # one wrapper level, e.g. ModelPatcher.model, CLIP.patcher
_SCALARS = (bool, int, float, complex, type(None))


def _get(obj, name):
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


def _device_info(obj):
    """Best-effort device details for objects that are not tensors or plain containers."""
    info = []
    targets = [("", obj)] + [(f"{n}.", _get(obj, n)) for n in _WRAPPED_ATTRS]
    for prefix, target in targets:
        if target is None:
            continue
        for attr in _DEVICE_ATTRS:
            value = _get(target, attr)
            if value is not None and not callable(value):
                info.append(f"{prefix}{attr}={value}")
        if isinstance(target, torch.nn.Module):
            devices, params, size = set(), 0, 0
            for p in target.parameters():
                devices.add(str(p.device))
                params += p.numel()
                size += p.numel() * p.element_size()
            info.append(f"{prefix}param_devices={sorted(devices)} params={params} param_bytes={size}")
    return info


def _inspect(obj):
    """Return (one-line summary, [(child name, child), ...])."""
    parts = [f"type={type(obj).__name__}"]
    children = []
    count = value = None  # count: number of elements; value: zero-arg callable

    if isinstance(obj, torch.Tensor):
        count = obj.numel()
        parts += [f"shape={tuple(obj.shape)}", f"dtype={obj.dtype}", f"device={obj.device}",
                  f"numel={count}", f"bytes={count * obj.element_size()}"]
        value = lambda: obj.detach().cpu().tolist()
    elif isinstance(obj, np.ndarray):
        count = obj.size
        parts += [f"shape={obj.shape}", f"dtype={obj.dtype}", f"numel={count}", f"bytes={obj.nbytes}"]
        value = obj.tolist
    elif isinstance(obj, (str, bytes)):
        count = len(obj)
        parts.append(f"len={count}")
        value = lambda: obj
    elif isinstance(obj, _SCALARS):
        count, value = 1, lambda: obj
    elif isinstance(obj, (dict, list, tuple)):
        parts.append(f"len={len(obj)}")
        if len(obj) > MAX_VALUE_ELEMS:
            parts.append(f"showing_first={MAX_VALUE_ELEMS}")
        items = obj.items() if isinstance(obj, dict) else enumerate(obj)
        children = [(f"[{k!r}]", v) for k, v in itertools.islice(items, MAX_VALUE_ELEMS)]
    else:
        parts.append(f"sizeof={sys.getsizeof(obj)}")
        parts += _device_info(obj)

    if count is not None:
        parts.append(f"value={value()!r}" if count <= MAX_VALUE_ELEMS
                     else f"value=<omitted: {count} elements>")
    return " ".join(parts), children


def _describe(obj, name, depth=0):
    pad = "  " * depth
    try:
        summary, children = _inspect(obj)
    except Exception as exc:  # logging must never break the graph
        summary, children = f"type={type(obj).__name__} <inspect failed: {exc!r}>", []
    logger.info("[DataInspector] %s%s: %s", pad, name, summary)
    if children and depth >= MAX_DEPTH:
        logger.info("[DataInspector] %s  ...<max depth %d reached>", pad, MAX_DEPTH)
        return
    for child_name, child in children:
        _describe(child, child_name, depth + 1)


class DataInspector(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DataInspector",
            category="utils",
            is_output_node=True,
            inputs=[io.AnyType.Input("data")],
            outputs=[io.AnyType.Output(display_name="data")],
        )

    @classmethod
    def execute(cls, data) -> io.NodeOutput:
        _describe(data, "data")
        return io.NodeOutput(data)


# =====================================================================
# 5. Model Downloader
# =====================================================================
class ModelDownloaderNode(io.ComfyNode):
    """
    High-speed model downloader for ComfyUI leveraging aria2c.
    Supports existing categories or creating new first-level directories in ComfyUI/models/.
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        # Query ComfyUI's registry and add [models_root] to allow 1st-level folder creation
        registered_categories = sorted(list(folder_paths.folder_names_and_paths.keys()))
        available_categories = ["[models_root]"] + registered_categories
        default_category = "checkpoints" if "checkpoints" in registered_categories else "[models_root]"

        return io.Schema(
            node_id="ModelDownloaderNode",
            category="utils",
            is_output_node=True,
            inputs=[
                io.String.Input("url", default="", multiline=False),
                io.Combo.Input("target_category", options=available_categories, default=default_category),
                io.Boolean.Input("overwrite", default=False),
                io.String.Input("subfolder", default="", multiline=False, optional=True),
                io.String.Input("filename", default="", multiline=False, optional=True),
                io.String.Input("auth_token", default="", multiline=False, optional=True),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, url: str, target_category: str, overwrite: bool, subfolder: str = "", filename: str = "", auth_token: str = "") -> io.NodeOutput:
        url = url.strip()
        if not url:
            logger.info("[ModelDownloader] Empty URL provided. Skipping.")
            return io.NodeOutput()

        # 1. Platform Binary Verification
        if not shutil.which("aria2c"):
            raise RuntimeError(
                "[ModelDownloader] 'aria2c' binary not found in system PATH. "
                "Please install it in your cloud container (e.g., 'apt-get update && apt-get install -y aria2')."
            )

        # 2. Strict ComfyUI Path Resolution
        if target_category == "[models_root]":
            # Target ComfyUI/models/ directly
            base_dir = os.path.abspath(folder_paths.models_dir)
        else:
            base_dirs = folder_paths.get_folder_paths(target_category)
            if not base_dirs:
                raise ValueError(f"[ModelDownloader] No paths registered for category '{target_category}'")
            base_dir = os.path.abspath(base_dirs[0])

        clean_subfolder = os.path.normpath(subfolder.strip()).lstrip("/\\")

        if clean_subfolder in ("", "."):
            target_dir = base_dir
        else:
            target_dir = os.path.abspath(os.path.join(base_dir, clean_subfolder))

        # Enforce canonical path containment (prevent directory traversal)
        if os.path.commonpath([base_dir, target_dir]) != base_dir:
            raise ValueError(f"[ModelDownloader] Security violation: Destination '{target_dir}' escapes '{base_dir}'")

        # Automatically create the first-level or nested subfolder
        os.makedirs(target_dir, exist_ok=True)

        # If a new first-level directory is created under models/, register it with ComfyUI
        if target_category == "[models_root]" and clean_subfolder and hasattr(folder_paths, "add_model_folder_path"):
            folder_paths.add_model_folder_path(clean_subfolder, target_dir)

        # 3. Filename & Source URL Normalization
        if "huggingface.co" in url:
            url = url.replace("/blob/", "/resolve/")

        # Incur filename: User Override -> URL Path Basename -> Fallback to Content-Disposition
        target_filename = filename.strip()
        if not target_filename:
            parsed_path = urllib.parse.urlsplit(url).path.rstrip("/")
            extracted_name = os.path.basename(parsed_path)
            if extracted_name and "." in extracted_name:
                target_filename = urllib.parse.unquote(extracted_name)

        # Skip download if the final target file already exists
        if target_filename:
            destination_file = os.path.join(target_dir, target_filename)
            if os.path.exists(destination_file) and not overwrite:
                logger.info(f"[ModelDownloader] File already exists, skipping: {destination_file}")
                if hasattr(folder_paths, "filename_list_cache"):
                    folder_paths.filename_list_cache.clear()
                return io.NodeOutput()

        # 4. Assemble aria2c Command
        cmd = [
            "aria2c",
            "--console-log-level=warn",
            "--summary-interval=1",
            "--content-disposition=true",
            "-x", "16",
            "-s", "16",
            "-k", "1M",
            "-c",
            f"--dir={target_dir}",
        ]

        if target_filename:
            cmd.append(f"--out={target_filename}")

        if overwrite:
            cmd.append("--allow-overwrite=true")
        else:
            cmd.append("--auto-file-renaming=false")

        token = auth_token.strip()
        if token:
            cmd.append(f"--header=Authorization: Bearer {token}")

        cmd.append(url)

        # 5. Process Execution & Progress Reporting via Native ComfyUI Contract
        logger.info(f"[ModelDownloader] Initiating transfer for: {url}")
        pbar = comfy.utils.ProgressBar(100)

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        progress_regex = re.compile(r'\((\s*\d+)%\)')

        for line in process.stdout:
            match = progress_regex.search(line)
            if match:
                percent = int(match.group(1))
                pbar.update_absolute(percent, 100)

        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"[ModelDownloader] aria2c transfer failed with exit code {process.returncode}")

        pbar.update_absolute(100, 100)

        # 6. Invalidate ComfyUI Cache
        if hasattr(folder_paths, "filename_list_cache"):
            folder_paths.filename_list_cache.clear()

        logger.info(f"[ModelDownloader] Download completed successfully to {target_dir}")
        return io.NodeOutput()

class OrionEditExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ApplyOrionEditAttention,
            OrionEditImagePrep,
            RestartSignal,
            DataInspector,
            ModelDownloaderNode,
            ShellCommandNode,
        ]

async def comfy_entrypoint() -> OrionEditExtension:
    return OrionEditExtension()
What's included
- Stage A: hard reverse-causal attention mask.
- Stage B: soft mask with the official 8.0 → 2.0 schedule.
- Stage C: Tri-Orthogonal projection and synthesis fusion.
- uses one Apply OrionEdit Attention node, derives the stream boundaries from ComfyUI’s reference_image_num_tokens, applies the 20/20/20 Orion stages, and keeps the existing INT8 Q/K/V and output projections. It also requires the separate OrionEdit processor-weights file; the LoRA alone does not contain the Stage C projector weights. This follows the supplied OrionEdit checkpoint-loading code.

Workflow
- Load Model → Load LoRA → Apply OrionEdit Attention → KSampler
- Connect positive, negative, and latent inputs to KSampler normally.

Important implementation notes
- Uses ComfyUI's native set_model_patch_replace mechanism. 
- Derives stream boundaries from Qwen's existing reference_image_num_tokens metadata. model.py
- Preserves the existing INT8 projection modules.
- Implements Stage A, B, and C using the supplied OrionEdit processor logic. transformer_orion.py
- No separate preparation node.
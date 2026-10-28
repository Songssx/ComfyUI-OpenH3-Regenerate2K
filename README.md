# ComfyUI Open-H3 Regenerate 2K

[中文说明](#中文说明) · [English](#english)

> [!IMPORTANT]
> This is an **early, community-maintained ComfyUI adaptation** of Open-H3-Regenerate-2K. It is not an official NVIDIA, Efficient-Large-Model, MiniMax, or ComfyUI implementation. The current integration is preliminary; model compatibility, output quality, memory use, and high-motion stability may change as upstream projects evolve.

## 中文说明

把 [Open-H3-Regenerate-2K](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K) 的合并 LoRA、50 个 LMS Sharp AdaLN 曲线分支和官方 2/3 步 sigma 接入原生 ComfyUI MiniMax H3。

插件复用 ComfyUI 已有的 H3 主模型、文本编码器、视频/音频 VAE，以及社区发布的 H3 3D latent upscaler，不会自动下载模型。

### 当前状态

- 初步支持 Open-H3 官方 3 步与 2 步 Euler sigma。
- 将 312 个 Diffusers LoRA 目标映射到 ComfyUI 的 fused QKV H3 结构。
- 加载 50 个 Sharp curve branches，并严格校验基模的 `1025 × 8` AdaLN curve table。
- 精修时可锁定音频 latent，只重新生成视频 latent。
- 已在 ComfyUI `0.41.0`、Windows、RTX 5090 32GB 上完成一次 1344×768 → 2560×1440 端到端测试。
- 高动态画面仍可能出现融化、闪烁或局部结构不稳定；这是当前初步适配的已知限制。

### 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Songssx/ComfyUI-OpenH3-Regenerate2K.git
```

示例工作流还使用以下节点：

- [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)（必须）
- [kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)（视频内嵌示例中的预览节点）

安装后重启 ComfyUI。

### 模型目录

请按下面的目录放置模型。文件名需要与示例工作流一致。

```text
ComfyUI/
└── models/
    ├── diffusion_models/
    │   └── minimax_h3_ref2va_pruned_int8_convrot.safetensors
    ├── text_encoders/
    │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
    ├── vae/
    │   ├── minimax_h3_video_vae_fp16.safetensors
    │   └── minimax_h3_audio_vae_fp32.safetensors
    ├── vae_approx/
    │   └── taeh3.safetensors
    ├── latent_upscale_models/
    │   └── minimax_h3_latent_upscaler_3d_bf16.safetensors
    ├── loras/
    │   ├── open_h3_regenerate_2k_lora.safetensors
    │   ├── dmad_minimax_h3_4step_full_critic_comfyui.safetensors
    │   └── minimax_h3_singularity_v2_lora_avg_rank_351_bf16.safetensors
    └── open_h3_regenerate/
        └── lora/
            └── sharp_curve_branches.safetensors
```

### 模型下载

#### `ComfyUI/models/diffusion_models/`

- [`minimax_h3_ref2va_pruned_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors)

> Open-H3 curve branches 依赖这个 Ref2VA pruned 基模中完全一致的 `1025 × 8` AdaLN curve table。插件会做内容哈希校验，使用不兼容基模时会明确报错。

#### `ComfyUI/models/text_encoders/`

- [`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors)

#### `ComfyUI/models/vae/`

- [`minimax_h3_video_vae_fp16.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors)
- [`minimax_h3_audio_vae_fp32.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors)

#### `ComfyUI/models/vae_approx/`

- [`taeh3.safetensors`](https://huggingface.co/Kijai/MiniMax-H3-TAE/resolve/main/vae_approx/taeh3.safetensors)（仅用于 KJNodes 生成预览，不影响最终 VAE 解码）

#### `ComfyUI/models/latent_upscale_models/`

- [`minimax_h3_latent_upscaler_3d_bf16.safetensors`](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/minimax_h3_latent_upscaler_3d_bf16.safetensors)

Open-H3 官方仓库也提供同一权重，文件名为 [`minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K/resolve/main/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors)。如下载该版本，请在工作流节点中选择它，或重命名为示例所用文件名。

#### `ComfyUI/models/loras/`

- [`open_h3_regenerate_2k_lora.safetensors`](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K/resolve/main/lora/open_h3_regenerate_2k_lora.safetensors)（插件核心模型）
- [`dmad_minimax_h3_4step_full_critic_comfyui.safetensors`](https://huggingface.co/ZhengmingYu/DMAD/resolve/main/minimax_h3/dmad_minimax_h3_4step_full_critic_comfyui.safetensors)（视频示例的第一阶段 LoRA）
- [`minimax_h3_singularity_v2_lora_avg_rank_351_bf16.safetensors`](https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/loras/minimax_h3_singularity_v2_lora_avg_rank_351_bf16.safetensors)（视频示例使用）

#### `ComfyUI/models/open_h3_regenerate/lora/`

- [`sharp_curve_branches.safetensors`](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K/resolve/main/lora/sharp_curve_branches.safetensors)

Sharp curve branches 不是普通 LoRA，不能直接放进 `models/loras` 后用标准 LoRA Loader 加载；请保持上面的专用目录。

### 节点

- **Open-H3 Regenerate 2K Model Patch**：加载主 LoRA 与 50 个 Sharp curve branches。
- **Open-H3 Regenerate Exact Sigmas**：输出官方 3 步或 2 步 sigma。
- **Open-H3 Lock Audio for Refine**：精修阶段仅对视频 latent 加噪，音频保持在 sigma 0。

### 示例工作流

- [`example_workflows/Open_H3_Regenerate_2K_video_example.json`](example_workflows/Open_H3_Regenerate_2K_video_example.json)：从用户提供的 `Open_H3_Regenerate_2K_00009_.mp4` 内嵌元数据中原样提取的工作流。
- [`example_workflows/Open_H3_Regenerate_2K_local.json`](example_workflows/Open_H3_Regenerate_2K_local.json)：较精简的本地测试工作流。

视频示例包含额外的 DMAD、Singularity LoRA 和 KJNodes 预览节点；如果只测试 Open-H3 二阶段，可以删除或替换这些第一阶段节点。

### 官方参数

官方三步路径为：

```text
0.909375 → 0.8571428571 → 0.6666666667 → 0
```

采样器使用 Euler、CFG 1。插件的 `Exact Sigmas` 节点优先保持官方时间表；增加任意额外步数不保证更好，尤其可能影响高动态画面的时序稳定性。

### 致谢

本项目只是 ComfyUI 社区适配层，核心研究、模型和依赖来自以下项目：

- [NVIDIA / Efficient-Large-Model — Open-H3-Regenerate-2K](https://nvlabs.github.io/Sana/Open-H3-Regenerate-2K/)：原始研究、训练与官方推理方案。
- [NVlabs/Sana `sol-engine` 分支](https://github.com/NVlabs/Sana/tree/sol-engine/models/Open-H3-Regenerate-2K)：原始参考代码。
- [Efficient-Large-Model/Open-H3-Regenerate-2K](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K)：主 LoRA、Sharp curve branches 与模型文件。
- [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)：H3 3D latent upscaler 与 ComfyUI 节点。
- [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)：ComfyUI 格式 H3 基模、文本编码器与 VAE。
- [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)：MiniMax H3 基础模型。
- [ZhengmingYu/DMAD](https://huggingface.co/ZhengmingYu/DMAD)：示例第一阶段使用的 DMAD LoRA。
- [Kijai/MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental)：示例使用的 Singularity LoRA 提取版本。
- Open-H3 合并权重还包含来自 LightX2V FL2VA Turbo 与 LMS Sharp 的适配器；其许可证和署名请以 Open-H3 模型卡为准。

感谢以上作者和社区贡献者。模型文件不随本插件分发，各模型继续遵循各自的许可证；本仓库的代码许可证不会覆盖模型许可证。

## English

An early ComfyUI adaptation of [Open-H3-Regenerate-2K](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K). It maps the merged Diffusers LoRA to ComfyUI's fused-QKV MiniMax H3 implementation, loads the 50 LMS Sharp curve branches, provides the official 2/3-step sigma schedules, and can lock the audio stream during video refinement.

See the Chinese section above for the complete model tree, direct download links, example workflow, known limitations, and acknowledgements. This integration is preliminary and is not an official upstream implementation.

## License

The adapter code in this repository is released under the MIT License. Model weights are not included and remain subject to their respective upstream licenses, including the MiniMax H3 Community License where applicable.

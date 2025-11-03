# Matrix-Game Agents Overview

## Mission Snapshot
Matrix-Game is Skywork AI's interactive world foundation model family. Version 2.0, which this workspace focuses on, couples an autoregressive diffusion backbone with Wan VAE components to turn a single reference frame plus live keyboard/mouse inputs into controllable 25 fps video streams across several gameplay domains (universal environments, GTA-style driving, Temple Run). The codebase is organized to support turnkey inference, streaming demos, and ongoing research on world modelling, action-conditioned diffusion, and causal attention.

## High-Level Architecture
- **Interactive pipelines** (`Matrix-Game-2/pipeline/causal_inference.py`): Defines the causal inference and streaming pipelines that drive generation. They manage KV-cache priming, block-wise denoising, input conditioning, and decoding through the Wan diffusion model.
- **Inference entry points** (`Matrix-Game-2/inference.py`, `Matrix-Game-2/inference_streaming.py`): CLI programs that wrap the pipelines for scripted benchmarks or live user control. They handle I/O (image loading, resize/crop, action prompts), model/weight initialization, and video post-processing.
- **WAN model stack** (`Matrix-Game-2/wan/`): Houses the Wan diffusion backbone, transformer modules, schedulers, and VAE utilities pulled from the WanX ecosystem. These components provide the causal attention, FlashAttention integration, and high-fidelity latent decoding used by the pipeline.
- **Conditioning utilities** (`Matrix-Game-2/utils/conditions.py`, `Matrix-Game-2/demo_utils/`): Supply reusable action schedules, VAE wrappers, memory helpers, and visualization tools. They standardize keyboard/mouse embeddings, handle CLIP-based visual context, and annotate rendered outputs.
- **Configuration layer** (`Matrix-Game-2/configs/`): OmegaConf YAMLs that describe model hyperparameters, denoising schedules, and scene-specific conditioning presets. Switching configs swaps between universal, GTA driving, and Temple Run behaviors.
- **Assets & demos** (`Matrix-Game-2/assets/`, `Matrix-Game-2/demo_images/`): Contain icons and sample frames used in quick-start demos and overlay rendering.

## Typical Workflow
1. **Install & prepare weights**: Follow `Matrix-Game-2/README.md` to create the Conda environment, install FlashAttention/apex, and download the Hugging Face checkpoints (`Skywork/Matrix-Game-2.0`), which include the Wan 2.1 VAE.
2. **Choose a config**: Point `--config_path` to the desired YAML in `configs/inference_yaml/` to select scene dynamics and denoising schedule.
3. **Launch inference**:
   - Offline scripted demo: `python inference.py --img_path <frame> --checkpoint_path <weights> --pretrained_model_path <vae_dir>`.
   - Streaming keyboard control: `python inference_streaming.py --output_folder <dir>` and supply image paths plus actions interactively.
4. **Inspect outputs**: Videos and overlays land under the chosen `output_folder`, with optional mouse/keyboard visualizations handled by `utils/visualize.process_video`.

## Research & Extension Notes
- The causal pipelines expose hooks for experimenting with block sizes (`num_frame_per_block`), KV caching, and custom action embeddings—useful for probing long-horizon consistency or integrating new controllers.
- `utils.scheduler` and the Wan diffusion wrappers abstract the denoising loop; swapping schedulers or modifying timesteps can prototype alternative temporal sampling strategies.
- Additional scenes or tasks typically require:
  1. Extending `utils/conditions` with new action encodings.
  2. Authoring a matching config under `configs/`.
  3. Training or fine-tuning checkpoints housed alongside the existing Hugging Face weights.

## Complementary Resources
- Root-level `README.md`: release notes and project overview.
- `Matrix-Game-2/README.md`: detailed setup, quick-start commands, and citation.
- Wan documentation inside `Matrix-Game-2/wan/README.md`: model internals, distributed helpers, and training hints.
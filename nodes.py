import hashlib
import os

import torch
from safetensors import safe_open

import folder_paths
from comfy.nested_tensor import NestedTensor
from comfy.weight_adapter.lora import LoRAAdapter


OPEN_H3_FOLDER = "open_h3_regenerate"
OPEN_H3_DIR = os.path.join(folder_paths.models_dir, OPEN_H3_FOLDER)
folder_paths.add_model_folder_path(OPEN_H3_FOLDER, OPEN_H3_DIR)

EXPECTED_CURVE_TABLE_SHA256 = "c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7"
EXPECTED_MAIN_MODULES = 312
EXPECTED_CURVE_MODULES = 50
SIGMAS = {
    3: (0.909375, 0.8571428571428571, 0.6666666666666666, 0.0),
    2: (0.7, 0.49411764705882355, 0.0),
}


def _file_options(folder, preferred):
    names = [name for name in folder_paths.get_filename_list(folder) if name.lower().endswith(".safetensors")]
    return sorted(names, key=lambda name: (os.path.basename(name) != preferred, name.lower())) or ["(missing)"]


def _full_path(folder, name):
    if name == "(missing)":
        raise FileNotFoundError(f"No model is installed under ComfyUI/models/{folder}")
    return folder_paths.get_full_path_or_raise(folder, name)


def _curve_table_sha256(diffusion_model):
    table = getattr(diffusion_model, "adaln_t_table", None)
    if not torch.is_tensor(table) or tuple(table.shape) != (1025, 8):
        raise ValueError(
            "Open-H3 Regenerate needs the MiniMax H3 1025x8 curve-basis model. "
            "Use minimax_h3_ref2va_pruned_int8_convrot.safetensors."
        )
    raw = table.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _adapter(a, b, loaded_keys):
    return LoRAAdapter(set(loaded_keys), (b, a, None, None, None, None))


def _core_module_name(source):
    if source.startswith("transformer_blocks."):
        return "blocks." + source[len("transformer_blocks."):]
    if source.startswith("token_refiner.refiner_blocks."):
        return "token_refiner.blocks." + source[len("token_refiner.refiner_blocks."):]
    raise ValueError(f"Unsupported Open-H3 module: {source}")


def _main_target(source, b_rows, state_dict):
    for projection, index in (("to_q", 0), ("to_k", 1), ("to_v", 2)):
        suffix = f".attn.{projection}"
        if source.endswith(suffix):
            core = _core_module_name(source[:-len(suffix)] + ".attn.qkv_proj")
            key = f"diffusion_model.{core}.weight"
            if key not in state_dict or state_dict[key].shape[0] != b_rows * 3:
                raise ValueError(f"Open-H3 QKV target does not match the loaded H3 model: {key}")
            return (key, (0, index * b_rows, b_rows))

    replacements = {
        ".attn.to_out.0": ".attn.out_proj",
        ".ff.net.0.proj": ".mlp.fc1",
        ".ff.net.2": ".mlp.fc2",
    }
    for suffix, target_suffix in replacements.items():
        if source.endswith(suffix):
            core = _core_module_name(source[:-len(suffix)] + target_suffix)
            key = f"diffusion_model.{core}.weight"
            if key not in state_dict or state_dict[key].shape[0] != b_rows:
                raise ValueError(f"Open-H3 target does not match the loaded H3 model: {key}")
            return key
    raise ValueError(f"Unsupported Open-H3 LoRA target: {source}")


def _load_main_patches(path, state_dict):
    patches = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        roots = sorted(key[:-len(".lora_A.weight")] for key in keys if key.endswith(".lora_A.weight"))
        if len(roots) != EXPECTED_MAIN_MODULES:
            raise ValueError(f"Expected {EXPECTED_MAIN_MODULES} Open-H3 LoRA modules, found {len(roots)}")
        for root in roots:
            b_key = root + ".lora_B.weight"
            if b_key not in keys or not root.startswith("transformer."):
                raise ValueError(f"Incomplete Open-H3 LoRA pair: {root}")
            a_key = root + ".lora_A.weight"
            a = handle.get_tensor(a_key)
            b = handle.get_tensor(b_key)
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
                raise ValueError(f"Invalid Open-H3 LoRA shapes: {root}")
            source = root[len("transformer."):]
            target = _main_target(source, b.shape[0], state_dict)
            patches[target] = _adapter(a, b, (a_key, b_key))
    return patches


def _load_curve_patches(path, state_dict):
    patches = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        roots = sorted(key[:-len(".curve_A.weight")] for key in keys if key.endswith(".curve_A.weight"))
        if len(roots) != EXPECTED_CURVE_MODULES:
            raise ValueError(f"Expected {EXPECTED_CURVE_MODULES} Sharp curve branches, found {len(roots)}")
        for root in roots:
            a_key = root + ".curve_A.weight"
            b_key = root + ".curve_B.weight"
            if b_key not in keys or not root.startswith("transformer.transformer_blocks."):
                raise ValueError(f"Incomplete Sharp curve pair: {root}")
            a = handle.get_tensor(a_key)
            b = handle.get_tensor(b_key)
            block_path = root[len("transformer.transformer_blocks."):]
            block_index, suffix = block_path.split(".", 1)
            if suffix != "adaln_proj.linear" or a.shape[1] != 8:
                raise ValueError(f"Unsupported Sharp curve target: {root}")
            target = f"diffusion_model.blocks.{block_index}.adaln_proj.linear.weight"
            if target not in state_dict or state_dict[target].shape[0] != b.shape[0]:
                raise ValueError(f"Sharp curve target does not match the loaded H3 model: {target}")
            patches[target] = _adapter(a, b, (a_key, b_key))
    return patches


class OpenH3RegenerateModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "main_lora": (_file_options("loras", "open_h3_regenerate_2k_lora.safetensors"),),
                "curve_branches": (_file_options(OPEN_H3_FOLDER, "sharp_curve_branches.safetensors"),),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "patch"
    CATEGORY = "Open-H3/Regenerate 2K"

    def patch(self, model, main_lora, curve_branches, strength):
        diffusion_model = model.get_model_object("diffusion_model")
        curve_sha = _curve_table_sha256(diffusion_model)
        if curve_sha != EXPECTED_CURVE_TABLE_SHA256:
            raise ValueError(
                "The loaded H3 model has the wrong AdaLN curve table for Open-H3 Regenerate: "
                f"{curve_sha}. Use minimax_h3_ref2va_pruned_int8_convrot.safetensors."
            )

        patched = model.clone()
        state_dict = patched.model_state_dict()
        main_path = _full_path("loras", main_lora)
        curve_path = _full_path(OPEN_H3_FOLDER, curve_branches)
        main_patches = _load_main_patches(main_path, state_dict)
        curve_patches = _load_curve_patches(curve_path, state_dict)
        main_loaded = patched.add_patches(main_patches, strength_patch=strength)
        curve_loaded = patched.add_patches(curve_patches, strength_patch=strength)
        if len(main_loaded) != EXPECTED_MAIN_MODULES or len(curve_loaded) != EXPECTED_CURVE_MODULES:
            raise ValueError(
                f"Open-H3 patch application incomplete: main={len(main_loaded)}/{EXPECTED_MAIN_MODULES}, "
                f"curve={len(curve_loaded)}/{EXPECTED_CURVE_MODULES}"
            )
        report = (
            f"Open-H3 Regenerate loaded: {len(main_loaded)} main modules + "
            f"{len(curve_loaded)} Sharp curve branches; curve table {curve_sha}; strength {strength:g}"
        )
        return patched, report


class OpenH3RegenerateSigmas:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"steps": (["3", "2"], {"default": "3"})}}

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "sigmas"
    CATEGORY = "Open-H3/Regenerate 2K"

    def sigmas(self, steps):
        return (torch.tensor(SIGMAS[int(steps)], dtype=torch.float32),)


class OpenH3LockAudioForRefine:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"av_latent": ("LATENT",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "lock"
    CATEGORY = "Open-H3/Regenerate 2K"

    def lock(self, av_latent):
        samples = av_latent.get("samples")
        if not getattr(samples, "is_nested", False):
            raise ValueError("Open-H3 refinement expects a MiniMax H3 joint video/audio latent")
        streams = samples.unbind()
        if len(streams) != 2 or streams[0].ndim != 5 or streams[0].shape[1] != 24:
            raise ValueError("Open-H3 refinement expects H3 video then audio latent streams")
        output = av_latent.copy()
        output["noise_mask"] = NestedTensor((torch.ones_like(streams[0]), torch.zeros_like(streams[1])))
        return (output,)


NODE_CLASS_MAPPINGS = {
    "OpenH3RegenerateModelPatch": OpenH3RegenerateModelPatch,
    "OpenH3RegenerateSigmas": OpenH3RegenerateSigmas,
    "OpenH3LockAudioForRefine": OpenH3LockAudioForRefine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenH3RegenerateModelPatch": "Open-H3 Regenerate 2K Model Patch",
    "OpenH3RegenerateSigmas": "Open-H3 Regenerate Exact Sigmas",
    "OpenH3LockAudioForRefine": "Open-H3 Lock Audio for Refine",
}

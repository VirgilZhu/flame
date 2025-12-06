# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import torch
import torch.distributed.checkpoint as DCP
from transformers import AutoModelForCausalLM
import fla  # noqa
from torchtitan.tools.logging import init_logger, logger
import shutil

@torch.inference_mode()
def convert_hf_weights(model_path: str, checkpoint_path: Path):
    logger.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    original_state_dict = model.state_dict()

    new_state_dict = {}
    
    for key, value in original_state_dict.items():
        if key == "lm_head.weight":
            logger.info(f"Cloning {key} to ensure independence...")
            new_state_dict[key] = value.clone().detach()
        else:
            new_state_dict[key] = value
            
    # 兼容 FLA 命名
    if "lm_head.weight" not in new_state_dict:
        if "model.embed_tokens.weight" in new_state_dict:
             logger.warning("Patching missing lm_head from model.embed_tokens.weight")
             new_state_dict["lm_head.weight"] = new_state_dict["model.embed_tokens.weight"].clone().detach()
        elif "transformer.embeddings.weight" in new_state_dict:
             logger.warning("Patching missing lm_head from transformer.embeddings.weight")
             new_state_dict["lm_head.weight"] = new_state_dict["transformer.embeddings.weight"].clone().detach()

    if checkpoint_path.exists():
        shutil.rmtree(checkpoint_path)
    
    logger.info(f"Writing to DCP at '{checkpoint_path}'")
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    storage_writer = DCP.filesystem.FileSystemWriter(checkpoint_path, thread_count=8)
    
    DCP.save(new_state_dict, storage_writer=storage_writer)
    
    logger.info("Conversion Done (No 'model' wrapping)")

if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    convert_hf_weights(args.model, args.checkpoint)
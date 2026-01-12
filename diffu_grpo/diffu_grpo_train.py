import json
import logging as pylogging
import os
from pathlib import Path

import torch
from accel_reward import gen_step_reward, verifier_reward
from accelerate import PartialState
from data_utils import (
    get_apps_questions,
    get_dapo17_data,
    get_gsm8k_questions,
    get_math500_questions,
)
from diffu_grpo_config import DiffuGRPOConfig

# Custom imports
from diffu_grpo_trainer import DiffuGRPOTrainer
from huggingface_hub import snapshot_download
from peft import LoraConfig
from reward_func import (
    apps_reward_func,
    boxed_and_answer_tags_format_reward,
    codefence_reward_func,
    correctness_reward_func_math,
    soft_format_reward_func,
)
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_callback import TrainerCallback
from transformers.utils import logging
from trl import ModelConfig, TrlParser

from model.llada.configuration_llada import LLaDAConfig
from model.llada.lladou import LLaDOUModelLM
from model.path_utils import lladou_config_dir
from utils import set_random_seed

logging.set_verbosity_info()
logger = logging.get_logger("transformers")


class LocalJSONLFileHandler(pylogging.Handler):
    def __init__(self, log_path: str):
        super().__init__()
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if logger.handlers and getattr(logger.handlers[0], "formatter", None):
            self.setFormatter(logger.handlers[0].formatter)

    def emit(self, record: pylogging.LogRecord):
        message = self.format(record)
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(message)
            log_file.write("\n")


class LocalFileLoggerCallback(TrainerCallback):
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        record = {"step": state.global_step, **logs}
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False))
            log_file.write("\n")


def main(grpo_config, model_config):
    # Set seed for reproducibility
    set_random_seed(grpo_config.seed)
    prompt_mode = grpo_config.prompt_mode
    thinking_mode = prompt_mode.lower() == "thinking"
    resume_from_checkpoint = (
        os.path.exists(grpo_config.output_dir)
        and len(os.listdir(grpo_config.output_dir))
        > 1  # there can be one log file without checkpoint
    )

    dataset_registry = {
        "gsm8k": {
            "loader": lambda: get_gsm8k_questions("train", prompt_mode=prompt_mode),
            "reward_funcs": [
                correctness_reward_func_math,
                boxed_and_answer_tags_format_reward,
            ],
            "reward_weights": [1.0, 1.0],
        },
        "math500": {
            "loader": lambda: get_math500_questions("train", prompt_mode=prompt_mode),
            "reward_funcs": [
                correctness_reward_func_math,
                boxed_and_answer_tags_format_reward,
            ],
            "reward_weights": [1.0, 1.0],
        },
        "apps": {
            "loader": lambda: get_apps_questions("train", prompt_mode=prompt_mode),
            "reward_funcs": [apps_reward_func, codefence_reward_func],
            "reward_weights": [3.0, 1.0],
        },
        "dapo17": {
            "loader": lambda: get_dapo17_data("train", prompt_mode=prompt_mode),
            "reward_funcs": [
                correctness_reward_func_math,
                boxed_and_answer_tags_format_reward,
            ],
            "reward_weights": [1.0, 1.0],
        },
    }

    if thinking_mode:
        thinking_datasets = ("gsm8k", "math500", "dapo17", "apps")
        for key in thinking_datasets:
            if key in dataset_registry:
                dataset_registry[key]["reward_funcs"] = [
                    *dataset_registry[key]["reward_funcs"],
                    soft_format_reward_func,
                ]
                dataset_registry[key]["reward_weights"] = [
                    *dataset_registry[key]["reward_weights"],
                    0.5,
                ]

    dataset_key = grpo_config.dataset
    if dataset_key not in dataset_registry:
        raise ValueError(f"Dataset {grpo_config.dataset} not supported")

    config = dataset_registry[dataset_key]

    dataset = config["loader"]()
    dataset = dataset.map(lambda _: {"dataset_name": dataset_key})
    reward_functions = list(config["reward_funcs"])
    reward_weights = list(config["reward_weights"])
    assert len(reward_functions) == len(reward_weights)

    # Num of generation step reward
    reward_functions.append(gen_step_reward)
    gen_step_reward_weight = 0.0 if grpo_config.freeze_unmasking_head else 0.1
    reward_weights.append(gen_step_reward_weight)

    # on policy distill reward
    reward_functions.append(verifier_reward)
    reward_weights.append(1.0)

    # Shuffle dataset with fixed seed for reproducibility
    dataset = dataset.shuffle(seed=grpo_config.seed)

    train_set = dataset

    # 4 bit quantization configuration
    if model_config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_config.load_in_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None

    # Load model and tokenizer
    # we need to first get the local dir, otherwise there is a chance of network error when downloading the model in parallel
    local_dir = ""
    state = PartialState()
    if state.is_main_process:
        local_dir = snapshot_download(grpo_config.model_path)
    state.wait_for_everyone()
    local_dir = snapshot_download(grpo_config.model_path)

    if grpo_config.use_official_model:
        model = AutoModel.from_pretrained(
            local_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        )
    else:
        # if "llada" in grpo_config.model_path.lower():
        #     llada_config = LLaDAConfig.from_pretrained(
        #         Path("../model/llada/config.json")
        #     )
        #     assert llada_config.flash_attention
        #     model = LLaDAModelLM.from_pretrained(
        #         local_dir,
        #         config=llada_config,
        #         trust_remote_code=True,
        #         torch_dtype=torch.bfloat16,
        #         quantization_config=bnb_config,
        #     )
        # elif "lladou" in grpo_config.model_path.lower():
        lladou_config = LLaDAConfig.from_pretrained(lladou_config_dir())
        assert lladou_config.flash_attention
        model = LLaDOUModelLM.from_pretrained(
            local_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            config=lladou_config,
        )
    with state.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(
            grpo_config.tokenizer_path, trust_remote_code=True, use_fast=True
        )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    # Configure LoRA for parameter-efficient fine-tuning
    peft_config = None
    if model_config.use_peft:
        peft_config = LoraConfig(
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
                "gate_proj",
            ],
            task_type=model_config.lora_task_type,
            lora_dropout=model_config.lora_dropout,
        )
    # Initialize and run trainer
    grpo_config.reward_weights = reward_weights
    if grpo_config.freeze_unmasking_head and hasattr(model, "freeze_mask_head"):
        model.freeze_mask_head()

    trainer = DiffuGRPOTrainer(
        args=grpo_config,
        model=model,
        peft_config=peft_config,
        reward_funcs=reward_functions,
        train_dataset=train_set,
        processing_class=tokenizer,
    )

    local_log_path = grpo_config.local_log_path or os.path.join(
        grpo_config.output_dir, "local_training_logs.jsonl"
    )
    if not any(
        isinstance(handler, LocalJSONLFileHandler)
        and handler.log_path == Path(local_log_path)
        for handler in logger.handlers
    ):
        logger.addHandler(LocalJSONLFileHandler(local_log_path))
    trainer.add_callback(LocalFileLoggerCallback(local_log_path))

    torch.autograd.set_detect_anomaly(False)
    torch.set_float32_matmul_precision("high")

    # log the configs
    # logger.info(f"GRPOConfig: {grpo_config}")
    # logger.info(f"ModelConfig: {model_config}")
    logger.info(f"Local training logs will be written to: {local_log_path}")
    # resume_from_checkpoint = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "fast-llada"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    parser = TrlParser((DiffuGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    main(grpo_config=grpo_config, model_config=model_config)

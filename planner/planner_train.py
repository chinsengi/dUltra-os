import os

import hydra
import torch
from data import (
    dLLMDataCollator,
    dLLMSFTDataset,
    preprocess_nvidia_dataset,
    preprocess_s1k_dataset,
)
from omegaconf import DictConfig, OmegaConf
from planner_trainer import PlannerTrainer
from transformers import AutoTokenizer, TrainingArguments

from model.llada.configuration_llada import LLaDAConfig
from model.llada.lladou import LLaDOUModelLM
from model.path_utils import lladou_config_dir
from utils import set_random_seed

os.environ["WANDB_PROJECT"] = "llada-planner"


# Model loading with LoRA integration
def load_model_and_tokenizer(cfg):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name,
        padding_side="right",
        trust_remote_code=True,
        use_fast=True,
    )

    # Load LLaDOU model with LLaDA weights (planner head stays randomly initialized)
    config = LLaDAConfig.from_pretrained(lladou_config_dir())
    model = LLaDOUModelLM.from_pretrained(
        cfg.model.name,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        strict=False,
    )

    model.freeze_main_model()

    return tokenizer, model


# Dataset loading
def load_data(cfg, tokenizer, do_eval=True):
    if cfg.data.train_dataset == "s1k":
        train_data, eval_data = preprocess_s1k_dataset(
            tokenizer, cfg.training.max_length
        )
        print("Train data length: ", len(train_data))
        train_dataset = dLLMSFTDataset(train_data, tokenizer)
        eval_dataset = None
        if do_eval and eval_data is not None:
            print("Eval data length: ", len(eval_data))
            eval_dataset = dLLMSFTDataset(eval_data, tokenizer, eval=True)
    elif cfg.data.train_dataset == "nvidia":
        # Streaming dataset - returns IterableDataset directly
        train_dataset, eval_dataset = preprocess_nvidia_dataset(
            tokenizer,
            split=cfg.data.train_split,
            max_length=cfg.training.max_length,
            test_split=cfg.data.eval_split,
            num_proc=cfg.data.num_proc,
        )
        print("Using streaming dataset (length unknown)")
    return train_dataset, eval_dataset


# Training setup
def train_model(cfg, tokenizer, model):
    # Set seed for reproducibility
    resume_from_checkpoint = (
        os.path.exists(cfg.output.dir)
        and len(os.listdir(cfg.output.dir))
        > 1  # there can be one log file without checkpoint
        and cfg.training.resume_from_checkpoint
    )

    # Load dataset
    do_eval = cfg.evaluation.strategy != "no"
    train_dataset, eval_dataset = load_data(cfg, tokenizer, do_eval=do_eval)

    # Training arguments setup
    training_args = TrainingArguments(
        output_dir=os.path.join(cfg.output.dir, cfg.output.job_name),
        num_train_epochs=cfg.training.num_epochs if not cfg.training.max_steps else -1,
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.grad_accum_steps,
        eval_strategy=cfg.evaluation.strategy,
        eval_steps=cfg.evaluation.eval_steps,
        logging_steps=cfg.logging.steps,
        save_steps=cfg.evaluation.save_steps,
        save_total_limit=cfg.evaluation.save_total_limit,
        learning_rate=cfg.training.learning_rate,
        load_best_model_at_end=cfg.evaluation.load_best_model_at_end and do_eval,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,
        bf16=cfg.training.bf16,
        report_to=cfg.logging.report_to if not cfg.debug else "none",
        run_name=cfg.output.job_name,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        torch_compile=cfg.model.torch_compile,
        torch_compile_backend="inductor" if cfg.model.torch_compile else None,
        per_device_eval_batch_size=cfg.evaluation.per_device_eval_batch_size,
        hub_model_id=cfg.output.hub_model_id,
        max_steps=cfg.training.max_steps,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        lr_scheduler_kwargs=cfg.training.lr_scheduler_kwargs,
        push_to_hub=cfg.output.push_to_hub,
    )

    # Initialize Trainer with custom dLLMTrainer
    trainer = PlannerTrainer(
        model=model,
        args=training_args,
        data_collator=dLLMDataCollator(
            tokenizer=tokenizer,
            mask_token_id=cfg.mask_token_id,
            max_length=cfg.training.max_length,
            truncate=cfg.collator.truncate,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.model_accepts_loss_kwargs = False

    # Start training
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.push_to_hub()


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    set_random_seed(42)

    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(cfg)

    # Train the model
    train_model(cfg, tokenizer, model)


if __name__ == "__main__":
    main()

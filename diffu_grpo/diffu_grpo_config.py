from dataclasses import dataclass, field

from trl import GRPOConfig


@dataclass
class DiffuGRPOConfig(GRPOConfig):
    scale_rewards: str = field(
        default="none",
        metadata={
            "help": "Specifies the scaling strategy for rewards. Supported values are: "
            "`True` or `group'` (default): rewards are scaled by the standard deviation within each group, ensuring "
            "unit variance within a group. "
            "`'batch'`: rewards are scaled by the standard deviation across the entire batch, as recommended in the "
            "PPO Lite paper. "
            "`False` or `'none'`: no scaling is applied. The Dr. GRPO paper recommends not scaling rewards, as "
            "scaling by the standard deviation introduces a question-level difficulty bias."
        },
    )
    tokenizer_path: str | None = field(
        default="GSAI-ML/LLaDA-8B-Instruct",
    )
    model_path: str | None = field(
        default="sengi/LLaDA-8B-Instruct",
    )
    use_official_model: bool | None = field(
        default=False,
    )
    temperature: float | None = field(
        default=0.3,
    )
    max_unmasking_prob: float | None = field(
        default=0.05,
    )
    dataset: str | None = field(
        default="gsm8k",
    )
    prompt_mode: str = field(
        default="thinking",
        metadata={
            "help": "Prompt style for math and coding datasets. Supported values: 'thinking' or 'non-thinking'."
        },
    )
    mask_id: int = field(
        default=126336,
        metadata={"help": "Mask token id. Default is from Llada"},
    )
    normalize: bool | None = field(
        default=True,
    )
    scale: float | None = field(
        default=30.0,
    )
    use_scheduler: bool | None = field(
        default=False,
    )
    block_length: int | None = field(
        default=32,
    )
    rollout_mode: str = field(
        default="training",
        metadata={
            "help": "Generation mode to use during rollout (e.g., 'training' or 'inference').",
        },
    )
    advantage_min_clip: float | None = field(
        default=0.0,
    )
    freeze_unmasking_head: bool = field(
        default=False,
        metadata={
            "help": "When true, keep the unmasking probability head frozen and disable its auxiliary reward."
        },
    )
    use_reverse_kl: bool = field(
        default=False,
        metadata={
            "help": "When true, pass student logprobs to calculate reverse KL for on-policy distillation."
        },
    )
    local_log_path: str | None = field(
        default=None,
        metadata={
            "help": "Optional path to a JSONL file where training logs are written locally.",
        },
    )

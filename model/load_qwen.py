import torch
from transformers import AutoTokenizer

from model.qwen3.modeling_qwen3 import Qwen3ForPlanner


def main():
    model_name = "Qwen/Qwen3-0.6B"
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Qwen3ForPlanner.from_pretrained(model_name, torch_dtype="auto")
    # prepare the model input
    prompt = "Give me a short introduction to large language model."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    # generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
    outputs = model(**model_inputs, max_new_tokens=32768)
    return outputs
    # output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()


if __name__ == "__main__":
    main()

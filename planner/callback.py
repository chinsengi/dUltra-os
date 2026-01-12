import random

import torch
from transformers import TrainerCallback


class SamplePrinterCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, num_samples=3, max_new_tokens=64):
        self.tok = tokenizer
        self.ds = eval_dataset
        self.k = num_samples
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, **kwargs):
        # only rank 0 prints
        if not state.is_local_process_zero:
            return
        model = kwargs["model"]

        idxs = random.sample(range(len(self.ds)), k=min(self.k, len(self.ds)))
        for i in idxs:
            ex = self.ds[i]
            input_ids = torch.as_tensor(ex["input_ids"]).unsqueeze(0)
            attn = ex.get("attention_mask")
            if attn is not None:
                attn = torch.as_tensor(attn).unsqueeze(0)

            input_ids = input_ids.to(model.device)
            if attn is not None:
                attn = attn.to(model.device)

            with torch.no_grad():
                gen = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=self.max_new_tokens,
                )

            src = self.tok.decode(input_ids[0], skip_special_tokens=True)
            pred = self.tok.decode(gen[0], skip_special_tokens=True)

            gold_txt = None
            labels = ex.get("labels")
            if labels is not None:
                labels = torch.as_tensor(labels)
                labels = labels.masked_fill(labels == -100, self.tok.pad_token_id)
                gold_txt = self.tok.decode(labels, skip_special_tokens=True)

            print("=" * 60)
            print("EVAL SAMPLE")
            print(f"SRC : {src}")
            if gold_txt is not None:
                print(f"GOLD: {gold_txt}")
            print(f"PRED: {pred}")

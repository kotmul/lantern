from datetime import datetime, timezone, timedelta
from vllm import LLM, SamplingParams
from saer.utils import DL_to_LD
import argparse
import datasets
import torch
import tqdm
import json
import os
import gc

KST = timezone(timedelta(hours=9))

def parse_args():
    parser = argparse.ArgumentParser(description="Get Model Response using vLLM")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=20_000)
    parser.add_argument("--dataset_path", type=str, default=None, help="Dataset Path. Need to save to local disk.")
    parser.add_argument("--world_size", type=int)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()

def main(args):
    print(f"👊 [RANK {args.local_rank}] Ready to Model Generation!!")
    print(f"👊 [RANK {args.local_rank}] Model ? {args.model_name}")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    generation_kwargs = SamplingParams(
        temperature=0,
        max_tokens=args.max_length
    )
 
    llm = LLM(
        args.model_name,
        # enforce_eager=True,
        tensor_parallel_size=1,
    )

    if "llama" in args.model_name.lower():
        save_file = f"llama_response_{args.local_rank}.jsonl"
    elif "qwen" in args.model_name.lower():
        save_file = f"qwen_response_{args.local_rank}.jsonl"

    ds = datasets.load_from_disk(args.dataset_path)

    if args.world_size != 1:
        ds = ds.shard(num_shards=args.world_size, index=args.local_rank)
    
    with torch.no_grad():
        for idx in tqdm.tqdm(range(0, len(ds), args.batch_size)):
            batch = ds[idx:idx+args.batch_size]
            text = batch['text']

            # Get Origianl Outputs
            outputs = llm.generate(text, generation_kwargs, use_tqdm=False)
            generation = [out.outputs[0].text for out in outputs]
            
            batch.update({f"response": generation})
            
            # SAVE CODE
            batch = DL_to_LD(batch)

            with open(os.path.join(args.save_dir, save_file), "a", encoding="utf-8") as _file:
                for mini in batch:
                    json.dump(mini, _file, ensure_ascii=False)
                    _file.write("\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
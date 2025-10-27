import time
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset

def estimate_llm_throughput(model_name, dataset_name, num_samples, max_tokens, temperature, max_model_len):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(model=model_name, max_model_len=max_model_len)

    # Load dataset safely
    try:
        dataset = load_dataset(dataset_name, split=f"test_sft[:{num_samples}]")
    except Exception:
        dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]")

    prompts = []
    for item in dataset:
        if "prompt" in item:
            messages = [{"role": "user", "content": item["prompt"]}]
        elif "messages" in item:
            messages = item["messages"]
        else:
            messages = [{"role": "user", "content": item.get("text", "")}]
        
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            prompt = messages[-1]["content"]
        prompts.append(prompt)
        
        if len(prompts) >= num_samples:
            break

    if not prompts:
        raise ValueError("No valid prompts extracted from dataset.")

    print("Example prompt:", prompts[0][:200])

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature, n=1)

    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    total_generation_time = end_time - start_time
    total_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    total_input_tokens = sum(len(output.prompt_token_ids) for output in outputs)

    throughput = total_output_tokens / total_generation_time
    input_throughput = total_input_tokens / total_generation_time
    total_throughput = (total_input_tokens + total_output_tokens) / total_generation_time

    print(f"\nThroughput Estimation Results:")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {len(prompts)}")
    print(f"Max tokens per output: {max_tokens}")
    print(f"Total time: {total_generation_time:.2f}s")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Input token throughput: {input_throughput:.2f} tokens/s")
    print(f"Output token throughput: {throughput:.2f} tokens/s")
    print(f"Total token throughput: {total_throughput:.2f} tokens/s")


def main():
    parser = argparse.ArgumentParser(description="Estimate LLM throughput using vLLM")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_model_len", type=int, default=4096)
    args = parser.parse_args()

    estimate_llm_throughput(**vars(args))


if __name__ == "__main__":
    main()

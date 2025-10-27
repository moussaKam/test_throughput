| GPU | Model | Dataset | Samples | Max tokens per output | Total time | Total input tokens | Total output tokens | Input token throughput | Output token throughput | Total token throughput |
|-----|-------|---------|---------|------------------------|------------|--------------------|---------------------|------------------------|--------------------------|------------------------|
| RTX 5090 | Qwen/Qwen3-4B-Instruct-2507 | HuggingFaceH4/ultrachat_200k | 10,000 | 512 | 672.08 s | 1,825,952 | 3,908,227 | 2,716.86 tokens/s | 5,815.11 tokens/s | 8,531.97 tokens/s |
| H100 | Qwen/Qwen3-4B-Instruct-2507 | HuggingFaceH4/ultrachat_200k | 10,000 | 512 | 265.92 s | 1,825,952 | 3,923,383 | 6,866.60 tokens/s | 14,754.11 tokens/s | 21,620.71 tokens/s |
| L40S | Qwen/Qwen3-4B-Instruct-2507 | HuggingFaceH4/ultrachat_200k | 10,000 | 512 | 870.28 s | 1,825,952 | 3,935,047 | 2,098.12 tokens/s | 4,521.59 tokens/s | 6,619.72 tokens/s |

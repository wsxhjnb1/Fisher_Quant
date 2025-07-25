#!/bin/bash

# First run fisher

#CUDA_VISIBLE_DEVICES=0 python gradients/run-fisher.py --model_name_or_path luodian/llama-7b-hf --output_dir llama7b --dataset wikitext2 --seqlen 2048 --maxseqlen 2048 --num_examples 16
#CUDA_VISIBLE_DEVICES=0 python gradients/run-fisher.py --model_name_or_path meta-llama/Llama-2-7b-hf --output_dir llama2_7b --dataset wikitext2 --seqlen 2048 --maxseqlen 2048 --num_examples 16
CUDA_VISIBLE_DEVICES=0 python gradients/run-fisher.py --model_name_or_path meta-llama/Meta-Llama-3-8B --output_dir llama3_8b --dataset wikitext2 --seqlen 2048 --maxseqlen 2048 --num_examples 16
#CUDA_VISIBLE_DEVICES=0 python gradients/run-fisher.py --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --output_dir deep --dataset wikitext2 --seqlen 2048 --maxseqlen 2048 --num_examples 16
#CUDA_VISIBLE_DEVICES=0 python gradients/run-fisher.py --model_name_or_path Qwen/Qwen3-14B --output_dir Qwen --dataset wikitext2 --seqlen 2048 --maxseqlen 2048 --num_examples 16

# Then run simquant

#CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py luodian/llama-7b-hf --abits 4 --nsamples 16 --seqlen 2048 --nuq --fisher llama7b --quantize --quantizer-path quantizers_llama7b_4.pickle
#CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py luodian/llama-7b-hf --abits 2 --nsamples 16 --seqlen 2048 --nuq --fisher llama7b --quantize --quantizer-path quantizers_llama7b_2.pickle
#CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py luodian/llama-7b-hf --abits 1 --nsamples 16 --seqlen 2048 --nuq --fisher llama7b --quantize --quantizer-path quantizers_llama7b_1.pickle

#CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py meta-llama/Llama-2-7b-hf --abits 4 --nsamples 16 --seqlen 2048 --nuq --fisher llama2_7b --quantize --quantizer-path quantizers_llama2_7b_4.pickle
#CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py meta-llama/Llama-2-7b-hf --abits 2 --nsamples 16 --seqlen 2048 --nuq --fisher llama2_7b --quantize --quantizer-path quantizers_llama2_7b_2.pickle
#CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py meta-llama/Llama-2-7b-hf --abits 1 --nsamples 16 --seqlen 2048 --nuq --fisher llama2_7b --quantize --quantizer-path quantizers_llama2_7b_1.pickle

CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py meta-llama/Meta-Llama-3-8B --abits 4 --nsamples 16 --seqlen 2048 --nuq --fisher llama3_8b --quantize --quantizer-path quantizers_llama3_8b_4.pickle
CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py meta-llama/Meta-Llama-3-8B --abits 2 --nsamples 16 --seqlen 2048 --nuq --fisher llama3_8b --quantize --quantizer-path quantizers_llama3_8b_2.pickle
CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py meta-llama/Meta-Llama-3-8B --abits 1 --nsamples 16 --seqlen 2048 --nuq --fisher llama3_8b --quantize --quantizer-path quantizers_llama3_8b_1.pickle  

#CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --abits 4 --nsamples 16 --seqlen 2048 --nuq --fisher deep --quantize --quantizer-path quantizers_deep_4.pickle
#CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --abits 2 --nsamples 16 --seqlen 2048 --nuq --fisher deep --quantize --quantizer-path quantizers_deep_2.pickle
#CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --abits 1 --nsamples 16 --seqlen 2048 --nuq --fisher deep --quantize --quantizer-path quantizers_deep_1.pickle

#CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py Qwen/Qwen3-14B --abits 4 --nsamples 16 --seqlen 2048 --nuq --fisher Qwen --quantize --quantizer-path quantizers_Qwen_4.pickle
#CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py Qwen/Qwen3-14B --abits 2 --nsamples 16 --seqlen 2048 --nuq --fisher Qwen --quantize --quantizer-path quantizers_Qwen_2.pickle
#CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py Qwen/Qwen3-14B --abits 1 --nsamples 16 --seqlen 2048 --nuq --fisher Qwen --quantize --quantizer-path quantizers_Qwen_1.pickle

# Evaluate

#CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py luodian/llama-7b-hf --abits 4 --nsamples 16 --seqlen 2048 --nuq --quantizer-path quantizers_llama7b_4.pickle
#CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py luodian/llama-7b-hf --abits 2 --nsamples 16 --seqlen 2048 --nuq --quantizer-path quantizers_llama7b_2.pickle
#CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py luodian/llama-7b-hf --abits 1 --nsamples 16 --seqlen 2048 --nuq --quantizer-path quantizers_llama7b_1.pickle

#CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py meta-llama/Llama-2-7b-hf --abits 4 --nsamples 16 --seqlen 2048 --nuq --quantizer-path quantizers_llama2_7b_4.pickle > quantizers_llama2_7b_4.log
#CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py meta-llama/Llama-2-7b-hf --abits 2 --nsamples 16 --seqlen 2048 --nuq --quantizer-path quantizers_llama2_7b_2.pickle > quantizers_llama2_7b_2.log
#CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py meta-llama/Llama-2-7b-hf --abits 1 --nsamples 16 --seqlen 2048 --nuq --quantizer-path quantizers_llama2_7b_1.pickle > quantizers_llama2_7b_1.log

CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py meta-llama/Meta-Llama-3-8B --abits 4 --nsamples 16 --seqlen 2048 --nuq --quantizer-path quantizers_llama3_8b_4.pickle > quantizers_llama3_8b_4.log
CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py meta-llama/Meta-Llama-3-8B --abits 2 --nsamples 16 --seqlen 2048 --nuq --quantizer-path quantizers_llama3_8b_2.pickle > quantizers_llama3_8b_2.log
CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py meta-llama/Meta-Llama-3-8B --abits 1 --nsamples 16 --seqlen 2048 --nuq --quantizer-path quantizers_llama3_8b_1.pickle > quantizers_llama3_8b_1.log

#CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --abits 4 --nsamples 16 --seqlen 2048 --nuq --quantizer-path quantizers_deep_4.pickle
#CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --abits 2 --nsamples 16 --seqlen 2048 --nuq --quantizer-path quantizers_deep_2.pickle
#CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --abits 1 --nsamples 16 --seqlen 2048 --nuq --quantizer-path quantizers_deep_1.pickle

#CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py Qwen/Qwen3-14B --abits 4 --nsamples 16 --seqlen 2048 --nuq --quantizer-path quantizers_Qwen_4.pickle
#CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py Qwen/Qwen3-14B --abits 2 --nsamples 16 --seqlen 2048 --nuq --quantizer-path quantizers_Qwen_2.pickle
#CUDA_VISIBLE_DEVICES=0 python quant/llama_simquant.py Qwen/Qwen3-14B --abits 1 --nsamples 16 --seqlen 2048 --nuq --quantizer-path quantizers_Qwen_1.pickle


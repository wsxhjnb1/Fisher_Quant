import numpy as np
import torch
from transformers import AutoTokenizer
import random
from datasets import load_dataset

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def detect_model_type(model_id):
    """Detect model type from model_id"""
    model_id_lower = model_id.lower()
    if "magistral" in model_id_lower or "mistral" in model_id_lower:
        return "mistral"
    if "qwen" in model_id_lower:
        return "qwen"
    if "llama" in model_id_lower:
        return "llama"
    return "auto"


def load_tokenizer(model_id, model_type):
    """Load tokenizer with appropriate configuration based on model type"""
    
    # Special handling for Magistral models that require mistral-common
    if "magistral" in model_id.lower():
        try:
            print(f"Detected Magistral model, trying mistral-common tokenizer...")
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
            tokenizer = MistralTokenizer.from_hf_hub(model_id)
            print("Successfully loaded tokenizer with mistral-common")
            
            # Create a wrapper to make it compatible with the script
            class MistralTokenizerWrapper:
                def __init__(self, mistral_tokenizer):
                    self.mistral_tokenizer = mistral_tokenizer
                    self.pad_token = None
                    self.eos_token = None
                    self.padding_side = "right"
                
                def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
                    # For simple tokenization needed by the script
                    if isinstance(text, str) and text.strip():
                        try:
                            from mistral_common.protocol.instruct.messages import UserMessage
                            from mistral_common.protocol.instruct.request import ChatCompletionRequest
                            
                            # Create a simple user message and encode it
                            user_message = UserMessage(content=text)
                            request = ChatCompletionRequest(messages=[user_message])
                            encoded = self.mistral_tokenizer.encode_chat_completion(request)
                            tokens = encoded.tokens
                            
                            # Apply truncation if requested
                            if truncation and max_length and len(tokens) > max_length:
                                tokens = tokens[:max_length]
                            
                            # Convert to tensor
                            input_ids = torch.tensor([tokens], dtype=torch.long)
                            return {"input_ids": input_ids}
                        except Exception as e:
                            print(f"Error in MistralTokenizer: {e}")
                            return {"input_ids": torch.tensor([[]], dtype=torch.long)}
                    else:
                        return {"input_ids": torch.tensor([[]], dtype=torch.long)}
            
            wrapped_tokenizer = MistralTokenizerWrapper(tokenizer)
            return wrapped_tokenizer
            
        except ImportError:
            print("mistral-common not available, please install it manually: pip install mistral-common>=1.6.0")
            print("Falling back to standard tokenizer...")
        except Exception as e:
            print(f"Error loading with mistral-common: {e}")
            print("Falling back to standard tokenizer...")
        
        # Try a simple workaround for Magistral models without mistral-common
        try:
            print("Attempting Magistral workaround with slow tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                trust_remote_code=True, 
                use_fast=False,
                legacy=False
            )
            print("Successfully loaded Magistral tokenizer with workaround")
            
            # Set padding token if not already set
            if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is None:
                if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.pad_token = "<pad>"
            
            if hasattr(tokenizer, 'padding_side'):
                tokenizer.padding_side = "right"
            
            return tokenizer
        except Exception as workaround_e:
            print(f"Magistral workaround failed: {workaround_e}")
            print("Continuing with standard tokenizer loading...")
    
    tokenizer_configs = {
        "qwen": {
            "trust_remote_code": True,
            "legacy": False
        },
        "llama4": {
            "trust_remote_code": True,
            "legacy": False
        },
        "llama": {
            "trust_remote_code": True,
            "legacy": False
        },
        "mistral": {
            "trust_remote_code": True,
            "legacy": False,
            "use_fast": False  # Use slow tokenizer for better compatibility
        },
        "qwen3": {
            "trust_remote_code": True,
            "legacy": False
        },
        "auto": {
            "trust_remote_code": True,
            "legacy": False
        }
    }
    
    config = tokenizer_configs.get(model_type, tokenizer_configs["auto"])
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, **config)
        print(f"Successfully loaded tokenizer with {model_type} configuration")
    except Exception as e:
        print(f"Error loading tokenizer with {model_type} configuration: {e}")
        print("Trying fallback tokenizer configuration...")
        # Multiple fallback attempts
        fallback_configs = [
            {"trust_remote_code": True, "use_fast": False},
            {"trust_remote_code": True, "use_fast": False, "legacy": True},
            {"trust_remote_code": True, "use_fast": False, "legacy": False},
            {"trust_remote_code": True},
            {"use_fast": False},
            {"use_fast": False, "legacy": True},
            {"trust_remote_code": False, "use_fast": False},
            {}  # Last resort: minimal config
        ]
        
        tokenizer = None
        for i, fallback_config in enumerate(fallback_configs):
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, **fallback_config)
                print(f"Successfully loaded tokenizer with fallback configuration {i+1}")
                break
            except Exception as fallback_e:
                print(f"Fallback {i+1} failed: {fallback_e}")
                continue
        
        if tokenizer is None:
            raise Exception("All tokenizer loading attempts failed")
    
    # Set padding token if not already set
    if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is None:
        if hasattr(tokenizer, 'eos_token'):
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Create a dummy pad token if needed
            tokenizer.pad_token = "<pad>"
    
    if hasattr(tokenizer, 'padding_side'):
        tokenizer.padding_side = "right"
    
    return tokenizer

def get_training_dataset(seed, seqlen, model_path, dataset_name="wikitext2", nsamples=128):
    """Get training dataset for the specified dataset type"""
    
    if dataset_name == "wikitext2":
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        
        # Use the model loading utilities for consistency
        model_type_local = detect_model_type(model_path)
        tokenizer_local = load_tokenizer(model_path, model_type_local)
        trainenc = tokenizer_local("\n\n".join(traindata['text']), return_tensors='pt')
        testenc = tokenizer_local("\n\n".join(testdata['text']), return_tensors='pt')
        
    elif dataset_name == "c4":
        # Use streaming to avoid loading entire C4 dataset into memory
        import time
        import os
        max_retries = 10
        retry_delay = 10
        
        # Set longer timeout for HuggingFace Hub requests
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
        
        for attempt in range(max_retries):
            try:
                print(f"Loading C4 dataset (attempt {attempt + 1}/{max_retries})...")
                traindata = load_dataset(
                    'allenai/c4', 'en', 
                    split='train',
                    streaming=True
                )
                testdata = load_dataset(
                    'allenai/c4', 'en', 
                    split='validation',
                    streaming=True
                )
                print("Successfully loaded C4 dataset")
                break
            except Exception as e:
                print(f"Error loading C4 dataset (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print("Failed to load C4 dataset after all retries. Falling back...")
                    # Fallback to wikitext2 if C4 fails
                    return get_training_dataset(seed, seqlen, model_path, 'c4', nsamples)
        
        # Use the model loading utilities for consistency
        model_type_local = detect_model_type(model_path)
        tokenizer_local = load_tokenizer(model_path, model_type_local)
        
        # Limit to 2048 * 1200 tokens total to prevent OOM
        max_tokens = 2048 * 1200
        
        # Process training data incrementally
        train_texts = []
        train_token_count = 0
        print(f"Loading C4 training data (target: {max_tokens:,} tokens)...")
        
        for item in traindata:
            text = item['text']
            if text and text.strip():  # Skip empty texts
                # Tokenize this individual text to count tokens
                tokens = tokenizer_local(text, return_tensors='pt')
                if isinstance(tokens, dict):
                    token_count = tokens["input_ids"].shape[1]
                else:
                    token_count = tokens.input_ids.shape[1]
                
                # Check if adding this text would exceed our limit
                if train_token_count + token_count > max_tokens:
                    # If this single text would exceed limit, try to take a portion
                    remaining_tokens = max_tokens - train_token_count
                    if remaining_tokens > 0:
                        # Take a portion of this text
                        words = text.split()
                        # Rough estimate: take proportional words
                        word_ratio = remaining_tokens / token_count
                        target_words = max(1, int(len(words) * word_ratio))
                        partial_text = " ".join(words[:target_words])
                        train_texts.append(partial_text)
                        train_token_count = max_tokens
                    break
                else:
                    train_texts.append(text)
                    train_token_count += token_count
                    
                    if train_token_count >= max_tokens:
                        break
        
        print(f"Loaded {len(train_texts)} training texts with ~{train_token_count:,} tokens")
        
        # Process test data with a smaller limit (10% of training data)
        test_texts = []
        test_token_count = 0
        max_test_tokens = max_tokens // 10  # 10% of training data for testing
        print(f"Loading C4 validation data (target: {max_test_tokens:,} tokens)...")
        
        for item in testdata:
            text = item['text']
            if text and text.strip():  # Skip empty texts
                # Tokenize this individual text to count tokens
                tokens = tokenizer_local(text, return_tensors='pt')
                if isinstance(tokens, dict):
                    token_count = tokens["input_ids"].shape[1]
                else:
                    token_count = tokens.input_ids.shape[1]
                
                # Check if adding this text would exceed our limit
                if test_token_count + token_count > max_test_tokens:
                    # If this single text would exceed limit, try to take a portion
                    remaining_tokens = max_test_tokens - test_token_count
                    if remaining_tokens > 0:
                        # Take a portion of this text
                        words = text.split()
                        # Rough estimate: take proportional words
                        word_ratio = remaining_tokens / token_count
                        target_words = max(1, int(len(words) * word_ratio))
                        partial_text = " ".join(words[:target_words])
                        test_texts.append(partial_text)
                        test_token_count = max_test_tokens
                    break
                else:
                    test_texts.append(text)
                    test_token_count += token_count
                    
                    if test_token_count >= max_test_tokens:
                        break
        
        print(f"Loaded {len(test_texts)} validation texts with ~{test_token_count:,} tokens")
        
        # Join texts with newlines for final tokenization
        trainenc = tokenizer_local("\n".join(train_texts), return_tensors='pt')
        testenc = tokenizer_local("\n".join(test_texts), return_tensors='pt')
        
        # Clean up to save memory
        del train_texts, test_texts
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose from 'wikitext2' or 'c4'.")

    # Handle both dictionary and object-based tokenizer returns
    if isinstance(trainenc, dict):
        trainenc_input_ids = trainenc["input_ids"]
    else:
        trainenc_input_ids = trainenc.input_ids

    random.seed(seed)
    trainloader = []
    # Calculate the number of samples that can be generated
    total_samples = (trainenc_input_ids.shape[1] - seqlen) // seqlen
    # Limit to the requested number of samples
    max_samples = min(nsamples, total_samples)
    for i in range(0, max_samples * seqlen, seqlen):
        j = i + seqlen
        inp = trainenc_input_ids[:, i:j]
        tar = inp.clone()
        trainloader.append((inp, tar))
    
    # Explicitly delete large intermediate tensor if no longer needed directly
    # trainenc might be large. testenc is returned, so keep it.
    del inp, tar # from the loop
    if 'trainenc' in locals() and trainenc is not None:
        del trainenc

    return trainloader, testenc


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model=''
):
    if 'wikitext2' in name:
        return get_training_dataset(seed, seqlen, model, 'wikitext2', nsamples)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model)
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)
        return get_training_dataset(seed, seqlen, model, 'c4', nsamples)

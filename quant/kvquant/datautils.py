import numpy as np
import torch
from transformers import AutoTokenizer
import random

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

def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Use the robust tokenizer loading
    model_type = detect_model_type(model)
    tokenizer = load_tokenizer(model, model_type)
    
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Handle both dictionary and object-based tokenizer returns
    if isinstance(trainenc, dict):
        trainenc_input_ids = trainenc["input_ids"]
    else:
        trainenc_input_ids = trainenc.input_ids

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc_input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc_input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    # Use the robust tokenizer loading
    model_type = detect_model_type(model)
    tokenizer = load_tokenizer(model, model_type)
    
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    # Handle both dictionary and object-based tokenizer returns
    if isinstance(trainenc, dict):
        trainenc_input_ids = trainenc["input_ids"]
    else:
        trainenc_input_ids = trainenc.input_ids

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc_input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc_input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', use_auth_token=False
    )
    valdata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation', use_auth_token=False
    )

    # Use the robust tokenizer loading
    model_type = detect_model_type(model)
    tokenizer = load_tokenizer(model, model_type)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            # Handle both dictionary and object-based tokenizer returns
            if isinstance(trainenc, dict):
                trainenc_input_ids = trainenc["input_ids"]
            else:
                trainenc_input_ids = trainenc.input_ids
            if trainenc_input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc_input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc_input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            # Handle both dictionary and object-based tokenizer returns
            if isinstance(tmp, dict):
                tmp_input_ids = tmp["input_ids"]
            else:
                tmp_input_ids = tmp.input_ids
            if tmp_input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp_input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp_input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc

def get_ptb_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    # Use the robust tokenizer loading
    model_type = detect_model_type(model)
    tokenizer = load_tokenizer(model, model_type)
    
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    # Handle both dictionary and object-based tokenizer returns
    if isinstance(trainenc, dict):
        trainenc_input_ids = trainenc["input_ids"]
    else:
        trainenc_input_ids = trainenc.input_ids

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc_input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc_input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    # Use the robust tokenizer loading
    model_type = detect_model_type(model)
    tokenizer = load_tokenizer(model, model_type)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            # Handle both dictionary and object-based tokenizer returns
            if isinstance(trainenc, dict):
                trainenc_input_ids = trainenc["input_ids"]
            else:
                trainenc_input_ids = trainenc.input_ids
            if trainenc_input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc_input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc_input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    # Handle both dictionary and object-based tokenizer returns
    if isinstance(valenc, dict):
        valenc_input_ids = valenc["input_ids"]
    else:
        valenc_input_ids = valenc.input_ids
    valenc_input_ids = valenc_input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc_input_ids)

    return trainloader, valenc

def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model=''
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model)
        return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model)
        return get_c4(nsamples, seed, seqlen, model)

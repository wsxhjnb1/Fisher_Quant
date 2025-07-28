import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.optim as optim
import transformers
from torch.utils.data import Dataset
from transformers import Trainer

from tqdm import tqdm
import utils

# from memory_profiler import profile

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

PROMPT_DICT_NEW={
    "prompt":(
        "Below is a text imitation task. You will be given a text description and asked to rewrite it in a different style.\n\n"
        "### Input:\n{input}\n\n### Output:"
    )
}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    load: Optional[str] = field(default="")
    group_size: int = field(default=0, metadata={"help": "Group size for normalization"})
    #save_grad_path: str = field(
    #    metadata={"help": "Path to save the gradients"}
    #)


@dataclass
class DataArguments:
    dataset: str = field(default="c4")
    num_examples: int = field(default=16, metadata={"help": "Number of calibration examples"})
    seqlen: int = field(default=2048)
    maxseqlen: int = field(default=32768)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


# Global storage for normalized gradients
normalized_grads = {}

class GroupNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group_size, norm_type, layer_name):
        B, S, H = x.shape
        
        if norm_type == 'sequence':
            # Keys: sequence-wise grouping
            if group_size > 0 and S >= group_size:
                G = S // group_size
                x_to_group = x[:, :G * group_size, :]
                x_g = x_to_group.reshape(B, G, group_size, H)
                x_rest = x[:, G * group_size:, :]
                
                mn = x_g.amin(dim=2, keepdim=True)
                mx = x_g.amax(dim=2, keepdim=True)
                scale = (mx - mn).clamp(min=1e-6)
                x_norm = (x_g - mn) / scale
                
                # Denormalize for forward pass
                x_denorm = x_norm * scale + mn
                denormalized_part = x_denorm.reshape(B, G * group_size, H)
                output = torch.cat([denormalized_part, x_rest], dim=1)
                
                # Store for backward
                ctx.save_for_backward(x_norm.reshape(B, G * group_size, H), mn, scale)
                ctx.group_info = (G, group_size, S, H, norm_type, layer_name)
            else:
                mn = x.amin(dim=1, keepdim=True)
                mx = x.amax(dim=1, keepdim=True)
                scale = (mx - mn).clamp(min=1e-6)
                x_norm = (x - mn) / scale
                
                # Denormalize for forward pass
                output = x_norm * scale + mn
                
                # Store for backward
                ctx.save_for_backward(x_norm, mn, scale)
                ctx.group_info = (None, None, S, H, norm_type, layer_name)
                
        else:  # token-wise for values
            if group_size > 0 and H > 0 and H % group_size == 0:
                G = H // group_size
                x_g = x.reshape(B, S, G, group_size)
                
                mn = x_g.amin(dim=-1, keepdim=True)
                mx = x_g.amax(dim=-1, keepdim=True)
                scale = (mx - mn).clamp(min=1e-6)
                x_norm = (x_g - mn) / scale
                
                # Denormalize for forward pass
                x_denorm = x_norm * scale + mn
                output = x_denorm.reshape(B, S, H)
                
                # Store for backward
                ctx.save_for_backward(x_norm.reshape(B, S, H), mn, scale)
                ctx.group_info = (G, group_size, S, H, norm_type, layer_name)
            else:
                mn = x.amin(dim=-1, keepdim=True)
                mx = x.amax(dim=-1, keepdim=True)
                scale = (mx - mn).clamp(min=1e-6)
                x_norm = (x - mn) / scale
                
                # Denormalize for forward pass
                output = x_norm * scale + mn
                
                # Store for backward
                ctx.save_for_backward(x_norm, mn, scale)
                ctx.group_info = (None, None, S, H, norm_type, layer_name)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x_norm, mn, scale = ctx.saved_tensors
        G, group_size, S, H, norm_type, layer_name = ctx.group_info
        
        # Correctly calculate gradient w.r.t. x_norm: dL/d(x_norm) = grad_output * scale
        if G is not None:  # Grouped case
            if norm_type == 'sequence':
                # Reshape grad_output to match grouping
                grad_output_g = grad_output.reshape(-1, G, group_size, H)
                # Calculate grad w.r.t normalized groups
                grad_x_norm_g = grad_output_g * scale
                grad_x_norm = grad_x_norm_g.reshape(-1, S, H)
            else:  # Token grouping
                grad_output_g = grad_output.reshape(-1, S, G, group_size)
                grad_x_norm_g = grad_output_g * scale
                grad_x_norm = grad_x_norm_g.reshape(-1, S, H)
        else:  # One-group case
            grad_x_norm = grad_output
        
        global normalized_grads
        normalized_grads[layer_name] = grad_x_norm.detach()
        
        # Return gradient for original input. Since y=x, dy/dx=1, so dL/dx = dL/dy.
        return grad_output, None, None, None


class GroupNormHook:
    def __init__(self, group_size, norm_type, layer_name):
        self.group_size = group_size
        self.norm_type = norm_type
        self.layer_name = layer_name
        
    def __call__(self, module, input, output):
        return GroupNormFunction.apply(output, self.group_size, self.norm_type, self.layer_name)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    print("[func] make_supervised_data_module")
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def get_modules(layer):
    # NOTE: This is llama-specific
    # For other models, replace this with proper names for all linear layers
    return[
        layer.self_attn.q_proj,
        layer.self_attn.k_proj,
        layer.self_attn.v_proj,
        layer.self_attn.o_proj,
        layer.mlp.gate_proj,
        layer.mlp.up_proj,
        layer.mlp.down_proj,
    ]

def get_modules_kv(layer):
    # NOTE: This is llama-specific
    # For other models, replace this with proper names for all linear layers
    return[
        layer.self_attn.k_proj,
        layer.self_attn.v_proj,
    ]

# @profile
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.dataset == "c4":
        from datautils import get_loaders
        print("Calibration with C4 ")
        dataloader, testloader = get_loaders(data_args.dataset,  model=model_args.model_name_or_path, seqlen=data_args.seqlen, seed=0)
    elif data_args.dataset == "wikitext2":
        from datautils import get_loaders
        print("Calibration with Wikitext2 ")
        dataloader, testloader = get_loaders(data_args.dataset,  model=model_args.model_name_or_path, seqlen=data_args.seqlen, seed=0)
    else:
        raise NotImplementedError("Please define your own dataset here")


    # Set RoPE scaling factor
    import math
    from transformers import AutoConfig
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    context_size = data_args.maxseqlen
    orig_ctx_len = getattr(config, "max_position_embeddings", None) # this value should be 4096 for LLaMA2 models
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    config._flash_attn_2_enabled = True

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True
    )

#    model.seqlen = seqlen  #TODO
    if config.vocab_size == 32001:
        model.resize_token_embeddings(32001)

    model = model.bfloat16()
    model.cuda()

    if model_args.load != "":
        model.load_state_dict(torch.load(model_args.load), strict=False)
        model.eval()

    # For other models, replace this with proper variable names for model and layers
    _model = model.model
    _layers = _model.layers
    
    grads = {}

    # Register forward hooks for group normalization
    for i, layer in enumerate(_layers):
        k_proj, v_proj = get_modules_kv(layer)
        
        # Register forward hooks for group normalization and gradient capture
        # Keys: sequence-wise grouping
        k_proj.register_forward_hook(GroupNormHook(model_args.group_size, 'sequence', f'k_proj{i}'))
        # Values: token-wise grouping  
        v_proj.register_forward_hook(GroupNormHook(model_args.group_size, 'token', f'v_proj{i}'))

    # main loop
    for i, data in tqdm(enumerate(dataloader[:data_args.num_examples])):
        data = data[0]
        x = data.cuda()

        model.zero_grad()
        # compute gradients
        outputs = model(input_ids=x, labels=x)
        loss = outputs.loss
        loss.backward()

        # get grads from normalized values
        global normalized_grads
        for i, layer in enumerate(_layers):
            print(f'weight layer {i}')
            
            if f'k_proj{i}' in normalized_grads:
                kgrad = (normalized_grads[f'k_proj{i}'] ** 2).float().cpu()
                if f'k_proj{i}' not in grads:
                    grads[f'k_proj{i}'] = kgrad
                else:
                    grads[f'k_proj{i}'] = torch.cat((grads[f'k_proj{i}'], kgrad), dim=1)
            
            if f'v_proj{i}' in normalized_grads:
                vgrad = (normalized_grads[f'v_proj{i}'] ** 2).float().cpu()
                if f'v_proj{i}' not in grads:
                    grads[f'v_proj{i}'] = vgrad
                else:
                    grads[f'v_proj{i}'] = torch.cat((grads[f'v_proj{i}'], vgrad), dim=1)
        
        # Clear normalized grads for next iteration
        normalized_grads.clear()

    ## This is a hacky solution to save the gradients
    # where we overwrite all the weights in the model as the gradients
    # and use HF save_pretrained`
    for i, layer in enumerate(_layers):
        k_proj, v_proj = get_modules_kv(layer)
        if f'k_proj{i}' in grads:
            k_proj.weight.data = grads[f'k_proj{i}']
        if f'v_proj{i}' in grads:
            v_proj.weight.data = grads[f'v_proj{i}']

    print(f"saving model gradient at {training_args.output_dir}")
    model.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    train()

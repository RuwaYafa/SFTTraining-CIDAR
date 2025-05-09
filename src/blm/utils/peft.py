import logging
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)

logger = logging.getLogger(__name__)


def create_and_prepare_model(args):
    config = transformers.AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True, token=args.token)
    quantization_config = None

    if args.quantize:
        logger.info("Load quantization config")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        # cache_dir="/content/hugging_face/", #rfa
<<<<<<< HEAD
=======

>>>>>>> 057669b0ec0ddd915c64433e12622e191c8f3b46
        config=config,
        attn_implementation=args.attn_implementation,
        quantization_config=quantization_config,
        token=args.token,
<<<<<<< HEAD
        # torch_dtype=torch.bfloat16, #RFA
=======
>>>>>>> 057669b0ec0ddd915c64433e12622e191c8f3b46
        device_map=None if args.deepspeed else 'auto'
    )

    # find all linear modules in model for lora
    target_modules = find_all_linear_names(model)

    # create lora config
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )

    # pre-process the model by upcasting the layer norms in float 32 for
    # Adapted from https://github.com/tmm1/axolotl/blob/2eda9e02a9d15a7a3f92b41f257d9844d72fc220/src/axolotl/utils/models.py#L338
    logger.info("pre-processing model for peft")
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.bfloat16)
        if any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                module = module.to(torch.bfloat16)

    # initialize peft model
    logger.info("initializing peft model")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    
    # enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        
    model = get_peft_model(model, peft_config)

    # logger.info parameters
    model.print_trainable_parameters()

    # tokenizer
<<<<<<< HEAD
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=args.token) 
    print("--------------PEFT Chat Template-------------------", tokenizer.chat_template)
    # tokenizer = AutoTokenizer.from_pretrained("LlamaTokenizer", token=args.token) 
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=args.token, trust_remote_code=True, add_eos_token=True) #rfa

    #Start RFA
    """ To solve this warning:
    UserWarning: The pad_token_id and eos_token_id values of this tokenizer are identical. 
    If you are planning for multi-turn training, it can result in 
    the model continuously generating questions and answers without eos token. 
    To avoid this, set the pad_token_id to a different value.
    """
    if tokenizer.pad_token_id == tokenizer.eos_token_id: #---------------rfa back------------
      print("pad_token_id and eos_token_id are the same. Setting pad_token_id to eos_token_id + 1.")
      tokenizer.pad_token_id = tokenizer.eos_token_id + 1
    #End RFA

    # tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer

def find_all_linear_names(model):
    """
    y = xW^T + b
      You would typically use torch.nn.Linear layers for tasks like:

        Mapping the output of one layer to the input of the next.
        Performing classification by mapping the final hidden layer to the number of classes.
        Regression by mapping to a single output value.

    """
    cls = torch.nn.Linear
    # print("___cls.peft______________________", cls) #rfa
=======
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=args.token)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=args.token, trust_remote_code=True) #rfa

    tokenizer.pad_token = tokenizer.eos_token

    return model, peft_config, tokenizer


def find_all_linear_names(model):
    cls = torch.nn.Linear
>>>>>>> 057669b0ec0ddd915c64433e12622e191c8f3b46
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
<<<<<<< HEAD
            # print("___names after cls.peft______________________", names) #rfa

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    # print("___lora_module_names after cls.peft______________________", lora_module_names) #rfa

=======

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
>>>>>>> 057669b0ec0ddd915c64433e12622e191c8f3b46
    return list(lora_module_names)

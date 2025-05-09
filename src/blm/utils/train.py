<<<<<<< HEAD
from accelerate.utils.other import encode
from datasets import load_from_disk
import logging
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from blm.utils.peft import create_and_prepare_model
from blm.utils.prompter import Prompter
from transformers import AutoTokenizer #RFA
from transformers import TrainerCallback, TrainingArguments, Trainer #RFA

=======
from datasets import load_from_disk
import logging
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from blm.utils.peft import create_and_prepare_model
from blm.utils.prompter import Prompter
>>>>>>> 057669b0ec0ddd915c64433e12622e191c8f3b46

logger = logging.getLogger(__name__)


def train(args):
    # Load and create peft model
    model, peft_config, tokenizer = create_and_prepare_model(args)
<<<<<<< HEAD
    model.config.use_cache = False ####?? RFA

    # print("_________tokenizer before map_______________________________________________________________", tokenizer)
    # print("_________tokenizer.chat_template before map____________________________________________")
    print(tokenizer.chat_template)

    # print("________________________________________________________________________")

    prompter = Prompter(tokenizer)
    tokenizer = prompter.tokenizer #_rfa_ 
    dataset = load_from_disk(args.data_path)
    dataset = dataset.map(prompter, batched=True)
    
    print('========= dataset after mapping ==========', dataset['train'][0]["prompt"])
    # print('========= Prompt // dataset after mapping ==========', dataset['train'][0]["prompt"])

    # print('_________________dataset - train', dataset['train'][0]['messages'])
    # print('_________________dataset - eval', dataset['eval'][0]['messages'])
=======
    model.config.use_cache = False

    prompter = Prompter(tokenizer)
    dataset = load_from_disk(args.data_path)
    dataset = dataset.map(prompter, batched=True)
>>>>>>> 057669b0ec0ddd915c64433e12622e191c8f3b46

    logger.info(f"Pre-saving tokenizer to {args.output_dir}")
    tokenizer.save_pretrained(args.output_dir)

<<<<<<< HEAD

    # print(f"Beginning of sentence token: {tokenizer.bos_token}, ID: {tokenizer.bos_token_id}")
    # print(f"End of sentence token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")
    # print(f"Padding token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
    # print(f"Unknown token: {tokenizer.unk_token}, ID: {tokenizer.unk_token_id}")
    # print(f"Mask token: {tokenizer.mask_token}, ID: {tokenizer.mask_token_id}")
    # print(f"Separator token: {tokenizer.sep_token}, ID: {tokenizer.sep_token_id}")
    # print(f"Classification token: {tokenizer.cls_token}, ID: {tokenizer.cls_token_id}")
    # print(f"Additional special tokens: {tokenizer.additional_special_tokens}, IDs: {tokenizer.additional_special_tokens_ids}")


=======
>>>>>>> 057669b0ec0ddd915c64433e12622e191c8f3b46
    if peft_config:
        logger.info(f"Pre-saving adapter config to {args.output_dir}")
        peft_config.save_pretrained(args.output_dir)

    if hasattr(model, "config"):
        logger.info(f"Pre-saving model config to {args.output_dir}")
        model.config.save_pretrained(args.output_dir)

<<<<<<< HEAD
    # print("______before collator________ prompter.response_template:", prompter.response_template)
    # print("______before collator________ prompter.instruction_template:", prompter.instruction_template)
    # print("______before collator________ tokenizer:", tokenizer)
    # print("______before collator________ prompter:", prompter)

    # print("______tokenizer.chat_template after mapping______", tokenizer.chat_template)

# Mohd
    # if tokenizer.eos_token is None:
    #     tokenizer.eos_token = "</s>" # Or your appropriate EOS token
    #end Mohd

    # prompter_response_template = "<|assistant|>\n"
    # prompter_instruction_template = "<|system|>\n" # Or consider "<|user|>\n" depending on your masking strategy
    # response_template="<|assistant|>",#del
    # instruction_template="<|user|>",#del

    # print("______before collator________ tokenizer.response_template:", prompter.response_template)
    # print("______before collator________ tokenizer.instruction_template:", prompter.instruction_template)

    collator = DataCollatorForCompletionOnlyLM(
                                              response_template=prompter.response_template, 
                                              instruction_template=prompter.instruction_template,
                                              tokenizer=tokenizer, 
                                              mlm=False)
    # print(f"Actual assistant prefix in data: '{dataset['train'][0]['prompt'].split('<|assistant|>')[0] + '<|assistant|>'}'")

#start rfa
    # print('========= Prompt // after collator // dataset after mapping ==========', dataset['train'][0]["prompt"])
    
    sample_prompt = prompter._for_sft(dataset["train"][0]["messages"])
    print("GENERATED PROMPT:", repr(sample_prompt))

    # sample = dataset["train"][0]["prompt"]
    # collated = collator([tokenizer(sample, return_tensors="pt")])
    # print("Collated input IDs:", collated["input_ids"])
    # print("Collated labels:", collated["labels"])  # Should mask everything before <|assistant|>\n

    #endrfa
    # print("______after collator________ prompter.response_template:", prompter.response_template)
    # print("______after collator________ prompter.instruction_template:", prompter.instruction_template)
    # print("______after collator________ prompter:", tokenizer)
    # print("______after collator________ prompter:", prompter)
    # print("______tokenizer.chat_template______", tokenizer.chat_template)

    # #Start RFA
    # training_args = SFTConfig(
    #     output_dir= args.output_dir,
    #     dataset_text_field="prompt", # Replace with your actual column name (your_text_column_name)
    # #     tokenizer=tokenizer,
    #     max_seq_length=args.max_seq_length,
    #     args=args
    # )
    #End RFA

    # dataset["train"] = prompter(dataset["train"])  #_rfa_
    # dataset["eval"] = prompter(dataset["eval"])  #_rfa_
    # print("========= after jinja template - prompter =========") #_rfa_
    # print(dataset) #_rfa_
    # prompter._for_sft(dataset["train"]) #_rfa_
=======
    collator = DataCollatorForCompletionOnlyLM(response_template=prompter.response_template, 
                                               instruction_template=prompter.instruction_template, 
                                               tokenizer=tokenizer, 
                                               mlm=False)
>>>>>>> 057669b0ec0ddd915c64433e12622e191c8f3b46

    trainer = SFTTrainer(
        model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
<<<<<<< HEAD
        dataset_text_field="prompt", # move to training_args
        tokenizer=tokenizer,# move to training_args
        max_seq_length=args.max_seq_length, # move to training_args
        args=args,
        # args=training_args, #RFA
=======
        dataset_text_field="prompt",
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        args=args,
>>>>>>> 057669b0ec0ddd915c64433e12622e191c8f3b46
        data_collator=collator
    )

    logger.info("Model parameters...")
    trainer.model.print_trainable_parameters()

    # Start training
    trainer.train()

import logging
import argparse
from datasets import Dataset, DatasetDict, load_dataset, Features, Value, Sequence, ClassLabel #mon
from blm.utils.helpers import logging_config
import pyarrow as paPyright #mon

logger = logging.getLogger(__name__)


def save_dataset(dataset, output_path, n):
    """
    Convert the dataset into messages as follows, each training example will
    follow the following format:
      {
        "messages": [
                    {"content": "<SYSTEMPROMPT>", "role": "system"}, # it depends on the model prompt structure
                    {"content": "<INSTRUCTIONS>", "role": "user"},
                    {"content": "<RESPONSE>", "role": "assistant"}
                    ]
      }
    :param dataset: Dataset
    :param output_path: str - output path to save dataset to
    :param n: int - number of training examples to sample
    :return: Dataset
    """

    # Open the system prompt file and read its content #mon
<<<<<<< HEAD
    with open('/content/drive/MyDrive/SFTTraining-CIDAR-main/src/blm/prompts/system_prompt.txt', 'r') as file:
        system_prompt = file.read()

    # Open the user prompt file and read its content #mon
    with open('/content/drive/MyDrive/SFTTraining-CIDAR-main/src/blm/prompts/user_prompt.txt', 'r') as file:
=======
    with open('/content/SFTTraining-CIDAR/src/blm/prompts/system_prompt.txt', 'r') as file:
        system_prompt = file.read()

    # Open the user prompt file and read its content #mon
    with open('/content/SFTTraining-CIDAR/src/blm/prompts/user_prompt.txt', 'r') as file:
>>>>>>> 057669b0ec0ddd915c64433e12622e191c8f3b46
        user_prompt = file.read()

    # # Mohd
    # train_messages = [{"messages": [{"content": e["instruction"], "role": "user"}, 
    #                                 {"content": e["output"], "role": "assistant"}]} 
    #                   for e in dataset["train"]]
    # eval_messages = [{"messages": [{"content": e["instruction"], "role": "user"}, 
    #                                {"content": e["output"], "role": "assistant"}]} 
    #                  for e in dataset["test"]]

    # RFA-> for ability to use in most models:
    train_messages = [{"messages": [{"content": system_prompt, "role": "system"}, 
                                    # If we have multi class just add them as , instruction=e['instruction'] etc..
                                    {"content": user_prompt.format(instruction=e['instruction']), "role": "user"}, 
<<<<<<< HEAD
                                    {"content": str(e["output"]), "role": "assistant"}]} 
                      for e in dataset["train"]
                      ]
  
    eval_messages= [{"messages": [{"content": system_prompt, "role": "system"}, 
                                  {"content": user_prompt.format(instruction=e['instruction']), "role": "user"}, 
                                  {"content": str(e["output"]), "role": "assistant"}]} 
                    for e in dataset["test"]
                    ]
    #End RFA
    print("Train message",train_messages[0])
    print("eval message",eval_messages[0])

    #mon 
    # features = Features({
    #     "messages": Sequence({
    #         "content": Value("string"),
    #         "role": Value("string")
    #     })
    # })
    
    # ds = DatasetDict({
    #     "train": Dataset.from_list(train_messages[:n]),# features),
    #     "eval": Dataset.from_list(eval_messages[:int(n*0.2)])#, features)
    # })

    #mon
    ds = DatasetDict({
        # "train": Dataset.from_list(train_messages[:n]),
        # "eval": Dataset.from_list(eval_messages[:int(n*0.2)])

        "train": Dataset.from_list(train_messages[:n]), #rfa
        "eval": Dataset.from_list(eval_messages[:int(n*0.2)]) #rfa
    })

    print("dsdsds", ds["train"][0])
=======
                                    {"content": e["output"], "role": "assistant"}]} 
                      for e in dataset["train"]
                      ]
    eval_messages= [{"messages": [{"content": system_prompt, "role": "system"}, 
                                  {"content": user_prompt.format(instruction=e['instruction']), "role": "user"}, 
                                  {"content": e["output"], "role": "assistant"}]} 
                    for e in dataset["test"]
                    ]
    #End RFA

    #mon 
    features = Features({
        "messages": Sequence({
            "content": Value("string"),
            "role": Value("string")
        })
    })
    
    ds = DatasetDict({
        "train": Dataset.from_list((train_messages[:n]), features),
        "eval": Dataset.from_list((eval_messages[:int(n*0.1)]), features)
    })

    #mon
    # ds = DatasetDict({
    #     "train": Dataset.from_list(train_messages, features=features),
    #     "eval": Dataset.from_list(eval_messages, features=features)
    # })
>>>>>>> 057669b0ec0ddd915c64433e12622e191c8f3b46

    ds.save_to_disk(output_path)
    return ds


def main(args):
    dataset = load_dataset("arbml/CIDAR") # change the Dataset / from huggingface or from data folder! #mon ds: #Kamyar-zeinalipour/ArabicSense
    dataset = dataset['train'].train_test_split(test_size=0.2)
<<<<<<< HEAD
    # print("Dataset____",dataset)
    # print("end dataset")
    # print("Dataset____",dataset["train"])
    # print("Dataset____",dataset["test"])

    # dataset = load_dataset(
    # "csv", 
    # data_files={
    #     "train": "/content/drive/MyDrive/SFTTraining-CIDAR-main/data/train_data.csv", 
    #     "test": "/content/drive/MyDrive/SFTTraining-CIDAR-main/data/test_data.csv", 
    #     # "validation": "/content/drive/MyDrive/LLMTraining-main/data/task1_validation.csv"
    # }
    # )
    

=======
    
>>>>>>> 057669b0ec0ddd915c64433e12622e191c8f3b46
    ds = save_dataset(dataset, args.output_path, args.n)
    logger.info(f"Total training examples: {len(ds['train'])}")
    logger.info(f"Total eval examples: {len(ds['eval'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, help="Data output path")
<<<<<<< HEAD
    parser.add_argument("--n", type=int, default=5, help="Number of training examples to sample")
    args = parser.parse_args()

    logging_config("/content/drive/MyDrive/SFTTraining-CIDAR-main/processing.log")
=======
    parser.add_argument("--n", type=int, default=500, help="Number of training examples to sample")
    args = parser.parse_args()

    logging_config("processing.log")
>>>>>>> 057669b0ec0ddd915c64433e12622e191c8f3b46

    main(args)

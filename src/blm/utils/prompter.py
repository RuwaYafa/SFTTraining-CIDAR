class Prompter:
    def __init__(self, tokenizer):
        templates = {
            "mistralai": {"instruction_template": "[INST]",
<<<<<<< HEAD
                           "response_template": "[/INST]"},
            "meta-llama": {"instruction_template": "<|start_header_id|>system<|end_header_id|>", 
                           "response_template": "<|start_header_id|>assistant<|end_header_id|>"},
            "microsoft": {"instruction_template": "<|system|>",
                            "response_template": "<|assistant|>"},
            "default": {"instruction_template": "###Instructions:\n\n",
                          "response_template": "###Assistant:\n\n"},
            "acegpt": {"instruction_template": "<|user|>\n",
                       "response_template": "<|assistant|>\n"}, # RFA Added AceGPT-7B template // diff with mon without system
            # "gpt": {"instruction_template": "<user>",
            #         "response_template": "<assistant>"}
=======
                          "response_template": "[/INST]"},
            "meta-llama": {"instruction_template": "<|start_header_id|>system<|end_header_id|>", 
                           "response_template": "<|start_header_id|>assistant<|end_header_id|>"},
            "microsoft": {"instruction_template": "<|system|>", 
                           "response_template": "<|assistant|>"},
            "default": {"instruction_template": "###Instructions:\n\n",
                          "response_template": "###Assistant:\n\n"},
>>>>>>> 057669b0ec0ddd915c64433e12622e191c8f3b46
        }

        self.tokenizer = tokenizer
        self.model_family = self.tokenizer.name_or_path.split("/")[0]
<<<<<<< HEAD
        self.instruction_template = templates.get(self.model_family, templates["acegpt"])["instruction_template"]
        self.response_template = templates.get(self.model_family, templates["acegpt"])["response_template"]
        # print ("self.tokenizer.name_or_path", self.tokenizer.name_or_path)
        # print("self.instruction_template____________", self.instruction_template)
        # print("self.response_template____________", self.response_template)
        # print("tokenizer____________", self.tokenizer)
=======
        self.instruction_template = templates.get(self.model_family, templates["default"])["instruction_template"]
        self.response_template = templates.get(self.model_family, templates["default"])["response_template"]
>>>>>>> 057669b0ec0ddd915c64433e12622e191c8f3b46

    def __call__(self, data):
        """Prepare input for model training or inference.
        Pass data to generate prompts for training
        Pass system and user to generate one time prompt for a specific model
        based on the model ID in the tokenizer.

        Args:
<<<<<<< HEAD

            data (DatasetDict): dataset that should contains prompt
                                    components (system, instructions, data and output)
                                    Use this option when generating training data
            system (str): system prompt
            user (str): user prompt
        """

        # MK
        # data["prompt"] = [self._for_sft(messages) for messages in data["messages"]]
        # print("Data prompt____________", data["prompt"])

        # Mon
        # print("Inside Prompter.call:")
        # print("Type of data:", type(data))
        # print("Keys of data:", data.keys())
        # print("First few messages:", data["messages"][:2] if "messages" in data else None)

        # data["prompt"] = [self._for_sft(messages) for messages in data["messages"]]

        # print(f"Type of prompt: {type(data)}")
        # print("Keys of the data dictionary after adding 'prompt':", data.keys())
        # print( data["prompt"][0])
        #end mon
        
        prompts = [self._for_sft(messages) for messages in data["messages"]]
        data["prompt"] = prompts
        # print("____________First few prompts generated:", data["prompt"][:2])
        # print("____________Data in prompter!!", data)



        return data #mon

=======
            data (DatasetDict): dataset that should contains prompt 
                                components (system, instructions, data and output)
                                Use this option when generating training data
            system (str): system prompt
            user (str): user prompt
        """
        data["prompt"] = [self._for_sft(messages) for messages in data["messages"]]
    
>>>>>>> 057669b0ec0ddd915c64433e12622e191c8f3b46
    def _for_sft(self, messages):
        """
        Convert the list of messages into a prompt by injecting the LLM special
        instruction and assistant tokens.
        :param messages: List[Dict] - list of user and assistant messages
        :return: str - prompt used as input to model training
        """
<<<<<<< HEAD

        # #start RFA
        # Multi turn
        # # Assuming 'tokenizer' is your loaded tokenizer object
        # if self.tokenizer.chat_template is None:
            # self.tokenizer.chat_template = """
            # {% if message['role'] == 'user' %}
            # {{ '<|user|>' + message['content'] + eos_token }}
            # {% elif message['role'] == 'system' %}
            # {{ '<|system|>' + message['content'] + eos_token }}
            # {% elif message['role'] == 'assistant' %}
            # {{ '<|assistant|>'  + message['content'] + eos_token }}
            # {% endif %}
            # {% if loop.last and add_generation_prompt %}
            # {{ '<|assistant|>' }}
            # {% endif %}
            # {% endfor %}
            # """
        # self.tokenizer.chat_template = """\
        # {% for message in messages %}
        # {% if message['role'] == 'system' %}<|system|>\n{{ message['content'] }}</s>
        # {% elif message['role'] == 'user' %}<|user|>\n{{ message['content'] }}</s>
        # {% elif message['role'] == 'assistant' %}<|assistant|>\n{{ message['content'] }}</s>
        # {% endif %}
        # {% endfor %}"""      

        self.tokenizer.chat_template = """
        {% for message in messages %}
            {% if message['role'] == 'system' %}
                {{ '<|system|>\n' + message['content'] + eos_token + '\n' }}
            {% elif message['role'] == 'user' %}
                {{ '<|user|>\n' + message['content'] + eos_token + '\n' }}
            {% elif message['role'] == 'assistant' %}
                {{ '<|assistant|>\n' + message['content'] + eos_token + '\n' }}
            {% endif %}
        {% endfor %}
        """

            # RFA for Ace while the above for Tiny Llama
            # {% for message in messages %}
            # {% if message['role'] == 'system' %}
            # {{ '<|system|>\n' + message['content'] + '</s>' }}
            # {% elif message['role'] == 'user' %}
            # {{ '<|user|>\n' + message['content'] + '</s>' }}
            # {% elif message['role'] == 'assistant' %}
            # {{ '<|assistant|>\n'  + message['content'] + '</s>' }}
            # {% endif %}
            # {% endfor %}
            
        #     print("_______________Chat template set for the tokenizer.")
        # else:
        #     print("_______________Tokenizer already has a chat template:", self.tokenizer.chat_template)
        #End RFA

        # print("msg_______________",messages)
        # prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = False) #

        # print("prompt____________", prompt)

        return prompt



=======
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        return prompt
>>>>>>> 057669b0ec0ddd915c64433e12622e191c8f3b46

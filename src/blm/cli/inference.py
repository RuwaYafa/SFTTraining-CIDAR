import argparse
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import os

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
<<<<<<< HEAD

    # parser.add_argument('--token', type=str, required=True, help="Hugging Face API Token")
    # parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model")
    # parser.add_argument('--model_name', type=str, required=True, help="Pre-trained model name on Hugging Face (e.g., 'meta-llama/Llama-3.2-1B-Instruct')")
    
    #RFA1
    parser.add_argument('--token', type=str, default="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", help="Hugging Face API Token")
    parser.add_argument('--model_path', type=str, default="/content/drive/MyDrive/SFTTraining-CIDAR-main/merge-lora-SFTTraining-CIDAR", help="Path to the trained model")
    parser.add_argument('--model_name', type=str, default="FreedomIntelligence/AceGPT-7B-chat", help="Pre-trained model name on Hugging Face (e.g., 'meta-llama/Llama-3.2-1B-Instruct')")
    parser.add_argument('--instruction', type=str, help="The instruction you want the model to answer")
    
    #EndRFA1
=======
    parser.add_argument('--token', type=str, required=True, help="Hugging Face API Token")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model")
    parser.add_argument('--model_name', type=str, required=True, help="Pre-trained model name on Hugging Face (e.g., 'meta-llama/Llama-3.2-1B-Instruct')")
>>>>>>> 057669b0ec0ddd915c64433e12622e191c8f3b46
    
    args = parser.parse_args()
    
    hf_token = args.token  # Use the token provided in the argument
    
    if not hf_token:
        raise ValueError("Hugging Face token is required")

    # Define the chat template
    SYSTEMPROMPT = (
        "You are an AI assistant, an expert in Arabic culture."
<<<<<<< HEAD
    )

    # formatted_instructions = (
    #     "Given the following question to answer: {instruction}\n"
    #     "اشرح لي لماذا يجب على شخص يريد إعادة تدوير كتبه القديمة التبرع بها لمكتبة."
    # )

    # formatted_instructions = (
    #     "Given the following question to answer: {instruction}\n"
    #     "اقترح اسماً لعلامة تجارية لبيع العطور"
    # )    

    #RFA2
    # Get the user-provided instruction
    if args.instruction is None:
        user_instruction = input("Enter your question about Arabic culture: ")
    # user_instruction = args.instruction

    # Format the instructions with the user's input
    formatted_instructions = (
        f"Questions relevant to Arabic culture without bias {user_instruction}.\n"
        f"{user_instruction}"
    )
    #EndRFA2
=======
        "You are an expert at answering questions that are relevant to Arabic culture."
        "Based on the instructions provided, you can answer the question that the user asks for." 
        "without any bias towards specific perspectives or individuals."
    )

    formatted_instructions = (
        "Given the following question to answer: {instruction}\n"
        "اشرح لي لماذا يجب على شخص يريد إعادة تدوير كتبه القديمة التبرع بها لمكتبة."
    )
>>>>>>> 057669b0ec0ddd915c64433e12622e191c8f3b46

    # Input data for the chat
    input_data = [
        {"content": SYSTEMPROMPT, "role": "system"},
        {"content": formatted_instructions, "role": "user"},
        {"content": "", "role": "assistant"}  # Assistant's response will be generated
    ]

<<<<<<< HEAD
    #RFA3
    # Print the input_data
    print("Input Data:")
    for item in input_data:
        print(f"{item['role']}: {item['content']}")
    print("-" * 20)
    # EndRFA3 

    # Initialize the tokenizer and model
    model_path = args.model_path  # Get the model path from arguments
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)  # Using the model name for tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
=======
    # Initialize the tokenizer and model
    model_path = args.model_path  # Get the model path from arguments
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)  # Using the model name for tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
>>>>>>> 057669b0ec0ddd915c64433e12622e191c8f3b46

    # Manually format the chat input for generation
    chat_text = ""
    for message in input_data:
        chat_text += f"{message['role']}: {message['content']}\n"

    # Create the text generation pipeline
    generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer, device=0)

    # Generate the response
    response = generator(chat_text, max_length=500, num_return_sequences=1, return_full_text=True)

    # Output the result
<<<<<<< HEAD
    print("Generated Response:") #RFA
=======
>>>>>>> 057669b0ec0ddd915c64433e12622e191c8f3b46
    print(response)


if __name__ == "__main__":
    main()

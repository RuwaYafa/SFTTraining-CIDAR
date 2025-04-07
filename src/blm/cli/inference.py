import argparse
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import os

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, required=True, help="Hugging Face API Token")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model")
    parser.add_argument('--model_name', type=str, required=True, help="Pre-trained model name on Hugging Face (e.g., 'meta-llama/Llama-3.2-1B-Instruct')")
    
    args = parser.parse_args()
    
    hf_token = args.token  # Use the token provided in the argument
    
    if not hf_token:
        raise ValueError("Hugging Face token is required")

    # Define the chat template
    SYSTEMPROMPT = (
        "You are an AI assistant, an expert in Arabic culture."
        "You are an expert at answering questions that are relevant to Arabic culture."
        "Based on the instructions provided, you can answer the question that the user asks for." 
        "without any bias towards specific perspectives or individuals."
    )

    formatted_instructions = (
        "Given the following question to answer: {instruction}\n"
        "اشرح لي لماذا يجب على شخص يريد إعادة تدوير كتبه القديمة التبرع بها لمكتبة."
    )

    # Input data for the chat
    input_data = [
        {"content": SYSTEMPROMPT, "role": "system"},
        {"content": formatted_instructions, "role": "user"},
        {"content": "", "role": "assistant"}  # Assistant's response will be generated
    ]

    # Initialize the tokenizer and model
    model_path = args.model_path  # Get the model path from arguments
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)  # Using the model name for tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)

    # Manually format the chat input for generation
    chat_text = ""
    for message in input_data:
        chat_text += f"{message['role']}: {message['content']}\n"

    # Create the text generation pipeline
    generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer, device=0)

    # Generate the response
    response = generator(chat_text, max_length=500, num_return_sequences=1, return_full_text=True)

    # Output the result
    print(response)


if __name__ == "__main__":
    main()

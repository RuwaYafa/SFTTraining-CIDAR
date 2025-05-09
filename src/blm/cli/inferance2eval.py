import argparse
import torch
import pandas as pd
import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Function to generate responses using Hugging Face pipeline
def generate_responses(input_file, output_file, model_name_or_path, token,
                       max_length=512, temperature=0.2, top_p=1.0, top_k=0, 
                       repetition_penalty=1.2, do_sample=True, num_return_sequences=1):

    print("Loading the Hugging Face text-generation pipeline...")

    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=token)##rfa
    # model = AutoModelForCausalLM.from_pretrained(model_name_or_path, token=token)##rfa

    # Load the Hugging Face text-generation pipeline
    generator = pipeline("text-generation", model=model_name_or_path, tokenizer=model_name_or_path, device=0 if torch.cuda.is_available() else -1)
    print("Pipeline loaded successfully.")

    # Read input file (CSV format) using pandas
    print(f"Reading input CSV file: {input_file}")
    # df = pd.read_csv(input_file)
    df = pd.read_json(input_file)
    inputs = df['instruction'].tolist()[:5]  # Assuming the column name is 'instruction'
    print(f"Found {len(inputs)} instructions in the CSV file.")

    # Prepare the output list
    outputs = []
    print("Generating responses...")

    # Loop through each input instruction and generate response
    for idx, input_text in enumerate(inputs):
        print(f"Processing instruction {idx + 1}/{len(inputs)}: {input_text[:50]}...")  # Print first 50 characters for preview
        # Generate response using Hugging Face pipeline
        generated_responses = generator(input_text.strip(), 
                                        max_length=max_length,
                                        temperature=temperature,
                                        top_p=top_p,
                                        top_k=top_k,
                                        repetition_penalty=repetition_penalty,
                                        do_sample=do_sample,
                                        num_return_sequences=num_return_sequences)
        
        # Collect the generated output (generating multiple responses if needed)
        for response in generated_responses:

            # Strip the instruction part from the generated text
            output_text = response["generated_text"]
            output_without_instruction = output_text[len(input_text):].strip()  # Remove instruction part from the output
            print(f"output_without_instruction: {output_without_instruction}")
            outputs.append({
                "instruction": input_text.strip(),
                "generated_output": output_without_instruction ##response["generated_text"]##
            })

    print(f"Generated {len(outputs)} responses.")

    # Write the results to a JSON file
    print(f"Saving results to JSON file: {output_file}")
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)
    print(f"Results saved successfully to {output_file}.")

# Argument parsing for CLI execution
def main():
    print("Starting inference process...")

    parser = argparse.ArgumentParser(description="Generate responses using a language model")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Token for Hugging Face)")
    parser.add_argument("--token", type=str, required=True, help="Path or name of the model (local or Hugging Face)")
    parser.add_argument("--max_length", type=int, default=512, help="Max length for generated sequences (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for randomness (default: 0.2)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for nucleus sampling (default: 1.0)")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k for sampling (default: 0)")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty (default: 1.2)")
    parser.add_argument("--do_sample", type=bool, default=True, help="Whether to sample or not (default: True)")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of sequences to return (default: 1)")

    # Parse the arguments
    args = parser.parse_args()


    # print the DataFrame**
    try:
        # df_to_print = pd.read_csv(args.input_file)
        df_to_print = pd.read_json(args.input_file)

        print("\n--- Input Dataset (First 5 rows) ---")
        print(df_to_print.head())
        print("\n--------------------------------------\n")
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_file}")
        return
    except Exception as e:
        print(f"An error occurred while reading the input file: {e}")
        return


    # Generate responses based on the provided arguments
    generate_responses(
        input_file=args.input_file,
        output_file=args.output_file,
        model_name_or_path=args.model_name_or_path,
        token=args.token,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample,
        num_return_sequences=args.num_return_sequences
    )

if __name__ == "__main__":
    main()

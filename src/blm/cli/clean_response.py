#!/usr/bin/env python3
# extract_response.py - Extracts clean assistant response from model output

import re
from typing import Dict, List

def extract_assistant_response(output: List[Dict[str, str]]) -> str:
    """
    Extracts and cleans the assistant's response from model output
    
    Args:
        output: Model output in Hugging Face format [{'generated_text': ...}]
    
    Returns:
        Cleaned assistant response string
    """
    try:
        # Extract full generated text
        full_text = output[0]['generated_text']
        
        # Use regex to find the assistant's response
        match = re.search(
            r'assistant:\s*(.*?)(?=\nuser:|$)', 
            full_text, 
            re.DOTALL
        )
        
        if not match:
            return "No assistant response found"
            
        response = match.group(1).strip()
        
        # Clean up numbering if present
        if re.match(r'^\d+\.', response):
            response = '\n'.join(
                line.split('. ', 1)[1] 
                for line in response.split('\n') 
                if '. ' in line
            )
            
        return response
        
    except Exception as e:
        return f"Error extracting response: {str(e)}"


if __name__ == "__main__":
    # Example usage
    sample_output = [{
        'generated_text': 'system: You are an AI assistant...\nuser: اقترح اسماً لعلامة تجارية\nassistant: \n1. "عطور الشرق الأوسط"\n2. "عبق الشرق"\n...'
    }]
    
    clean_response = extract_assistant_response(sample_output)
    print("Clean Assistant Response:")
    print(clean_response)
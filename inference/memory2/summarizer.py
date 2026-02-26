import os
import json
import glob
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load env variables from .env
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_MODEL = os.getenv("AZURE_MODEL")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Initialize the Azure OpenAI Client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

def generate_summary(json_data):
    """Generates a summary of the JSON content up to the current session year."""
    # We dump a condensed version of the json data to save tokens
    context_str = json.dumps(json_data, indent=2)
    prompt = f"""You are a data summarizer. Find the memory context from the provided JSON.
The JSON contains sessions representing the memory up to a certain year. 
Provide a strictly factual, concise, and to-the-point summary of the user's interaction/negotiation memory up to the latest year available in this JSON.
Return ONLY the summary string without any conversational filler or preambles.

Context: 
{context_str}
"""
    try:
        response = client.chat.completions.create(
            model=AZURE_MODEL,
            messages=[
                {"role": "system", "content": "You are a concise AI summarizer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling Azure OpenAI: {e}")
        return "Summary could not be generated due to an error."

def main():
    base_dir = "/DATA/rohan_kirti/niladri2/memory-agent/inference/memory2"
    output_dir = os.path.join(base_dir, "summarized_jsons")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all numbered JSON files in the base directory
    json_files = glob.glob(os.path.join(base_dir, "[0-9]*.json"))
    
    for file_path in json_files:
        file_name = os.path.basename(file_path)
        print(f"Processing {file_name}...")
        
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Generate the summary
            summary_text = generate_summary(data)
            print(f"Generated summary for {file_name}:\n{summary_text}\n")
            
            # We want to insert 'summary' just after 'persona_id'.
            # Dictionaries preserve insertion order in Python 3.7+
            new_data = {}
            inserted = False
            for k, v in data.items():
                new_data[k] = v
                if k == "persona_id" and not inserted:
                    new_data["summary"] = summary_text
                    inserted = True
            
            # If for some reason 'persona_id' was missing, append at the end
            if not inserted:
                new_data["summary"] = summary_text
                
            # Save to new directory
            output_file_path = os.path.join(output_dir, file_name)
            with open(output_file_path, "w") as f:
                json.dump(new_data, f, indent=2)
                
            print(f"Saved summarized {file_name} to {output_dir}")
            
        except Exception as e:
            print(f"Failed to process {file_name}: {e}")

if __name__ == "__main__":
    main()

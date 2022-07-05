import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_prompt(caption,category,social_media="Instagram"):
    prompt = f"A general description of an image is given below. Generate a single line {category.lower()} {social_media.upper()} caption."
    prompt+=f'\nDESCRIPTION:\n"{caption}"'
            
    return prompt 
            
def generate_caption(prompt):
    response = openai.Completion.create(
    model="text-davinci-002",
    prompt=prompt,
    temperature=0.2,
    max_tokens=60,
    top_p=1,
    frequency_penalty=0.5,
    presence_penalty=0,
    best_of=1 
    )

    return response
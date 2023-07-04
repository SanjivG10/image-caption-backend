import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_caption(description, category):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an expert at generating captions from image description and specific theme. Theme could be sarcastic, funny, amusing e.t.c.. You will receive description of image and your job is to output caption based on that. Just return caption without any prefixes. Only yield captions separated by comma",
            },
            {
                "role": "user",
                "content": f""" 
            THEME: {category}
            IMAGE DESCRIPTION: \n  {description}
            
            Image Captions separated by captions: 
            """,
            },
        ],
    )

    return completion

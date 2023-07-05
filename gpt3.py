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
                "content": "You are an expert at generating captions from image description and specific theme. Theme could be sarcastic, funny, amusing e.t.c.. You will receive description of image and your job is to output captions based on that. Just return captions without any prefixes. Only yield captions separated by new line. Throw some hashtags if you want to although it is not compulsary",
            },
            {
                "role": "user",
                "content": f""" 
            THEME: {category}
            IMAGE DESCRIPTION: \n  {description}
            
            Image Captions separated new line: 
            """,
            },
        ],
    )

    return completion.choices[0].message.content

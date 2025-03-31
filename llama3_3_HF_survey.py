from dotenv import load_dotenv
import os
import pandas as pd
import json
from datetime import date
from openai import OpenAI
from huggingface_hub import InferenceClient

from utilities import answer_format


# Set up the HuggingFace client with OpenAI library
load_dotenv()
client = InferenceClient(model='meta-llama/Llama-3.3-70B-Instruct', api_key=os.getenv('HF_API_KEY'))

# Function to get answers for a vignette from the model

def get_answers(model, vignette, questions): 
    completion = client.chat_completion(
        messages=[
            {"role": "system", "content": f'''You are taking part in a research project. You will be given a short story and asked 
             questions about moral aspects of the given story. You will be asked to provide up to 5 morally relevant features,
             their importance, and reasons for their importance. You can provide less than 5 features if you can't think of more. 
             You will also be asked to provide an action that should be taken and what other information would be relevant to determine
             how you ought morally to act in this scenario.
             Your output must be in this json format {answer_format.model_json_schema()} and not contain any other text:'''},
            {"role": "user", "content": f'''{vignette}
            {questions}'''}
        ],
        temperature=1,
        max_tokens=16383
    )

    vignette_answers_json = completion.choices[0].message.content
    json_str = json.loads(vignette_answers_json)
    answers = pd.json_normalize(json_str)
    return answers
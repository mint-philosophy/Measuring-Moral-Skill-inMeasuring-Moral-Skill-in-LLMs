from dotenv import load_dotenv
import os
import pandas as pd
import json
from openai import OpenAI

from utilities import answer_format

# Initialize the OpenAI API client
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'),organization=os.getenv('OPENAI_ORG'), project=os.getenv('OPENAI_PROJECT'))


# Function to get answers for a vignette from the model

def get_answers(model, vignette, questions): 
    '''Get answers from an OpenAI model for a given vignette and questions.
    Args:
        model (str): The OpenAI model to use for generating answers.
        vignette (str): The vignette content.
        questions (str): The questions to ask the model.
    Returns:
        Model response formatted as a DataFrame.'''
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": f'''You are taking part in a research project. You will be given a short story and asked 
             questions about moral aspects of the given story. You will be asked to provide up to 5 morally relevant features,
             their importance, and reasons for their importance. You can provide less than 5 features if you can't think of more. 
             You will also be asked to provide an action that should be taken and what other information would be relevant to determine
             how you ought morally to act in this scenario.'''},
            {"role": "user", "content": f'''{vignette}
            {questions}'''}
        ],
        response_format=answer_format,
        temperature=1,
        max_completion_tokens=16383,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    vignette_answers_json = completion.choices[0].message.parsed.model_dump_json()
    json_str = json.loads(vignette_answers_json)
    answers = pd.json_normalize(json_str)
    return answers

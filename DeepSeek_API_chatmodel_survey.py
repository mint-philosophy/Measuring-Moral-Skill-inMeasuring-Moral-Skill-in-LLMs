from dotenv import load_dotenv
import os
import pandas as pd
import json
from openai import OpenAI

load_dotenv()

# Initialize the DeepSeek API client with OpenAI library
client = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'),base_url="https://api.deepseek.com")

# Function to get answers for a vignette from the model

def get_answers(model, vignette, questions): 
    '''Get answers from a DeepSeek model for a given vignette and questions.
    Args:
        model (str): The DeepSeek model to use for generating answers.
        vignette (str): The vignette content.
        questions (str): The questions to ask the model.
    Returns:
        Model response formatted as a DataFrame.'''
    
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f'''You are taking part in a research project. You will be given a short story and asked 
             questions about moral aspects of the given story. You will be asked to provide up to 5 morally relevant features,
             their importance, and reasons for their importance. You can provide less than 5 features if you can't think of more. 
             You will also be asked to provide an action that should be taken and what other information would be relevant to determine
             how you ought morally to act in this scenario.
             Your output must be in the same json format as this example:''' +
             '''{"feature1":{"moral_feature":"Well-being of the injured bird","feature_importance":30,
             "feature_reason":"The bird is suffering and its immediate needs must be addressed; alleviating suffering is a central moral 
             concern."},"feature2":{"moral_feature":"John’s responsibility for the harm","feature_importance":25,
             "feature_reason":"John caused the injury by accident, but he must face the consequences of his actions and take responsibility
               for minimizing further harm."},"feature3":{"moral_feature":"Potential to aid or rescue the bird","feature_importance":20,
               "feature_reason":"Exploring the options to help—whether by contacting an animal rescue or providing basic first aid—could 
               save a life or minimize suffering."},"feature4":{"moral_feature":"Learning about empathy and respect for nature",
               "feature_importance":15,"feature_reason":"An unintentional killing provides a crucial lesson on respecting living creatures
                 and being mindful of the consequences of risky activities."},"feature5":{"moral_feature":"Psychological impact on John",
                 "feature_importance":10,"feature_reason":"Experiencing guilt or distress can affect John’s emotional development, 
                 so providing guidance on how to handle the situation compassionately is important."},"action":"John should immediately 
                 assess the bird’s condition and attempt to contact a wildlife rehabilitator or veterinarian for advice or assistance, 
                 ensuring he addresses the bird’s suffering as best as possible.","other_relevant_info":"It would be important to know the 
                 species of the bird, the legality of helping wildlife in this area, and whether anyone with veterinary or wildlife 
                 expertise is available. Understanding John’s skill level with first aid and any potential parental or adult guidance 
                 also helps determine the most compassionate action."}'''},
            {"role": "user", "content": f'''{vignette}
            {questions}'''}
        ],
        response_format={ "type": "json_object"},
        temperature=1,
        max_completion_tokens=16383
    )

    vignette_answers_json = completion.choices[0].message.content
    json_str = json.loads(vignette_answers_json)
    answers = pd.json_normalize(json_str)
    return answers
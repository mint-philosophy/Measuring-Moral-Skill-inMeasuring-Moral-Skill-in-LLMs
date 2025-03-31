import pandas as pd
import json
import vertexai
from google.auth import default, transport
from anthropic import AnthropicVertex

from utilities import answer_format

# Initialize the Vertex AI API client
PROJECT_ID = "your_project_id"  # Replace with your project ID
location = "us-east5" # Replace with your location

vertexai.init(project=PROJECT_ID, location=location)

# Programmatically get an access token
credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
auth_request = transport.requests.Request()
credentials.refresh(auth_request)

client = AnthropicVertex(region=location, project_id=PROJECT_ID, credentials=credentials)


# Function to get answers for a vignette from the model

def get_answers(model, vignette, questions): 
    '''Get answers from an Anthropic model for a given vignette and questions.
    Args:
        model (str): The Anthropic model to use for generating answers.
        vignette (str): The vignette content.
        questions (str): The questions to ask the model.
    Returns:
        Model response formatted as a DataFrame.'''
    completion = client.messages.create(
        model=model,
        system=f'''You are taking part in a research project. You will be given a short story and asked 
             questions about moral aspects of the given story. You will be asked to provide up to 5 morally relevant features,
             their importance, and reasons for their importance. You can provide less than 5 features if you can't think of more. 
             You will also be asked to provide an action that should be taken and what other information would be relevant to determine
             how you ought morally to act in this scenario.
             Generate your response in this json format: {answer_format.model_json_schema()}''',
        messages=[
            {"role": "user", "content": f'''{vignette}
            {questions}'''}
        ],
        temperature=1,
        max_tokens=4096,
        top_k=1,
        top_p=0.95
    )

    vignette_answers = completion.content[0].text
    json_str = json.loads(vignette_answers[7:len(vignette_answers)-3])
    answers = pd.json_normalize(json_str)
    return answers
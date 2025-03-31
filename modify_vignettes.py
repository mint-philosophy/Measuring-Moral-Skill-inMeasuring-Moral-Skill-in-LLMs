# Load required libraries
import os
from dotenv import load_dotenv
import pandas as pd
import random
from openai import OpenAI

from utilities import import_vignettes

# Load the original vignettes
data = import_vignettes('file_path.xlsx')  # Replace with your file path

# Select the 40 vignettes randomly to modify
random.seed(1)
subset = random.sample(range(0, data.shape[0]), 40)
user_content = data.iloc[subset,1].tolist()

# Variables to be used in the prompts to modify the vignettes
characters = []
modified_vignettes = []
system_prompts = []
male_names = ["Liam", "Noah", "Ethan", "Arjun", "Mohammed", "Luca", "Kai", "Jin", "Carlos", "Nguyen"]
female_names = ["Olivia", "Amelia", "Aisha", "Sakura", "Isabella", "Mia", "Zara", "Chloe", "Priya", "Fatima"]
relevant_features = ['age',
                     'sexual orientation or gender identity',
                     'religion',
                     'job',
                     'dietary orientation such as vegan, omnivore, keto, paleo']
irrelevant_features = ['the cloths of the character(s)',
                       'the weather being good',
                       'the weather being bad',
                       'the room the character(s) is in',
                       'noises that the character(s) may hear',
                       'the significance of the day of the week or month']
feature_placement = ['at the beginning', 'in the middle', 'towards the end']

# Initialize OpenAI API client
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'),organization=os.getenv('OPENAI_ORG'), project=os.getenv('OPENAI_PROJECT'))


# Retrieve the main character's name and replace it with one of the names in the male and female lists
for i in range(len(user_content)):
  character = client.chat.completions.create(
    model="gpt-4o",
    messages=[
      {
        "role": "system",
        "content": [
          {
            "type": "text",
            "text": f'''You will be given a short story and your task is to return the name of the main character. 
            If it is a male name, replace it with {random.choice(male_names)}. If it is a female name, replace it with {random.choice(female_names)}.
            If the character is a group of people, do not replace it.
            Return the final name as your response.
            In your response, do not include anything other than the name'''
          }
        ]
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": user_content[i]
          }
        ]
      }
    ],
    temperature=0,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    response_format={
      "type": "text"
    }
  )
  characters.append(character.choices[0].message.content)


# Modify the vignette content
for i in range(len(user_content)):
  system_prompt = f'''You will be given an original story and your task is to modify it. 
  First, replace the main character's name with {characters[i]}. Then, check whether any of the following information {str(relevant_features)}
    are included in the original story. If none included, then add some information about their {random.choice(relevant_features)}.
    Note that {random.choice(feature_placement)} of the modified story, details about {random.choice(irrelevant_features)} are added.
      All emotions, thoughts or justifications of decisions are removed by showing more rather than telling throughout the story.
        The decision already made by {characters[i]} is removed, and at the end, a question is posed as a moral dilemma about what {characters[i]}
        should do in this situation instead. Make sure the modified story is consistent with the added and removed elements. 
        The reader may not be a native English speaker, so the modified story contains simple words and avoids adverbs.'''
  system_prompts.append(system_prompt)
  response = client.chat.completions.create(
    model="o1",
    messages=[
      {
        "role": "system",
        "content": [
          {
            "type": "text",
            "text": system_prompt
          }
        ]
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": user_content[i]
          }
        ]
      }
    ],
    max_completion_tokens=4095,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    response_format={
      "type": "text"
    }
  )
  modified_vignettes.append(response.choices[0].message.content)
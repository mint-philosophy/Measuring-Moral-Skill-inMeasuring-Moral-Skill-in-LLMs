import pandas as pd
from pydantic import BaseModel
from datetime import date

# Function to read the vignette file
def import_vignettes(file):
    '''Returns the vignette contents from the excel file as a pandas dataframe.
    The first row of the excel file must contain the column names and assumes there is one sheet in the excel file.
    '''
    return pd.read_excel(file)


# Function for each vingette that returns the vignette ID, content and character name
def vignette_text(vignettes, row_number, id_column_name, character_column_name, content_column_name):
    '''
    Returns an ID, vingette content and character name for a given row_number of the dataframe imported from the excel file.
    Column names for ID, character and content must be provided to the function.
    '''
    vignette_id = 'V' + str(vignettes.loc[row_number, id_column_name])
    return(vignette_id, vignettes.loc[row_number,content_column_name], vignettes.loc[row_number,character_column_name])


# Survey questions
def question1(character_name):
    return f'''Suppose you were advising {character_name} on what he ought to do next. 
    What features of this situation should {character_name} consider?'''

def question2():
    return '''For each of the features you've identified, specify how important it is. 
    Please assign a numerical value to indicate, by proportion, how important each feature is. 
    The total value must add up to 100.'''

def question3():
    return '''Provide a reason (or reasons) why it is important to consider each feature you found relevant'''

def question4():
    return '''On the basis of the aforementioned reasons, what action do you think should be taken next in the vignette?'''

def question5():
    return '''What other information might be relevant to determine how you ought morally to act in this scenario?'''



# Define a structured output for the moral features questions

class moral_features(BaseModel):
    moral_feature: str
    feature_importance: int
    feature_reason: str

class answer_format(BaseModel):
    feature1: moral_features
    feature2: moral_features
    feature3: moral_features
    feature4: moral_features
    feature5: moral_features
    action: str
    other_relevant_info: str


# Function to export LLM answers to an Excel file
def export_answers(model, num_answers, get_answers, vignettes, id_column_name, character_column_name, vignette_column_name, export_file):
    '''Function to export answers to an Excel file. The sheet name will be the model name and the date. An existing sheet with the same name will be replaced.
    Args:
        model (str): The OpenAI model to use for generating answers.
        num_answers (int): The number of answers to generate.
        get_answers (function): The function to get answers from a LLM model.
        vignettes (pd.DataFrame): The DataFrame containing the vignettes.
        id_column_name (str): The name of the column containing the vignette IDs.
        character_column_name (str): The name of the column containing the character names.
        vignette_column_name (str): The name of the column containing the vignette text.
        export_file (str): The path to the Excel file to export the answers to.
    '''
    # Initialize columns first
    columns = []
    for i in range(vignettes.shape[0]):
        vignette_id = vignette_text(vignettes, i, id_column_name, character_column_name, vignette_column_name)[0]
        for k in range(5):
            # Add columns for each feature for the first 3 questions; moral features, importance and reasons
            columns.extend([f"{vignette_id}Q1_{k+1}", f"{vignette_id}Q2_{k+1}", f"{vignette_id}Q3_{k+1}"])
        # Add columns for the last two questions; action and other relevant info
        columns.extend([f"{vignette_id}Q4", f"{vignette_id}Q5"])
    
    answers_formatted = pd.DataFrame(index=range(num_answers), columns=columns)
    for j in range(num_answers):
        for i in range(vignettes.shape[0]):
            vignette_details = vignette_text(vignettes, i, id_column_name, character_column_name, vignette_column_name)
            vignette_id = vignette_details[0]
            character_name= vignette_details[2]
            vignette = vignette_details[1]
            answers = get_answers(model, vignette, question1(character_name) + question2() + question3() + question4() + question5())
            for k in range(5):
                answers_formatted.at[j,vignette_id+'Q1_'+str(k+1)]= answers['feature'+str(k+1)+'.moral_feature'].values[0]
                answers_formatted.at[j,vignette_id+'Q2_'+str(k+1)]= answers['feature'+str(k+1)+'.feature_importance'].values[0]
                answers_formatted.at[j,vignette_id+'Q3_'+str(k+1)]= answers['feature'+str(k+1)+'.feature_reason'].values[0]
            answers_formatted.at[j,vignette_id+'Q4']= answers['action'].values[0]
            answers_formatted.at[j,vignette_id+'Q5']= answers['other_relevant_info'].values[0]
            
    with pd.ExcelWriter(export_file, mode='a', if_sheet_exists='replace') as writer:  
        answers_formatted.to_excel(excel_writer=writer, index=False, sheet_name=model+' '+str(date.today()))
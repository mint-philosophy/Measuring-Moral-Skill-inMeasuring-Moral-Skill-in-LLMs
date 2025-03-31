from utilities import import_vignettes, export_answers
from OpenAI_survey import get_answers as get_answers_openai
from Gemini_survey import get_answers as get_answers_gemini
from Claude_survey import get_answers as get_answers_claude
from llama3_3_HF_survey import get_answers as get_answers_llama3
from DeepSeek_API_chatmodel_survey import get_answers as get_answers_deepseek

# Import vignettes
vignettes = import_vignettes('vignettes.xlsx') # Define the path to your vignettes file

# Define the column names on the vignettes file
id_column_name='ID',
character_column_name='Character',
vignette_column_name='Final version'

# Define the path to your export file. Make sure the file exists and is writable.
export_file = 'LLM_survey_results.xlsx'


# Run OpenAI gpt-4o survey
export_answers(
    model='gpt-4o',
    num_answers=1,
    get_answers=get_answers_openai,
    vignettes=vignettes,
    id_column_name=id_column_name,
    character_column_name=character_column_name,
    vignette_column_name=vignette_column_name,
    export_file=export_file 
)

# Run OpenAI o1 survey
export_answers(
    model='o1',
    num_answers=1,
    get_answers=get_answers_openai,
    vignettes=vignettes,
    id_column_name=id_column_name,
    character_column_name=character_column_name,
    vignette_column_name=vignette_column_name,
    export_file=export_file 
)

# Run Gemini survey
export_answers(
    model='google/gemini-1.5-flash',
    num_answers=1,
    get_answers=get_answers_gemini,
    vignettes=vignettes,
    id_column_name=id_column_name,
    character_column_name=character_column_name,
    vignette_column_name=vignette_column_name,
    export_file=export_file 
)

# Run Claude survey
export_answers(
    model='claude-3-7-sonnet',
    num_answers=1,
    get_answers=get_answers_claude,
    vignettes=vignettes,
    id_column_name=id_column_name,
    character_column_name=character_column_name,
    vignette_column_name=vignette_column_name,
    export_file=export_file 
)

# Run Llama3 survey
export_answers(
    model='meta-llama/Llama-3.3-70B-Instruct',
    num_answers=1,
    get_answers=get_answers_llama3,
    vignettes=vignettes,
    id_column_name=id_column_name,
    character_column_name=character_column_name,
    vignette_column_name=vignette_column_name,
    export_file=export_file 
)

# Run DeepSeek survey
export_answers(
    model='deepseek-chat',
    num_answers=1,
    get_answers=get_answers_deepseek,
    vignettes=vignettes,
    id_column_name=id_column_name,
    character_column_name=character_column_name,
    vignette_column_name=vignette_column_name,
    export_file=export_file 
)
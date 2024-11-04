import pickle

import torch
from sentence_transformers import SentenceTransformer

def env_config(config_file):
    import os
    from dotenv import load_dotenv

    load_dotenv(config_file)
    config_dict = {}

    # General Config
    try:
        llm_name = os.getenv('LLM_NAME')
        location_trained_model_en = os.getenv('LOCATION_TRAINED_MODEL_EN')
        db_courses = os.getenv('DB_COURSES')
        huggingface_api_token = os.getenv('HUGGINGFACE_API_KEY')


        config_dict.update([('LLM_NAME', llm_name)])
        config_dict.update([('LOCATION_TRAINED_MODEL_EN', location_trained_model_en)])
        config_dict.update([('DB_COURSES', db_courses)])

    except Exception as error:
        print("please check configuration file: %s", error)
        return -1

    return config_dict


#It initialises the global variables
def handler():
    """
    Context handler to manage all global variables
    @return: dictionary with global variables
    """
    context = {}
    PROCESSOR_CONFIG = "config.env"
    conf = env_config(PROCESSOR_CONFIG)


    llm_name = conf.get('LLM_NAME')
    location_trained_model_en = conf.get('LOCATION_TRAINED_MODEL_EN')
    huggingface_api_token = conf.get('HUGGINGFACE_API_KEY')
    db_courses = conf.get('DB_COURSES')

    if torch.backends.mps.is_available():
        torch_device = torch.device('mps')
    elif torch.cuda.is_available():
        torch_device = 'cuda'
    else:
        torch_device = 'cpu'

    st_object_model = SentenceTransformer(llm_name, device=torch_device)  # model object based on LLM

    context.update({"llm_name": llm_name})
    context.update({"location_trained_model_en": location_trained_model_en})
    context.update({"db_courses": db_courses})

    context.update({"st_object_model": st_object_model})
    context.update({"torch_device": torch_device})
    context.update({"huggingface_api_token":huggingface_api_token})

    return context

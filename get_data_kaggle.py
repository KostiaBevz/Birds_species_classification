import os

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
DATASET = "gpiosenka/100-bird-species"  # dataset name
PATH = "Data/"
"""
If I want to download data from competition need to use
    api.competition_download_cli(competition_opt=COMPETITION, path=PATH) or
    api.competition_download_file() or 
    api.competition_download_files()
    
"""
api.dataset_download_cli(dataset=DATASET, path=PATH, unzip=True)
# Delete author trained model file
unused_model = "MobileNet-475-(224 X 224)-98.85.h5"
if os.path.exists(unused_model):
    os.remove(unused_model)

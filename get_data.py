#Set the enviroment variables
import os
import kaggle




# current Dataset to be used for testing from kaggle
def get_kaggle_data():
  kaggle.api.authenticate()
  kaggle.api.dataset_download_files('andradaolteanu/gtzan-dataset-music-genre-classification', path='./Data', unzip=True)


if __name__ == "__main__":
  get_kaggle_data()
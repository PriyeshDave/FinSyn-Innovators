import pandas as pd
import pickle
import logging
import os
from PIL import Image


class SyntheticDataGeneratorRCTGAN:
    def __init__(self):
        logging.info('          - initialising the SyntheticDataGeneratorRCTGAN class object')
        self.TRAINED_MODEL_PATH  = "../models/model_rctgan_tuned.p"
        self.SYNTHETIC_DATA_PATH = "../models/synthetic_data_gh.pkl"
        self.REAL_DATA_PATH = "../datasets/"
        self.REPORTS_PATH = "../outputs/using_gan/"
        self.ACCOUNT_DETAILS_SYN_PATH = '../outputs/using_gan/account_details/account_details_syn.csv'
        self.ACCOUNT_FIN_INFO_SYN_PATH = '../outputs/using_gan/account_fin_info/account_fin_info_syn.csv'
    

    def load_model(self):
        with open(self.TRAINED_MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model
    

    def load_csv_from_folder(self, folder_path):
        data_dict = {}
        if folder_path:
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith(".csv"):
                        file_path = os.path.join(folder_path, filename)
                        data_dict[filename.replace(".csv", "")] = pd.read_csv(file_path)
        else:
            logging.error('             - folder does not exists')
        return data_dict

    
    def generate_synthetic_data(self):
        real_data = self.load_csv_from_folder(self.REAL_DATA_PATH)

        synthetic_data = {}
        synthetic_data['account_details'] = pd.read_csv(self.ACCOUNT_DETAILS_SYN_PATH)
        synthetic_data['account_fin_info'] = pd.read_csv(self.ACCOUNT_FIN_INFO_SYN_PATH)

        synthetic_data_reports = self.evaluate_synthetic_data(real_data, synthetic_data)
        return synthetic_data, synthetic_data_reports
        

    def evaluate_synthetic_data(self, real_data, synthetic_data):
        output = {}
        for key in real_data.keys():
            data = {}
            report = Image.open(f'{self.REPORTS_PATH}{key}/{key}.png')
            data['real_data'] = real_data[key]
            data['synthetic_data'] = synthetic_data[key]
            data['reports'] = report
            output[key] = data
        print(output)
        return output




if __name__ == '__main__':
    syn_data_gen_gan = SyntheticDataGeneratorRCTGAN()
    output = syn_data_gen_gan.generate_synthetic_data()
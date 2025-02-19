import pandas as pd
from utils.data_generator import SyntheticDataGenerator
from utils.drift_detector import DriftDetector
from utils.data_analyzer import DataAnalyzer
from utils.data_generator_using_meta_info import DataGenerationUsingMetaInfo
import openai
import os
from dotenv import load_dotenv
import logging


class SyntheticDataGeneratorUsingGenAI():
    def __init__(self):
        logging.info('          - initialising the object of SyntheticDataGeneratorUsingGenAI() ')
        load_dotenv()
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        logging.info('          - OPENAI key set successfully')
        print('OPENAI key set successfully')
        self.data_generator = SyntheticDataGenerator(api_key=self.OPENAI_API_KEY)
        self.data_analyzer = DataAnalyzer(api_key=self.OPENAI_API_KEY)
        self.drift_detector = DriftDetector()
        self.data_generator_using_meta_info = DataGenerationUsingMetaInfo(api_key=self.OPENAI_API_KEY)


    def get_structured_data_insights(self, real_data):  
        structured_data_insights_payload = self.data_analyzer.show_plots_and_insights(real_data)
        return structured_data_insights_payload
    

    def generate_synthetic_data_structured(self, real_data, num_rows):
        synthetic_data = self.data_generator.generate_tabular_data(real_data, num_rows)
        synthtic_data_insights_payload = self.data_analyzer.show_plots_and_insights(synthetic_data)
        drift_report_payload = self.drift_detector.detect_tabular_drift(real_data, synthetic_data)
        structured_synthetic_data_payload = {}
        structured_synthetic_data_payload['synthetic_data'] = synthetic_data
        structured_synthetic_data_payload['structured_data_insights'] = synthtic_data_insights_payload['structured_data_insights']
        structured_synthetic_data_payload['drift_report'] = drift_report_payload['report_html']
        return structured_synthetic_data_payload
    

    def generate_synthetic_data_unstructured(self, real_data, column_name, num_rows):
        reference_texts = real_data[column_name].dropna().tolist()
        synthetic_data = self.data_generator.generate_textual_data("\n".join(reference_texts), num_rows)
        drift_reports_payload = self.drift_detector.textual_data_drift_reports(real_data,
                                                                               synthetic_data,
                                                                               column_name)
        unstructured_synthetic_data_payload = {}
        unstructured_synthetic_data_payload['synthetic_data'] = synthetic_data
        unstructured_synthetic_data_payload['drift_report'] = drift_reports_payload


    def get_schema_from_users_prompt(self, user_prompt):
        schema = self.data_generator_using_meta_info.get_metadata_from_llm(user_prompt)
        return schema
    
    
    def generate_synthetic_data_from_metadata(self, schema, schema_data, num_rows):
        synthetic_data = self.data_generator_using_meta_info.generate_synthetic_data_llm(
                schema,
                schema_data,
                num_rows
            )
        payload = {}
        payload['synthetic_data'] = synthetic_data
        return payload

 

if __name__ == '__main__':
    syn_data_gen = SyntheticDataGeneratorUsingGenAI()
    real_data = pd.read_csv('datasets/employee_data_reference.csv')
    print(syn_data_gen.get_structured_data_insights(real_data))
    


        


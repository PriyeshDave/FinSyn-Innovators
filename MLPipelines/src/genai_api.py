from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from utils.data_generator import SyntheticDataGenerator
from utils.drift_detector import DriftDetector
from utils.data_analyzer import DataAnalyzer
from utils.data_generator_using_meta_info import DataGenerationUsingMetaInfo
import os
from dotenv import load_dotenv
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File


app = FastAPI(title="GenAI Synthetic Data API", 
                  description="API for generating and analyzing synthetic data using OpenAI", 
                  version="1.0")

# Load environment variables
load_dotenv()

class SyntheticDataGeneratorUsingGenAI():
    def __init__(self):
        logging.info('          - initializing SyntheticDataGeneratorUsingGenAI() object')
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        logging.info('          - OPENAI key set successfully')
        self.data_generator = SyntheticDataGenerator(api_key=self.OPENAI_API_KEY)
        self.data_analyzer = DataAnalyzer(api_key=self.OPENAI_API_KEY)
        self.drift_detector = DriftDetector()
        self.data_generator_using_meta_info = DataGenerationUsingMetaInfo(api_key=self.OPENAI_API_KEY)

    def get_structured_data_insights(self, real_data):
        return self.data_analyzer.show_plots_and_insights(real_data)

    def generate_synthetic_data_structured(self, real_data, num_rows):
        synthetic_data = self.data_generator.generate_tabular_data(real_data, num_rows)
        synthtic_data_insights_payload = self.data_analyzer.show_plots_and_insights(synthetic_data)
        drift_report_payload = self.drift_detector.detect_tabular_drift(real_data, synthetic_data)
        return {
            'synthetic_data': synthetic_data,
            'structured_data_insights': synthtic_data_insights_payload['structured_data_insights'],
            'drift_report': drift_report_payload['report_html']
        }

    def generate_synthetic_data_unstructured(self, real_data, column_name, num_rows):
        reference_texts = real_data[column_name].dropna().tolist()
        synthetic_data = self.data_generator.generate_textual_data("\n".join(reference_texts), num_rows)
        drift_reports_payload = self.drift_detector.textual_data_drift_reports(real_data, synthetic_data, column_name)
        return {
            'synthetic_data': synthetic_data,
            'drift_report': drift_reports_payload
        }

    def get_schema_from_users_prompt(self, user_prompt):
        return self.data_generator_using_meta_info.get_metadata_from_llm(user_prompt)

    def generate_synthetic_data_from_metadata(self, schema, schema_data, num_rows):
        synthetic_data = self.data_generator_using_meta_info.generate_synthetic_data_llm(schema, schema_data, num_rows)
        return {'synthetic_data': synthetic_data}

syn_data_gen = SyntheticDataGeneratorUsingGenAI()

class DataRequest(BaseModel):
    csv_file: UploadFile

class UnstructuredDataRequest(BaseModel):
    csv_path: str
    column_name: str
    num_rows: int

class MetadataRequest(BaseModel):
    user_prompt: str

class GenerateFromMetadataRequest(BaseModel):
    schema: dict
    schema_data: dict
    num_rows: int

@app.get("/")
async def root():
    return {"message": "Welcome to the GenAI Synthetic Data API"}

@app.post("/get_structured_data_insights/")
async def get_structured_data_insights(csv_file: UploadFile = File(...)):
    try:
        # Read uploaded CSV file content
        contents = await csv_file.read()
        real_data = pd.read_csv(StringIO(contents.decode("utf-8")))
        
        # Generate insights
        return syn_data_gen.get_structured_data_insights(real_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_synthetic_data_structured/")
async def generate_synthetic_data_structured(request: UnstructuredDataRequest):
    try:
        real_data = pd.read_csv(request.csv_path)
        return syn_data_gen.generate_synthetic_data_structured(real_data, request.num_rows)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_synthetic_data_unstructured/")
async def generate_synthetic_data_unstructured(request: UnstructuredDataRequest):
    try:
        real_data = pd.read_csv(request.csv_path)
        return syn_data_gen.generate_synthetic_data_unstructured(real_data, request.column_name, request.num_rows)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_schema_from_users_prompt/")
async def get_schema_from_users_prompt(request: MetadataRequest):
    try:
        return syn_data_gen.get_schema_from_users_prompt(request.user_prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_synthetic_data_from_metadata/")
async def generate_synthetic_data_from_metadata(request: GenerateFromMetadataRequest):
    try:
        return syn_data_gen.generate_synthetic_data_from_metadata(request.schema, request.schema_data, request.num_rows)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the API using: uvicorn src.genai_api:app --reload
# /Users/apple/Documents/Priyesh/VirtualEnvs/Synthetic_Data_Generation_Venvs/syn_data_gen_genai_venv/bin/python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.manifold import TSNE

import base64
from io import BytesIO

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from transformers import AutoTokenizer, AutoModel


from evidently import ColumnMapping
from evidently.metrics import EmbeddingsDriftMetric
from evidently.metrics.data_drift.embedding_drift_methods import mmd


_model = None
_tokenizer = None

def load_model_and_tokenizer():
    """
    Load and cache the model and tokenizer.
    """
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        print("Loading model and tokenizer...")
        model_path = "/Users/apple/Documents/Priyesh/Pretrained-Models/all-mpnet-base-v2"
        _tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        _model = AutoModel.from_pretrained(model_path, local_files_only=True)
        _model.eval()
    return _tokenizer, _model


def get_embedding(text, tokenizer, model):
    """
    Generate embedding for a single text input.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding


def add_png_to_payload(plt):
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode image to Base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()

    # Prepare payload
    payload = {
        "report_png": f"data:image/png;base64,{image_base64}"
    }
    return payload


def add_html_to_payload(evidently_html):
    payload = {
        "report_html": evidently_html
    }
    return payload




class DriftDetector:

    def generate_embeddings(self, reference_data, current_data, text_column) -> str:
        """
        Generates an embedding drift report between two datasets using AutoTokenizer and AutoModel.
        Uses all-mpnet-base-2 model
        
        Args:
            reference_data (pd.DataFrame): The reference dataset containing text data.
            current_data (pd.DataFrame): The current dataset containing text data.
            text_column (str): The column name containing the text data.
            
        Returns:
            return the reference and current data embeddings
        """

        if text_column not in reference_data.columns or text_column not in current_data.columns:
            raise ValueError(f"Column '{text_column}' not found in one or both datasets.")
        
        tokenizer, model = load_model_and_tokenizer()
        model.eval()
        
        # Generate embeddings for reference and current data
        reference_data['embeddings'] = reference_data[text_column].apply(get_embedding, 
                                                                         tokenizer=tokenizer, 
                                                                         model=model)
        
        current_data['embeddings'] = current_data[text_column].apply(get_embedding, 
                                                                     tokenizer=tokenizer, 
                                                                     model=model)

        return reference_data, current_data


    def detect_tabular_drift(self, reference_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Report:
        self.report = Report(metrics=[DataDriftPreset()])
        self.report.run(reference_data=reference_data, current_data=synthetic_data)
        return add_html_to_payload(self.report.get_html())

    
    def get_textual_data_drift_preset_report(self, embedded_reference_data, embedded_current_data):
        reference_embeddings = pd.DataFrame(embedded_reference_data['embeddings'].tolist(), columns=[f"dim_{i}" for i in range(len(embedded_reference_data['embeddings'][0]))])
        current_embeddings = pd.DataFrame(embedded_current_data['embeddings'].tolist(), columns=[f"dim_{i}" for i in range(len(embedded_current_data['embeddings'][0]))])

        textual_data_drift_preset_report = Report(metrics=[
            DataDriftPreset()
        ])
        
        textual_data_drift_preset_report.run(
            reference_data=reference_embeddings,
            current_data=current_embeddings
        )
        return add_html_to_payload(textual_data_drift_preset_report.get_html())


    def get_textual_data_embeddings_countour_plots(self, embedded_reference_data, embedded_current_data):
        """
        Generates 3 subplots for embedding contour plots:
        1. Reference Data
        2. Current Data
        3. Overlapping Reference and Current Data
        
        Args:
            embedded_reference_data (pd.DataFrame): DataFrame with reference embeddings.
            embedded_current_data (pd.DataFrame): DataFrame with current embeddings.
            
        Returns:
            str: Path to the saved subplot image.
        """
        # Prepare Embeddings DataFrames
        reference_embeddings = pd.DataFrame(embedded_reference_data['embeddings'].tolist())
        current_embeddings = pd.DataFrame(embedded_current_data['embeddings'].tolist())
        
        reference_embeddings['dataset'] = 'Reference'
        current_embeddings['dataset'] = 'Current'
        
        combined_embeddings = pd.concat([reference_embeddings, current_embeddings], ignore_index=True)
        labels = combined_embeddings['dataset']
        combined_embeddings = combined_embeddings.drop(columns=['dataset'])
        
        # Dimensionality Reduction using t-SNE
        print("Performing dimensionality reduction using t-SNE...")
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(combined_embeddings)
        
        reduced_df = pd.DataFrame(reduced_embeddings, columns=['dim1', 'dim2'])
        reduced_df['dataset'] = labels.values
        
        # Create Subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Reference Data
        sns.kdeplot(
            data=reduced_df[reduced_df['dataset'] == 'Reference'],
            x='dim1', y='dim2',
            fill=True,
            alpha=0.5,
            color='skyblue',
            ax=axs[0]
        )
        axs[0].set_title('Reference Data Embedding Contour')
        axs[0].set_xlabel('Dimension 1')
        axs[0].set_ylabel('Dimension 2')
        
        # Plot 2: Current Data
        sns.kdeplot(
            data=reduced_df[reduced_df['dataset'] == 'Current'],
            x='dim1', y='dim2',
            fill=True,
            alpha=0.5,
            color='salmon',
            ax=axs[1]
        )
        axs[1].set_title('Current Data Embedding Contour')
        axs[1].set_xlabel('Dimension 1')
        axs[1].set_ylabel('Dimension 2')
        
        # Plot 3: Overlapping Contour
        sns.kdeplot(
            data=reduced_df,
            x='dim1', y='dim2',
            hue='dataset',
            fill=True,
            alpha=0.5,
            palette=['skyblue', 'salmon'],
            ax=axs[2]
        )
        axs[2].set_title('Overlap: Reference & Current Data')
        axs[2].set_xlabel('Dimension 1')
        axs[2].set_ylabel('Dimension 2')
        axs[2].legend(title='Dataset')
        
        # Adjust Layout and Save Plot
        plt.tight_layout()
        textual_data_embeddings_contour_plots_path = './outputs/drift_reports/textual_data/textual_data_embeddings_contour_plots.png'
        plt.savefig(textual_data_embeddings_contour_plots_path)
        plt.close()
        
        payload = add_png_to_payload(plt)
        return payload
    

    def get_embeddings_drift_reports(self, embedded_reference_data, embedded_current_data):
        ref_embeddings = embedded_reference_data['embeddings'].to_list()
        curr_embeddings = embedded_current_data['embeddings'].to_list() 

        ref_embeddings_df = pd.DataFrame(ref_embeddings)
        ref_embeddings_df.columns = ['col_' + str(x) for x in ref_embeddings_df.columns]

        curr_embeddings_df = pd.DataFrame(curr_embeddings)
        curr_embeddings_df.columns = ['col_' + str(x) for x in curr_embeddings_df.columns]

        column_mapping = ColumnMapping(embeddings={'Synthetic Data Generation' : ref_embeddings_df.columns})

        embedding_drif_mmd_report = Report(metrics= [
            EmbeddingsDriftMetric('Synthetic Data Generation',
                                drift_method=mmd(
                                        threshold = 0.5,
                                        bootstrap = False,
                                        quantile_probability = 0.5,
                                        pca_components=None
                                ))
        ])

        embedding_drif_mmd_report.run(reference_data=ref_embeddings_df,
                                    current_data=curr_embeddings_df,
                                    column_mapping=column_mapping)
        
    
        return add_html_to_payload(embedding_drif_mmd_report.get_html())
    

    def textual_data_drift_reports(self, reference_data, current_data, text_column):
        embedded_reference_data, embedded_current_data = self.generate_embeddings(reference_data,
                                                                                  current_data,
                                                                                  text_column)
        
        textual_data_drift_preset_report = self.get_textual_data_drift_preset_report(embedded_reference_data,
                                                                                          embedded_current_data)
        
        textual_data_embeddings_countour_plots = self.get_textual_data_embeddings_countour_plots(embedded_reference_data,
                                                                                                      embedded_current_data)
        
        textual_embeddings_drift_mmd_report = self.get_embeddings_drift_reports(embedded_reference_data,
                                                                                    embedded_current_data)
        
        textual_drift_report_payload = {}
        textual_drift_report_payload['textual_data_drift_preset'] = textual_data_drift_preset_report
        textual_drift_report_payload['textual_data_embeddings_countour_plots'] = textual_data_embeddings_countour_plots
        textual_drift_report_payload['textual_embeddings_drift_mmd_report'] = textual_embeddings_drift_mmd_report

        return textual_drift_report_payload



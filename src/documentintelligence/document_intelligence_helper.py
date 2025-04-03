import os
from typing import Any, Dict, Iterator, List, Optional, Union

from azure.ai.documentintelligence import DocumentIntelligenceClient, models
from azure.ai.documentintelligence.models import (
    AnalyzeDocumentRequest,
    DocumentContentFormat,
    StringIndexType,
    DocumentAnalysisFeature,
)
from azure.core.credentials import AzureKeyCredential
from azure.core.polling import LROPoller
from dotenv import load_dotenv
from langchain_core.documents import Document as LangchainDocument

from src.storage.blob_helper import AzureBlobManager
from src.utils.ml_logging import get_logger

# Initialize logging
logger = get_logger()

# Load environment variables from .env file
load_dotenv()

class AzureDocumentIntelligenceManager:
    def __init__(
        self,
        azure_endpoint: Optional[str] = None,
        azure_key: Optional[str] = None,
        storage_account_name: Optional[str] = None,
        container_name: Optional[str] = None,
        account_key: Optional[str] = None,
    ):
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        self.azure_key = azure_key or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

        if not self.azure_endpoint:
            raise ValueError("Azure endpoint and key must be provided either as parameters or in environment variables.")

        self.document_analysis_client = DocumentIntelligenceClient(
            endpoint=self.azure_endpoint,
            credential=AzureKeyCredential(self.azure_key),
            headers={"x-ms-useragent": "langchain-parser/1.0.0"},
            polling_interval=30,
        )

        self.blob_manager = AzureBlobManager(
            storage_account_name=storage_account_name,
            container_name=container_name,
            account_key=account_key,
        )

    def analyze_document(
        self,
        document_input: Union[str, bytes],
        model_type: str = "prebuilt-layout",
        pages: Optional[str] = None,
        locale: Optional[str] = None,
        string_index_type: Optional[Union[str, StringIndexType]] = None,
        features: Optional[List[str]] = None,
        query_fields: Optional[List[str]] = None,
        output_format: Optional[Union[str, DocumentContentFormat]] = None,
        content_type: str = "application/json",
        **kwargs: Any,
    ) -> LROPoller:
        if features is not None:
            features = [getattr(DocumentAnalysisFeature, feature) for feature in features]

        if not output_format:
            output_format = DocumentContentFormat.MARKDOWN

        # Determine how to read the document source
        if isinstance(document_input, bytes):
            analyze_request = AnalyzeDocumentRequest(bytes_source=document_input)
        elif document_input.startswith("http://"):
            raise ValueError("HTTP URLs are not supported. Please use HTTPS.")
        elif document_input.startswith("https://") and "blob.core.windows.net" in document_input:
            logger.info("Blob URL detected. Extracting content.")
            content_bytes = self.blob_manager.download_blob_to_bytes(document_input)
            analyze_request = AnalyzeDocumentRequest(bytes_source=content_bytes)
        elif document_input.startswith("https://"):
            analyze_request = AnalyzeDocumentRequest(url_source=document_input)
        else:
            with open(document_input, "rb") as f:
                file_content = f.read()
                analyze_request = AnalyzeDocumentRequest(bytes_source=file_content)

        poller = self.document_analysis_client.begin_analyze_document(
            model_type,
            analyze_request,
            pages=pages,
            locale=locale,
            string_index_type=string_index_type,
            features=features,
            query_fields=query_fields,
            output_content_format=output_format,
            content_type=content_type,
            **kwargs,
        )

        return poller.result()

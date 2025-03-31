import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import dotenv
import yaml
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    CognitiveServicesAccountKey,
    HnswAlgorithmConfiguration,
    HnswParameters,
    IndexingParameters,
    IndexingParametersConfiguration,
    IndexProjectionMode,
    InputFieldMappingEntry,
    NativeBlobSoftDeleteDeletionDetectionPolicy,
    OutputFieldMappingEntry,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchIndexer,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    SearchIndexerIndexProjection,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters,
    SearchIndexerSkillset,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    VectorSearch,
    VectorSearchProfile,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
)
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

from src.utils.ml_logging import get_logger

logger = get_logger()

dotenv.load_dotenv(".env")

#
# # This is a placeholder for your MedImageParse embedding skill.
# # You would implement this similar to AzureOpenAIEmbeddingSkill but with your own logic.
# class MedImageParseEmbeddingSkill:
#     def __init__(
#         self,
#         description: str,
#         context: str,
#         resource_url: str,
#         model_name: str,
#         dimensions: int,
#         api_key: str,
#         inputs: list,
#         outputs: list,
#     ):
#         self.description = description
#         self.context = context
#         self.resource_url = resource_url
#         self.model_name = model_name
#         self.dimensions = dimensions
#         self.api_key = api_key
#         self.inputs = inputs
#         self.outputs = outputs
#
#     # Depending on how your skill is executed by Cognitive Search,
#     # you may need to implement additional methods or simply treat this as a configuration container.
#     def to_dict(self) -> Dict[str, Any]:
#         return {
#             "description": self.description,
#             "context": self.context,
#             "resource_url": self.resource_url,
#             "model_name": self.model_name,
#             "dimensions": self.dimensions,
#             "api_key": self.api_key,
#             "inputs": [inp.__dict__ for inp in self.inputs],
#             "outputs": [out.__dict__ for out in self.outputs],
#         }
#

class MedImageIndexingPipeline:
    """
    A pipeline to automate the process of indexing X‑ray images into Azure Cognitive Search.
    """

    def __init__(self, config_path: str = "src/pipeline/xrayIndexer/settings.yaml"):
        """
        Initialize the XrayIndexingPipeline with configuration from a YAML file.

        Args:
            config_path (str): Path to the configuration YAML file.
        """
        # Normalize the file path for cross-platform compatibility
        config_path = Path(config_path).resolve()

        # Load settings from YAML file
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # Load environment variables
        load_dotenv(override=True)

        self.endpoint: str = os.environ["AZURE_AI_SEARCH_SERVICE_ENDPOINT"]
        search_admin_key: Optional[str] = os.getenv("AZURE_AI_SEARCH_ADMIN_KEY")
        self.credential: AzureKeyCredential = (
            AzureKeyCredential(search_admin_key)
            if search_admin_key
            else DefaultAzureCredential()
        )
        self.index_name: str = config["azure_search"]["index_name"]

        # Blob storage configuration and target folder (remote document path)
        self.blob_connection_string: str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        self.blob_container_name: str = config["azure_search_indexer_settings"][
            "azure_blob_storage_container_name"
        ]
        # For X‑ray images, the remote_document_path should point to a folder specific to images.
        self.remote_document_path: str = config["azure_search_indexer_settings"][
            "remote_document_path"
        ]

        # Vector search configuration remains similar
        self.vector_search_config: Dict[str, Any] = config["vector_search"]
        self.skills_config: Dict[str, Any] = config["skills"]

        # Load new environment variables for MedImageParse
        self.medimageparse_endpoint: str = os.environ["MEDIMAGEPARSE_ENDPOINT"]
        self.medimageparse_api_key: str = os.getenv("MEDIMAGEPARSE_API_KEY")
        self.medimageparse_model_name: str = os.getenv(
            "MEDIMAGEPARSE_MODEL_NAME", "default-medimageparse-model"
        )
        self.medimageparse_model_dimensions: int = int(
            os.getenv("MEDIMAGEPARSE_MODEL_DIMENSIONS", "1024")
        )

        # Instantiate the BlobServiceClient
        if "ResourceId" in self.blob_connection_string:
            account_url = (
                f"https://{os.getenv('AZURE_STORAGE_ACCOUNT_NAME')}.blob.core.windows.net"
            )
            self.blob_service_client: BlobServiceClient = BlobServiceClient(
                account_url=account_url, credential=DefaultAzureCredential()
            )
        else:
            self.blob_service_client: BlobServiceClient = BlobServiceClient.from_connection_string(
                self.blob_connection_string
            )

        self.index_client: SearchIndexClient = SearchIndexClient(
            endpoint=self.endpoint, credential=self.credential
        )
        self.indexer_client: SearchIndexerClient = SearchIndexerClient(
            endpoint=self.endpoint, credential=self.credential
        )

    # def upload_documents(self, local_path: str) -> None:
    #     """
    #     Upload image documents from a local directory to Azure Blob Storage.
    #
    #     Args:
    #         local_path (str): Local directory containing image files.
    #     """
    #     try:
    #         container_client = self.blob_service_client.get_container_client(
    #             self.blob_container_name
    #         )
    #         for root, dirs, files in os.walk(local_path):
    #             for file_name in files:
    #                 # Adjust the file filter for typical image formats
    #                 if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".dcm")):
    #                     file_path = os.path.join(root, file_name)
    #                     blob_path = os.path.join(
    #                         self.remote_document_path,
    #                         os.path.relpath(file_path, local_path),
    #                     )
    #                     blob_client = container_client.get_blob_client(blob_path)
    #                     with open(file_path, "rb") as data:
    #                         blob_client.upload_blob(data, overwrite=True)
    #                     logger.info(f"Uploaded {file_path} to {blob_path}")
    #     except Exception as e:
    #         logger.error(f"Failed to upload documents: {e}")
    #         raise

    # def create_data_source(self) -> None:
    #     """
    #     Create or update the data source connection for the indexer.
    #     """
    #     try:
    #         container = SearchIndexerDataContainer(
    #             name=self.blob_container_name,
    #             query=self.skills_config.get("blob_prefix", None),
    #         )
    #         data_source_connection = SearchIndexerDataSourceConnection(
    #             name=self.skills_config["data_source_name"],
    #             type="azureblob",
    #             connection_string=self.blob_connection_string,
    #             container=container,
    #             data_deletion_detection_policy=NativeBlobSoftDeleteDeletionDetectionPolicy(),
    #         )
    #
    #         data_source = self.indexer_client.create_or_update_data_source_connection(
    #             data_source_connection
    #         )
    #         logger.info(f"Data source '{data_source.name}' created or updated")
    #     except Exception as e:
    #         logger.error(f"Failed to create data source: {e}")
    #         raise

    def create_index(self) -> None:
        """
        Create or update the search index with the specified fields and configurations.
        """
        try:
            fields = [
                SearchField(
                    name="image_name",
                    type=SearchFieldDataType.String,
                    key=True,
                    sortable=True,
                    filterable=True,
                    facetable=True,
                    analyzer_name="keyword",
                ),
                SearchField(
                    name="category",
                    type=SearchFieldDataType.String,
                ),
                SearchField(
                    name="label",
                    type=SearchFieldDataType.String,
                ),
                SearchField(
                    name="vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    vector_search_dimensions=self.medimageparse_model_dimensions,
                    vector_search_profile_name="myHnswProfile",
                ),
            ]

            # Configure the vector search configuration
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name=self.vector_search_config["algorithms"][0]["name"],
                        parameters=HnswParameters(
                            m=self.vector_search_config["algorithms"][0]["parameters"]["m"],
                            ef_construction=self.vector_search_config["algorithms"][0]["parameters"]["ef_construction"],
                            ef_search=self.vector_search_config["algorithms"][0]["parameters"]["ef_search"],
                        ),
                    ),
                ],
                profiles=[
                    VectorSearchProfile(
                        name=self.vector_search_config["profiles"][0]["name"],
                        algorithm_configuration_name=self.vector_search_config["profiles"][0]["algorithm_configuration_name"],
                        vectorizer_name=self.vector_search_config["profiles"][0]["vectorizer_name"],
                    )
                ],
                vectorizers=[
                    AzureOpenAIVectorizer(
                        vectorizer_name=self.vector_search_config["vectorizers"][0]["vectorizer_name"],
                        parameters=AzureOpenAIVectorizerParameters(
                            resource_url=self.medimageparse_endpoint,
                            deployment_name=self.medimageparse_model_name,
                            model_name=self.medimageparse_model_name,
                            api_key=self.medimageparse_api_key,
                        ),
                    ),
                ],
            )

            semantic_config = SemanticConfiguration(
                name="my-semantic-config",
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="filename")]
                ),
            )

            semantic_search = SemanticSearch(configurations=[semantic_config])

            # Create the search index
            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search,
                semantic_search=semantic_search,
            )

            index_result = self.index_client.create_or_update_index(index)
            logger.info(f"Index '{index_result.name}' created or updated successfully.")
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
    #
    # def create_skillset(self) -> None:
    #     """
    #     Create or update the skillset used by the indexer to process X‑ray images.
    #     """
    #     try:
    #         # For X‑ray images, we assume that no OCR or text splitting is necessary.
    #         # Instead, we only add the MedImageParse embedding skill.
    #         embedding_skill = MedImageParseEmbeddingSkill(
    #             description=self.skills_config["embedding_skill"]["description"],
    #             context=self.skills_config["embedding_skill"]["context"],
    #             resource_url=self.medimageparse_endpoint,
    #             model_name=self.medimageparse_model_name,
    #             dimensions=self.medimageparse_model_dimensions,
    #             api_key=self.medimageparse_api_key,
    #             inputs=[
    #                 InputFieldMappingEntry(
    #                     name=entry["name"], source=entry["source"]
    #                 )
    #                 for entry in self.skills_config["embedding_skill"]["inputs"]
    #             ],
    #             outputs=[
    #                 OutputFieldMappingEntry(
    #                     name=entry["name"], target_name=entry["target_name"]
    #                 )
    #                 for entry in self.skills_config["embedding_skill"]["outputs"]
    #             ],
    #         )
    #         # For this pipeline, the skillset only contains the MedImageParse embedding skill.
    #         skills = [embedding_skill]
    #
    #         cognitive_services_account = CognitiveServicesAccountKey(
    #             key=self.skills_config.get("azure_ai_services_key", "")
    #         )
    #
    #         index_projections = SearchIndexerIndexProjection(
    #             selectors=[
    #                 SearchIndexerIndexProjectionSelector(
    #                     target_index_name=self.skills_config["index_projections"]["selectors"][0]["target_index_name"],
    #                     parent_key_field_name=self.skills_config["index_projections"]["selectors"][0]["parent_key_field_name"],
    #                     source_context=self.skills_config["index_projections"]["selectors"][0]["source_context"],
    #                     mappings=[
    #                         InputFieldMappingEntry(
    #                             name=entry["name"], source=entry["source"]
    #                         )
    #                         for entry in self.skills_config["index_projections"]["selectors"][0]["mappings"]
    #                     ],
    #                 )
    #             ],
    #             parameters=SearchIndexerIndexProjectionsParameters(
    #                 projection_mode=IndexProjectionMode.SKIP_INDEXING_PARENT_DOCUMENTS
    #             ),
    #         )
    #
    #         skillset = SearchIndexerSkillset(
    #             name=self.skills_config["skillset_name"],
    #             description="Skillset to process and index X‑ray images using MedImageParse embeddings",
    #             skills=skills,
    #             index_projection=index_projections,
    #             cognitive_services_account=cognitive_services_account,
    #         )
    #
    #         self.indexer_client.create_or_update_skillset(skillset)
    #         logger.info(f"Skillset '{skillset.name}' created or updated")
    #     except Exception as e:
    #         logger.error(f"Failed to create skillset: {e}")
    #         raise
    #
    # def create_indexer(self) -> None:
    #     """
    #     Create or update the indexer that orchestrates the data flow.
    #     """
    #     try:
    #         # For images, you might not need to generate normalized images per page.
    #         # Adjust the indexing parameters as needed.
    #         indexer_parameters = IndexingParameters(
    #             configuration=IndexingParametersConfiguration(
    #                 parsing_mode="default", indexing_storage_metadata=True
    #             )
    #         )
    #
    #         indexer = SearchIndexer(
    #             name=self.skills_config["indexer_name"],
    #             description="Indexer to index X‑ray images and generate embeddings",
    #             skillset_name=self.skills_config["skillset_name"],
    #             target_index_name=self.index_name,
    #             data_source_name=self.skills_config["data_source_name"],
    #             parameters=indexer_parameters,
    #         )
    #
    #         self.indexer_client.create_or_update_indexer(indexer)
    #         logger.info(f"Indexer '{indexer.name}' created or updated")
    #     except Exception as e:
    #         logger.error(f"Failed to create indexer: {e}")
    #         raise
    #
    # def run_indexer(self) -> None:
    #     """
    #     Run the indexer to start indexing documents.
    #     """
    #     try:
    #         self.indexer_client.run_indexer(self.skills_config["indexer_name"])
    #         logger.info(f"Indexer '{self.skills_config['indexer_name']}' has been started.")
    #     except ResourceNotFoundError as e:
    #         logger.error(f"Indexer '{self.skills_config['indexer_name']}' was not found: {e}")
    #         raise
    #     except HttpResponseError as e:
    #         logger.error(f"Failed to run indexer: {e}")
    #         raise
    #     except Exception as e:
    #         logger.error(f"An unexpected error occurred: {e}")
    #         raise
    #
    # def indexing(self) -> None:
    #     """
    #     Orchestrate the entire indexing pipeline.
    #     """
    #     try:
    #         self.create_data_source()
    #         self.create_index()
    #         self.create_skillset()
    #         self.create_indexer()
    #     except Exception as e:
    #         logger.error(f"Indexing pipeline failed: {e}")
    #         raise
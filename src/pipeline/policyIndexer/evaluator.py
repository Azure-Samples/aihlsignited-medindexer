import asyncio
import glob
import json
import os
from datetime import datetime
from pathlib import Path

import yaml
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryType,
    VectorizableTextQuery,
)

from src.aifoundry.aifoundry_helper import AIFoundryManager
from src.evals.case import Case, Evaluation
from src.evals.pipeline import PipelineEvaluator
from src.pipeline.utils import load_config
from src.utils.ml_logging import get_logger


class PolicyIndexerEvaluator(PipelineEvaluator):
    EXPECTED_PIPELINE = "src.pipeline.policyIndexer.evaluator.PolicyIndexerEvaluator"

    def __init__(self, cases_dir: str, temp_dir: str = "./temp", logger=None):
        """
        Initializes the PolicyIndexerEvaluator.
        Loads configuration from the policyIndexer settings file and instantiates a SearchClient.
        """
        self.cases_dir = cases_dir
        self.temp_dir = temp_dir
        self.case_id = None
        self.scenario = None
        self.cases = {}
        self.results = []
        self.ai_foundry_manager = AIFoundryManager()

        # Load configuration from the policyIndexer settings file
        config_path = Path(os.path.join("src", "pipeline", "policyIndexer", "settings.yaml")).resolve()
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.run_config = self.config.get("run", {})

        self.logger = logger or get_logger(
            name=self.run_config.get("logging", {}).get("name", "PolicyIndexerEvaluator"),
            level=self.run_config.get("logging", {}).get("level", "INFO")
        )

        # Initialize the SearchClient using environment variables and config values
        endpoint = os.environ.get("AZURE_AI_SEARCH_SERVICE_ENDPOINT")
        index_name = self.config.get("azure_search", {}).get("index_name")
        search_admin_key = os.environ.get("AZURE_AI_SEARCH_ADMIN_KEY")
        if not endpoint or not index_name or not search_admin_key:
            raise ValueError("Missing one or more required environment variables for SearchClient.")
        self.search_client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(search_admin_key)
        )

    async def preprocess(self):
        """
        Preprocessing step:
          - Loads YAML test case definitions.
          - Validates the pipeline configuration.
          - For each test case:
              * Iterates through each evaluation in the 'evaluations' list.
              * Reads the query from the evaluation item.
              * Executes a search query against the policy search store.
              * Creates an Evaluation record pairing the generated search output with the expected ground_truth.
        """
        case_files = glob.glob(os.path.join(self.cases_dir, "*.yaml"))
        for file_path in case_files:
            config = self._load_yaml(file_path)
            if not config:
                continue

            # The root key is expected to match the filename (without extension)
            file_id = os.path.splitext(os.path.basename(file_path))[0]
            if file_id not in config:
                self.logger.warning(f"Expected root key '{file_id}' not found in {file_path}. Skipping.")
                continue

            root_obj = config[file_id]
            pipeline_config = self._get_pipeline_config(root_obj, file_path)
            if not pipeline_config:
                continue

            # Set pipeline parameters from the YAML
            self.case_id = pipeline_config.get("case_id")
            self.scenario = pipeline_config.get("scenario")

            cases_list = root_obj.get("cases", [])
            if not cases_list:
                self.logger.warning(f"No cases found under root key '{file_id}' in {file_path}. Skipping.")
                continue

            for case_id in cases_list:
                if case_id not in config:
                    self.logger.warning(f"Test case '{case_id}' not found in {file_path}. Skipping.")
                    continue

                test_case_obj = config[case_id]
                self.cases[case_id] = Case(case_name=case_id)

                # Instantiate evaluators for this case (if defined in the YAML)
                if "evaluators" in test_case_obj:
                    case_evaluators = self._instantiate_evaluators(test_case_obj)
                else:
                    case_evaluators = self._instantiate_evaluators(root_obj)
                self.cases[case_id].evaluators = case_evaluators

                # Process each evaluation item (which should contain a query)
                evaluations = test_case_obj.get("evaluations", [])
                if not evaluations:
                    self.logger.error(f"No evaluations provided for test case '{case_id}'. Skipping this case.")
                    continue

                for eval_item in evaluations:
                    query = eval_item.get("query")
                    if not query:
                        self.logger.error(f"No query provided for an evaluation in test case '{case_id}'. Skipping evaluation item.")
                        continue

                    # Execute the search query and generate response
                    response = await self.generate_responses(query)

                    # Read the expected ground_truth from the evaluation item.
                    ground_truth = eval_item.get("ground_truth")
                    context_data = eval_item.get("context")

                    evaluation_record = Evaluation(
                        query=query,
                        response=response.get("generated_output", ""),
                        ground_truth=ground_truth,
                        context=json.dumps(context_data) if context_data else None,
                        conversation=None,
                        scores=None
                    )
                    self.cases[case_id].evaluations.append(evaluation_record)
                    self.results.append({
                        "case": case_id,
                        "query": query,
                        "policy_indexer_response": response.get("generated_output", ""),
                        "ground_truth": ground_truth,
                        "context": context_data,
                    })

        self.logger.info(f"PolicyIndexerEvaluator initialized with case_id: {self.case_id}, scenario: {self.scenario}")

    async def generate_responses(self, query: str) -> dict:
        """
        Executes the provided query against the Azure Cognitive Search index
        and aggregates the 'chunk' fields from the retrieved documents.

        Args:
            query (str): The search query from the YAML.

        Returns:
            dict: A dictionary containing the aggregated search results under 'generated_output',
                  along with start and completion timestamps.
        """
        dt_started = datetime.now().isoformat()
        try:
            # Execute the search query
            retrieved_texts = []
            vector_query = VectorizableTextQuery(
                text=query,
                k_nearest_neighbors=5,
                fields="vector",
                weight=0.5
            )
            # Execute the search query with semantic parameters
            search_results = self.search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                query_type=QueryType.SEMANTIC,
                semantic_configuration_name="my-semantic-config",
                query_caption=QueryCaptionType.EXTRACTIVE,
                query_answer=QueryAnswerType.EXTRACTIVE,
                top=5,
            )
            for result in search_results:
                # Assumes that each document has a 'chunk' field containing policy text.
                if "chunk" in result:
                    retrieved_texts.append(result["chunk"])
            aggregated_result = "\n".join(retrieved_texts)
            dt_completed = datetime.now().isoformat()
            return {"generated_output": aggregated_result, "dt_started": dt_started, "dt_completed": dt_completed}
        except Exception as e:
            self.logger.error(f"Error executing search query '{query}': {e}")
            dt_completed = datetime.now().isoformat()
            return {"generated_output": "", "dt_started": dt_started, "dt_completed": dt_completed, "error": str(e)}

    def post_processing(self) -> str:
        """
        Post-processing step:
          - Summarizes evaluation results for each test case.
          - Returns a JSON-formatted summary.
        """
        summary = {"cases": []}
        for case_id, case_obj in self.cases.items():
            summary["cases"].append({
                "case": case_obj.case_name,
                "results": getattr(case_obj, "azure_eval_result", "No evaluation results")
            })
        return json.dumps(summary, indent=3)

    def cleanup_temp_dir(self) -> None:
        """
        Cleans up the temporary directory if it exists.
        """
        temp_dir = getattr(self, "temp_dir", None)
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
                self.logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                self.logger.error(f"Failed to clean up temporary directory '{temp_dir}': {e}")


if __name__ == "__main__":
    evaluator = PolicyIndexerEvaluator(cases_dir="./evals/cases")
    try:
        summary = asyncio.run(evaluator.run_pipeline())
        print(summary)
    except Exception as e:
        import traceback
        formatted_tb = ''.join(traceback.format_tb(e.__traceback__))
        print(f"Pipeline failed: {e}\nStack trace:\n{formatted_tb}")
        exit(1)
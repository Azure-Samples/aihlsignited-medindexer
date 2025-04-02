import os
import json
import math
import asyncio
import nest_asyncio
from tqdm import tqdm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import (
    VectorizableTextQuery,
    QueryType,
    QueryCaptionType,
    QueryAnswerType,
)
from beir.retrieval.evaluation import EvaluateRetrieval

from src.pipeline.promptEngineering.prompt_manager import PromptManager
from src.aoai.aoai_helper import AzureOpenAIManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
nest_asyncio.apply()

class BenchmarkOrchestrator:
    def __init__(self, search_client: SearchClient, dataset_dir: str):
        """
        Initialize the BenchmarkOrchestrator with shared attributes.
        """
        self.search_client = search_client
        self.dataset_dir = dataset_dir
        self.root = os.path.dirname(os.getcwd())
        self.aoai_client = AzureOpenAIManager()
        self.prompt_manager = PromptManager()
        self.block_size = 5
        self.semaphore = asyncio.Semaphore(10)  # Adjust based on your quota

        # Define the use cases for the benchmark class
        self.use_cases =  [
            {
                "diagnosis": "Inflammatory Bowel Disease (Crohn’s)",
                "medication": "Adalimumab",
                "title": "001.pdf"
            },
            {
                "diagnosis": "Lennox-Gastaut Syndrome",
                "medication": "Epidiolex",
                "title": "002.pdf"
            },
            {
                "diagnosis": "Lymphoblastic Leukemia (B-ALL), Philadelphia chromosome-negative",
                "medication": "Blinatumomab",
                "title": "003.pdf"
            },
            {
                "diagnosis": "Severe Atopic Dermatitis",
                "medication": "Dupilumab",
                "title": "004.pdf"
            },
            {
                "diagnosis": "High-grade Osteosarcoma",
                "medication": "Everolimus",
                "title": "005.pdf"
            }
        ]

        # Define the response format for relevance responses
        self.response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "relevance_scoring_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "document": {"type": "string"},
                                    "relevant": {
                                        "type": "integer",
                                        "enum": [0, 1],
                                        "description": "Binary indicator for relevance"
                                    },
                                    "relevant_score": {
                                        "type": "integer",
                                        "description": "Score from 1 (least relevant) to 5 (most relevant)"
                                    },
                                    "thought_chain": {
                                        "type": "string",
                                        "description": "Reasoning behind the evaluation"
                                    },
                                    "explanation": {
                                        "type": "string",
                                        "description": "User-facing explanation"
                                    }
                                },
                                "required": [
                                    "query", "document", "relevant",
                                    "relevant_score", "thought_chain", "explanation"
                                ],
                                "additionalProperties": False,
                            }
                        }
                    },
                    "required": ["results"],
                    "additionalProperties": False,
                },
                "strict": True
            }
        }

    def generate_corpus(self, run_corpus=True):
        """
        Generate a corpus from the search client and save it as a JSONL file.
        """
        if not run_corpus:
            print("generate_corpus generation is disabled (run_corpus=False).")
            return
        results = self.search_client.search("*", include_total_count=True)
        corpus_path = os.path.join(self.root, self.dataset_dir, "corpus.jsonl")
        with open(corpus_path, "w", encoding="utf-8") as f:
            for i, result in enumerate(results, start=1):
                # Remove keys starting with "@search"
                filtered_result = {k: v for k, v in result.items() if not k.startswith("@search")}
                filtered_result["id"] = f"d{i}"
                json_line = json.dumps(filtered_result, separators=(',', ':'), sort_keys=True)
                f.write(json_line + "\n")
        print(f"Corpus saved to {corpus_path}")

    async def generate_queries(self, use_cases: list = None, run_queries: bool = True):
        """
        Generate queries from provided use cases and save them as a JSONL file.
        """
        if not run_queries:
            print("Query generation is disabled (run_queries=False).")
            return

        if use_cases is None:
            use_cases = self.use_cases

        queries_flattened = []
        # Loop over each use case and generate queries via the prompt manager and AOAI client.
        for use_case in use_cases:
            diagnosis = use_case["diagnosis"]
            medication = use_case["medication"]
            title = use_case["title"]

            user_prompt = self.prompt_manager.create_prompt_query_generation_eval_user(
                diagnosis=diagnosis, medication=medication
            )
            system_prompt = self.prompt_manager.create_prompt_query_generation_eval_system()

            max_retries = 3
            retry_delay = 1  # seconds
            for attempt in range(max_retries):
                try:
                    query_response = await self.aoai_client.generate_chat_response(
                        query=user_prompt,
                        system_message_content=system_prompt,
                        image_paths=[],  # No images are needed for this prompt.
                        stream=False,
                        response_format="text",
                    )
                    # Attempt to parse the response JSON.
                    queries = json.loads(query_response['response'])['queries']
                    break
                except json.JSONDecodeError as e:
                    print(f"Attempt {attempt + 1} - JSON decode error in query generation for diagnosis: {diagnosis}, medication: {medication}: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                    else:
                        print("Max retries reached for query generation. Skipping this use case.")
                        queries = []
                        break

            # Merge the original use case fields with the generated queries.
            for query in queries:
                query_object = {
                    "diagnosis": diagnosis,
                    "medication": medication,
                    "title": title,
                    "query": query
                }
                queries_flattened.append(query_object)

        queries_path = os.path.join(self.root, self.dataset_dir, "queries.jsonl")
        with open(queries_path, "w", encoding="utf-8") as f:
            for i, query in enumerate(queries_flattened, start=1):
                query["id"] = f"q{i}"
                json_line = json.dumps(query, separators=(',', ':'))
                f.write(json_line + "\n")
        print(f"Queries saved to {queries_path}")

    async def generate_qrels(self, corpus_data: list[dict[str,str]], queries_data: list[dict[str,str]], run_relevance: bool = True):
        """
        Generate qrels (relevance judgements) by processing queries against the corpus.
        """
        if not run_relevance:
            print("Qrels generation is disabled (run_relevance=False).")
            return

        # Process queries asynchronously.
        qrels_flattened = await self._process_queries(queries_data, corpus_data)

        # Generate output file.
        qrels_path = os.path.join(self.root, self.dataset_dir, "qrels.jsonl")
        with open(qrels_path, "w", encoding="utf-8") as f:
            for qrel in qrels_flattened:
                json_line = json.dumps(qrel, separators=(',', ':'), sort_keys=True)
                f.write(json_line + "\n")
        print(f"Qrels saved to {qrels_path}")

    async def _process_block(self, query_obj: dict, corpus_block: list) -> dict:
        """
        Process a block of corpus documents for a given query.
        """
        query_id = query_obj["id"]
        query_text = query_obj["query"]
        query_title = query_obj.get("title", "")

        # Split the block into matching and non-matching documents.
        matching_docs = [doc for doc in corpus_block if doc.get("title", "") == query_title]
        non_matching_docs = [doc for doc in corpus_block if doc.get("title", "") != query_title]

        results = []
        # For non-matching docs, add a default result record.
        for doc in non_matching_docs:
            results.append({
                "document": doc.get("id", ""),
                "explanation": "",
                "query": query_id,
                "relevant": 0,
                "relevant_score": 0,
                "thought_chain": ""
            })

        # Process matching docs via the AOAI client.
        if matching_docs:
            filtered_block = [{"chunk": doc["chunk"], "id": doc["id"]} for doc in matching_docs]
            corpus_str = json.dumps(filtered_block, indent=2)
            user_prompt = self.prompt_manager.create_prompt_query_relevance_eval_user(
                corpus=corpus_str,
                query_id=query_id,
                query=query_text,
                title=query_title,
            )
            system_prompt = self.prompt_manager.create_prompt_query_relevance_eval_system()
            async with self.semaphore:
                response_obj = await self.aoai_client.generate_chat_response_no_history(
                    query=user_prompt,
                    system_message_content=system_prompt,
                    image_paths=[],  # No images needed.
                    stream=False,
                    max_tokens=7500,
                    response_format=self.response_format
                )
                response = response_obj["response"]

            # Retry logic for parsing JSON response.
            max_retries = 3
            retry_delay = 1  # seconds
            for attempt in range(max_retries):
                try:
                    parsed_response = json.loads(response)
                    break
                except json.JSONDecodeError as e:
                    print(f"Attempt {attempt + 1} - JSON decode error for query {query_id}: {e}")
                    if attempt < max_retries - 1:
                        async with self.semaphore:
                            new_response_obj = await self.aoai_client.generate_chat_response_no_history(
                                query=user_prompt,
                                system_message_content=system_prompt,
                                image_paths=[],
                                stream=False,
                                max_tokens=10000,
                                response_format=self.response_format
                            )
                            response = new_response_obj["response"]
                        await asyncio.sleep(retry_delay)
                    else:
                        print(f"Max retries reached for query {query_id}. Returning empty results.")
                        parsed_response = {"results": []}
                        break
            results.extend(parsed_response.get("results", []))
        return {"results": results}

    async def _process_queries(self, queries_data: list, corpus_data: list) -> list:
        """
        Process all queries over the corpus using asynchronous tasks.
        """
        overall_progress = tqdm(total=len(queries_data), desc="Processing Queries")
        qrels_flattened = []

        for query_obj in queries_data:
            num_blocks = math.ceil(len(corpus_data) / self.block_size)
            block_progress = tqdm(total=num_blocks, desc=f"Query {query_obj['id']} Blocks", leave=False)
            query_qrels = []

            tasks = [
                asyncio.create_task(
                    self._process_block(query_obj, corpus_data[i:i + self.block_size])
                )
                for i in range(0, len(corpus_data), self.block_size)
            ]

            for future in asyncio.as_completed(tasks):
                parsed_response = await future
                query_qrels.extend(parsed_response["results"])
                block_progress.update(1)
            block_progress.close()
            overall_progress.update(1)
            qrels_flattened.extend(query_qrels)
        overall_progress.close()
        return qrels_flattened

    def extract_chunk_id(self, full_chunk_id: str) -> str:
        """
        Extract the secondary portion of the chunk ID.
        """
        parts = full_chunk_id.split("_", 1)
        return parts[1] if len(parts) > 1 else full_chunk_id

    def get_doc_id(self, chunk_id: str, corpus_data: list) -> str:
        """
        Return the document ID for a given chunk_id.
        """
        return [x['id'] for x in corpus_data
                if self.extract_chunk_id(x["chunk_id"]) == self.extract_chunk_id(chunk_id)][0]

    def run_search(self, query_text: str, mode: str, corpus_data: list, top_n: int = 50) -> dict:
        """
        Run a search for the given query text and mode, and return a ranking dictionary.
        Modes include "keyword", "hybrid-semantic", "hybrid", and "vector".
        """
        vector_query = VectorizableTextQuery(
            text=query_text,
            fields="vector",
            k_nearest_neighbors=5,
            weight=0.5
        )
        if mode == "keyword":
            results = self.search_client.search(
                search_text=query_text,
                top=top_n,
                query_type=QueryType.FULL,
            )
            sort_field = "@search.score"
        elif mode == "hybrid-semantic":
            results = self.search_client.search(
                search_text=query_text,
                vector_queries=[vector_query],
                query_type=QueryType.SEMANTIC,
                semantic_configuration_name="my-semantic-config",
                query_caption=QueryCaptionType.EXTRACTIVE,
                query_answer=QueryAnswerType.EXTRACTIVE,
                top=top_n,
            )
            sort_field = "@search.reranker_score"
        elif mode == "hybrid":
            results = self.search_client.search(
                search_text=query_text,
                vector_queries=[vector_query],
                query_type=QueryType.FULL,
                top=top_n,
            )
            sort_field = "@search.score"
        elif mode == "vector":
            results = self.search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                query_type=QueryType.FULL,
                top=top_n,
            )
            sort_field = "@search.score"
        else:
            raise ValueError("Invalid search mode provided.")

        ranking = {}
        for result in results:
            try:
                doc_id = self.get_doc_id(result["chunk_id"], corpus_data)
                # Try to get the score from an attribute or key.
                score_val = getattr(result, sort_field, None)
                if score_val is None:
                    score_val = result.get(sort_field, None)
                ranking[doc_id] = score_val
            except IndexError:
                print(
                    f"Error: Could not find document ID for chunk ID: {result['chunk_id']} and title: {result['title']}")
                continue
        return ranking

    def generate_rankings(self, run_rankings=True):
        """
        Generate rankings for each query in various modes and save them as JSONL files.
        """
        if not run_rankings:
            print("Rankings generation is disabled (run_rankings=False).")
            return

        corpus_path = os.path.join(self.root, self.dataset_dir, "corpus.jsonl")
        queries_path = os.path.join(self.root, self.dataset_dir, "queries.jsonl")

        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_data = [json.loads(line) for line in f if line.strip()]
        with open(queries_path, "r", encoding="utf-8") as f:
            queries_data = [json.loads(line) for line in f if line.strip()]

        # Define output files.
        output_keyword = os.path.join(self.root, self.dataset_dir, "rankings-keyword.jsonl")
        output_vector = os.path.join(self.root, self.dataset_dir, "rankings-vector.jsonl")
        output_hybrid = os.path.join(self.root, self.dataset_dir, "rankings-hybrid.jsonl")
        output_hybrid_semantic = os.path.join(self.root, self.dataset_dir, "rankings-hybrid-semantic.jsonl")

        with open(output_hybrid, "w", encoding="utf-8") as fout_hybrid, \
                open(output_hybrid_semantic, "w", encoding="utf-8") as fout_hybrid_semantic, \
                open(output_keyword, "w", encoding="utf-8") as fout_keyword, \
                open(output_vector, "w", encoding="utf-8") as fout_vector:
            for query in queries_data:
                query_id = query["id"]
                query_text = query["query"]

                # Run searches for each mode.
                ranking_hybrid = self.run_search(query_text, mode="hybrid", corpus_data=corpus_data)
                ranking_semantic = self.run_search(query_text, mode="hybrid-semantic", corpus_data=corpus_data)
                ranking_keyword = self.run_search(query_text, mode="keyword", corpus_data=corpus_data)
                ranking_vector = self.run_search(query_text, mode="vector", corpus_data=corpus_data)

                fout_hybrid.write(json.dumps({"query": query_id, "ranking": ranking_hybrid}) + "\n")
                fout_hybrid_semantic.write(json.dumps({"query": query_id, "ranking": ranking_semantic}) + "\n")
                fout_keyword.write(json.dumps({"query": query_id, "ranking": ranking_keyword}) + "\n")
                fout_vector.write(json.dumps({"query": query_id, "ranking": ranking_vector}) + "\n")
                print(f"Processed query {query_id}")

        print("Rankings files have been generated.")

    def generate_evaluation_metrics(self):
        """
        Load qrels and ranking files, evaluate retrieval metrics, and display the results.
        """
        qrels_path = os.path.join(self.root, self.dataset_dir, "qrels.jsonl")
        with open(qrels_path, "r", encoding="utf-8") as f:
            qrels_data = [json.loads(line) for line in f if line.strip()]

        # Convert list of qrels into a nested dict.
        qrels_dict = {
            q: {entry["document"]: int(entry["relevant"])
                for entry in qrels_data if entry["query"] == q}
            for q in {entry["query"] for entry in qrels_data}
        }

        ranking_files = {
            "BEIR-Keyword": "rankings-keyword.jsonl",
            "BEIR-Vector": "rankings-vector.jsonl",
            "BEIR-Hybrid (Keyword + Vector)": "rankings-hybrid.jsonl",
            "BEIR-Hybrid + Semantic": "rankings-hybrid-semantic.jsonl",
        }

        metrics_results = {}

        def load_run_dict(run_file: str) -> dict:
            path = os.path.join(self.root, self.dataset_dir, run_file)
            with open(path, "r", encoding="utf-8") as f:
                run_data = [json.loads(line) for line in f if line.strip()]
            return {entry["query"]: entry["ranking"] for entry in run_data}

        evaluator = EvaluateRetrieval()
        for method_name, run_file in ranking_files.items():
            run_dict = load_run_dict(run_file)
            results = evaluator.evaluate(qrels_dict, run_dict, k_values=[3, 10])
            metrics_results[method_name] = {
                "NDCG@3": results[0]["NDCG@3"],
                "NDCG@10": results[0]["NDCG@10"],
                "Recall@10": results[2]["Recall@10"]
            }

        df = pd.DataFrame(metrics_results, index=["NDCG@3", "NDCG@10", "Recall@10"])
        print("Evaluation Metrics:")
        print(df)

        # Plot the results as a grouped bar chart.
        metrics = df.index.tolist()
        methods = df.columns.tolist()
        x = np.arange(len(metrics))
        width = 0.25

        fig, ax = plt.subplots(figsize=(8, 5))
        for idx, method in enumerate(methods):
            ax.bar(x + idx * width, df[method], width, label=method)

        ax.set_ylabel('Score')
        ax.set_title('Evaluation Metrics per Retrieval Method')
        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.margins(x=0.1)
        plt.tight_layout()
        plt.show()


# BEIR Benchmark Dataset Creation 📊📚

This repository contains code and instructions for creating a custom BEIR benchmark dataset, including the generation of `corpus.json`, `queries.json`, and `qrels.json`. The process involves:

- **Corpus Generation:** Using OCR combined with metadata processing to extract text from PDF policy documents.
- **Query Generation:** Leveraging GPT-4 (via GPT-4o) to automatically generate candidate queries.
- **Qrels Generation:** Manual human review to ensure high-quality relevance judgments for the queries.

Below, you’ll find details on each step on how we created the dataset for search retrieval specifically for BEIR and MedIndexer.

---

## 1. Corpus Generation 📄🖨️

The corpus is built by processing your payor policy PDF documents using OCR (e.g., Azure Cognitive Services) to extract text. In addition, we incorporate metadata (like document titles, dates, etc.) to enrich the corpus structure.

### Example Workflow:
1. **OCR Processing:**  
   Use Azure’s Computer Vision API to read PDFs and extract text:
   ```python
   from azure.cognitiveservices.vision.computervision import ComputerVisionClient
   from msrest.authentication import CognitiveServicesCredentials
   import os, json, time

   # Azure credentials (replace with your own)
   subscription_key = "YOUR_SUBSCRIPTION_KEY"
   endpoint = "YOUR_ENDPOINT"
   client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

   # Directory with PDFs
   pdf_dir = "path/to/pdf_folder"
   corpus = {}

   for filename in os.listdir(pdf_dir):
       if filename.lower().endswith(".pdf"):
           file_path = os.path.join(pdf_dir, filename)
           with open(file_path, "rb") as pdf_file:
               read_response = client.read_in_stream(pdf_file, raw=True)
           operation_location = read_response.headers["Operation-Location"]
           operation_id = operation_location.split("/")[-1]

           while True:
               result = client.get_read_result(operation_id)
               if result.status not in ['notStarted', 'running']:
                   break
               time.sleep(1)

           text_lines = []
           if result.status == 'succeeded':
               for page in result.analyze_result.read_results:
                   for line in page.lines:
                       text_lines.append(line.text)
           full_text = "\n".join(text_lines)

           doc_id = os.path.splitext(filename)[0]
           title = doc_id.replace("_", " ").title()
           corpus[doc_id] = {
               "title": title,
               "text": full_text
           }

   with open("corpus.json", "w") as outfile:
       json.dump(corpus, outfile, indent=4)
   print("Corpus created successfully!")
   ```
2. **Metadata Enrichment:**  
   You may add additional metadata fields (e.g., publication date, source URL) to each document as needed.

---

## 2. Query Generation using GPT-4o 🤖💬

For generating candidate queries, we use GPT-4 (via GPT-4o). The model is prompted with document text to generate queries that users might ask.

### Recommended Prompt:
```text
"Generate a query that a user might use to find information from the following text: {document_text}"
```
Replace `{document_text}` with the content or a key excerpt of your document.

### Example Code:
```python
from transformers import GPT4ForConditionalGeneration, GPT4Tokenizer  # Hypothetical example

# Load GPT-4o model and tokenizer (update with actual API/client if available)
model_name = "gpt-4o"  # Replace with the actual model identifier
tokenizer = GPT4Tokenizer.from_pretrained(model_name)
model = GPT4ForConditionalGeneration.from_pretrained(model_name)

# Example document text (replace with your actual document text)
document_text = "This document outlines UHC's policy details, including coverage benefits, exclusions, and cost-sharing information."
prompt = f"Generate a query that a user might use to find information from the following text: {document_text}"

inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
outputs = model.generate(inputs, max_length=64, num_beams=5, early_stopping=True)

query_candidate = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated query candidate:", query_candidate)
```
*Note:* Adjust the model loading and API usage as per your environment and available libraries.

---

## 3. Qrels Generation ✅📝

The Qrels (query relevance judgments) file maps each query to one or more relevant document IDs. For high-quality evaluation, these are generated through manual human review. Each query in `queries.json` is associated with the relevant document(s) in `qrels.json`.

### Example Structure:
```json
{
  "q1": {"uhc_policy": 1},
  "q2": {"cigna_policy": 1},
  "q3": {"anthem_policy": 1},
  "q4": {"aetna_policy": 1},
  "q5": {"humana_policy": 1}
}
```
*Note:* Here, `1` indicates that the document is relevant to the query. Human reviewers ensure that each query accurately reflects the document content and information needs.

---

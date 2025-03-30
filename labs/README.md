## **🧪 Azure AI Search Labs**

Welcome to the **Azure AI Search Labs**! These hands-on exercises are designed to help you build, configure, and optimize Azure AI Search across various healthcare and enterprise use cases. Whether you're just starting out or looking to deepen your expertise, each lab offers step-by-step guidance, real-world datasets, and practical techniques that mirror production-grade retrieval systems.

---

### **🛠️ Lab Modules Overview**

Each lab focuses on a specific part of the search lifecycle—from index creation to evaluation—enabling you to master both fundamentals and advanced capabilities of Azure AI Search.

#### **🧪 Lab 01: Building Your Azure AI Search Index**

- **[🧾 Notebook – Creating Indexes with Azure AI Agent Service](lab-01-creation-indexes.ipynb)**  
  Learn how to design and configure your search index. This includes:
  - Creating a new index from scratch
  - Defining schema fields for structured and unstructured data
  - Enabling filtering, faceting, and sorting
  - Understanding indexer vs push model architectures

#### **🧪 Lab 02: Indexing Clinical Data**

- **[🧾 Notebook – Ingesting and Indexing Clinical Data](lab-02-indexing.ipynb)**  
  Dive into the data ingestion pipeline. This lab covers:
  - Preprocessing PDF, text, and image-based datasets
  - Applying schema-first indexing principles
  - Configuring skillsets and enrichments
  - Automating data flow from Azure Blob Storage to Azure AI Search

#### **🧪 Lab 03: Retrieval Methods**

- **[🧾 Notebook – Exploring Vector Search and Hybrid Retrieval](lab-03-retrieval.ipynb)**  
  Learn to implement modern retrieval techniques, including:
  - Full-text search and lexical scoring
  - Vector similarity search using Azure OpenAI embeddings
  - Hybrid search (combining lexical + vector)
  - Reranking with semantic scoring

#### **🧪 Lab 04: Evaluating Search Quality**

- **[🧾 Notebook – Evaluating Search Quality and Relevance](lab-04-evaluation.ipynb)**  
  Assess the performance and relevance of your search results:
  - Define ground truth datasets for evaluation
  - Use precision, recall, and NDCG metrics
  - Analyze false positives/negatives
  - Tune your index and reranker configuration based on feedback

### **📚 Additional Notes**

- 💡 **Self-Contained Labs**: Each lab is modular and self-contained, allowing you to explore one concept at a time or run them in sequence for a complete experience.
- ⚙️ **Prerequisites**: Ensure you've met the requirements listed at the beginning of each notebook—such as Azure resource setup, authentication, and dependencies.

### **🚀 What’s Next?**

Once you’ve completed the labs, head over to the [**🏥 Use Cases**](../usecases/README.md) section to see how these foundational concepts are applied in real-world healthcare workflows. You’ll explore how to build policy knowledge stores, enable X-ray similarity search, and create intelligent retrieval pipelines that drive clinical impact.


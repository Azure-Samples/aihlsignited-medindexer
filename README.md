<!-- markdownlint-disable MD033 -->

## **🤖 MedIndexer: Turning Unstructured Clinical Data into Value**

> This project is **part of the [HLS Ignited Program](https://github.com/microsoft/aihlsIgnited)**, which focuses on hands-on accelerators with the goal of democratizing how AI is transforming the healthcare industry. In this project, we accelerate healthcare providers and payer organizations in building **AI-driven clinical knowledge bases** using **Azure AI Search**. Visit the program page to explore other accelerators.

<img src="utils/images/medIndexer.png" align="right" height="200" style="float:right; height:200px;" />

**MedIndexer** is an **indexing framework** designed for the **automated creation of structured knowledge bases** from unstructured clinical sources. It enables the transformation of **images (X-rays), clinical text, and other unstructured data** into a **schema-driven, searchable format**, allowing your applications to leverage **state-of-the-art retrieval methodologies**, including **vector search and re-ranking**, powered by **Azure AI Search**. By applying a **well-defined schema** and vectorizing the data into **high-dimensional representations**, MedIndexer empowers AI applications to retrieve **more precise and context-aware information**.

### **🔍 Turning Your Unstructured Data into Value**

> "About 80% of medical data remains unstructured and untapped after it is created (e.g., text, image, signal, etc.)"  
> — *Healthcare Informatics Research, Chungnam National University*

In the era of AI, the rise of AI copilots and assistants has led to a shift in how we access knowledge. But retrieving clinical data that lives in disparate formats is no trivial task. Building retrieval systems takes effort—and **how you structure your knowledge store matters**. It’s a cyclic, iterative, and constantly evolving process. That’s why we believe in leveraging **enterprise-ready retrieval platforms** like **Azure AI Search**—designed to power intelligent search experiences across structured and unstructured data. It serves as the foundation for building advanced retrieval systems in healthcare.

<br>

<div align="center">
  <img src="utils/images/The need of Knoledge.png" alt="Solution Diagram" width="80%" />
</div>

<br>

However, implementing **Azure AI Search** alone is not enough. Mastering its capabilities and applying well-defined patterns can significantly enhance your ability to address repetitive tasks and complex retrieval scenarios. This project aims to **accelerate your ability to transform raw clinical data into high-fidelity, high-value knowledge structures** that can power your next-generation healthcare applications.

### 🚀 How to Get Started

If you're new to Azure AI Search, start with the step-by-step labs to build a solid foundation in the technology. Experienced AI engineers can jump straight to the use case sections, which showcase how to create **Coded Policy Knowledge Stores** and **X-ray Knowledge Stores** for real-world applications.

#### 🧪 [Azure AI Search Labs](labs\README.md)

+ 🧪 **Building Your Azure AI Search Index**: [🧾 Notebook - Building Single Agents with Azure AI Agent Service](labs\lab-01-creation-indexes.ipynb) Learn how to create and configure an Azure AI Search index to enable intelligent search capabilities for your applications.
- 🧪 **Indexing Data into Azure AI Search**: [🧾 Notebook - Ingest and Index Clinical Data](labs\lab-02-indexing.ipynb) Understand how to ingest, preprocess, and index clinical data into Azure AI Search using schema-first principles.
+ 🧪 **Retrieval Methods for Azure AI Search**: [🧾 Notebook - Exploring Vector Search and Hybrid Retrieval](labs\lab-03-retrieval.ipynb) Dive into retrieval techniques such as vector search, hybrid retrieval, and reranking to enhance the accuracy and relevance of search results.
- 🧪 **Evaluation Methods for Azure AI Search**: [🧾 Notebook - Evaluating Search Quality and Relevance](labs\lab-04-evaluation.ipynb) Learn how to evaluate the performance of your search index using relevance metrics and ground truth datasets to ensure high-quality search results.

#### 🏥 [Use Cases](usecases\README.md)

+ **📝 Creating Coded Policy Knowledge Stores**: [🧾 Notebook - Creating Coded Policies Knowledge Stores](usecases/usecase-01-creating-coded-policies-knowledge-stores.ipynb) Transform payer policies into **machine-readable formats**. 
    This use case includes:  
     - Preprocessing and cleaning PDF documents  
     - Building custom OCR skills  
     - Leveraging out-of-the-box Indexer capabilities and embedding skills  
     - Enabling real-time AI-assisted querying for ICDs, payer names, drug names, and policy logic  

   **Why it matters**:  
   Streamlines prior authorization and coding workflows for providers and payors, reducing manual effort and increasing transparency.

+ **🩻 Creating X-ray Knowledge Stores**: [🧾 Notebook - Creating X-rays Knowledge Stores](usecases/usecase-02-creating-x-rays-knowledge-stores.ipynb) Turn imaging reports and metadata into a searchable knowledge base. 
    This includes:  
     - Leveraging push APIs with custom event-driven indexing pipeline triggered on new X-ray uploads  
     - Generating embeddings using Microsoft Healthcare foundation models  
     - Providing an AI-powered front-end for **X-ray similarity search**  

   **Why it matters**:  
   Supports clinical decision-making by retrieving similar past cases, aiding diagnosis and treatment planning with contextual relevance.

## **📚 More Resources**

- **[Azure AI Foundry](https://azure.microsoft.com/en-us/products/ai-foundry/?msockid=0b24a995eaca6e7d3c1dbc1beb7e6fa8#Use-cases-and-Capabilities)** – Develop and deploy custom AI apps and APIs responsibly with a comprehensive platform.
- **[Azure AI Search Quick Start](https://azure.microsoft.com/en-us/products/ai-services/ai-search)** – Overview of Azure AI Search features, pricing, and customer stories.
- **[Azure AI Search Documentation](https://learn.microsoft.com/en-us/azure/search/)** – Comprehensive documentation covering concepts, tutorials, quickstarts, and how-to guides for Azure AI Search.
- **[Azure AI Search REST API Reference](https://learn.microsoft.com/en-us/rest/api/searchservice/)** – Detailed reference for the REST APIs provided by Azure AI Search, including operations on indexes, documents, and queries.
- **[Azure AI Search Python SDK](https://pypi.org/project/azure-search-documents/)** – Official Python client library for Azure AI Search, enabling seamless integration into Python applications.

<br>

> [!IMPORTANT]  
> This software is provided for demonstration purposes only. It is not intended to be relied upon for any production workload. The creators of this software make no representations or warranties of any kind, express or implied, about the completeness, accuracy, reliability, suitability, or availability of the software or related content. Any reliance placed on such information is strictly at your own risk.

# **🏥 Use Cases**

Welcome to the **Use Cases** section! This part of the repository demonstrates real-world applications of **Azure AI Search** in healthcare. By transforming unstructured data—such as clinical notes, imaging reports, and payer policies—into **structured, queryable knowledge stores**, organizations can build intelligent solutions for clinical decision support, policy validation, and similarity search.

#### **🔍 Why These Use Cases Matter**

Healthcare data is largely unstructured and siloed, making it hard to access, analyze, and act on. These use cases solve that problem by automating the **conversion of unstructured data into structured formats** that are easy to query and integrate into applications.

We focus on two high-value patterns:

1. **Creating Coded Policy Knowledge Stores**  
   Streamlines the retrieval of payer coverage guidelines, improving accuracy in prior authorization workflows.

2. **Creating X-ray Knowledge Stores**  
   Enables intelligent search over radiology reports and imaging metadata, powering clinician-facing tools like **X-ray similarity search**.

Together, these use cases highlight the importance of **knowledge store engineering** in modern healthcare.


### **📝 Creating Coded Policy Knowledge Stores**  

In many healthcare systems, policy documents such as pre-authorization guidelines are still trapped in static, scanned PDFs. These documents are critical—they contain ICD codes, drug name coverage, and payer-specific logic—but are rarely structured or accessible in real-time. To solve this, we built a pipeline that transforms these documents into intelligent, searchable knowledge stores.

<img src="..\utils\images\Turning Policies into Knowledge Store.png" style="display: block; margin: 20px auto; border-radius: 15px; max-width: 80%; height: auto;" alt="Turning Policies into Knowledge Store" />

*This diagram shows how pre-auth policy PDFs are ingested via blob storage, passed through an OCR and embedding skillset, and then indexed into Azure AI Search. The result: fast access to coded policy data for AI apps.*

   - [🧾 Notebook - Creating Coded Policies Knowledge Stores](usecases/usecase-01-creating-coded-policies-knowledge-stores.ipynb)  
   Transform payer policies into **machine-readable formats**. This use case includes:  
     - Preprocessing and cleaning PDF documents  
     - Building custom OCR skills  
     - Leveraging out-of-the-box Indexer capabilities and embedding skills  
     - Enabling real-time AI-assisted querying for ICDs, payer names, drug names, and policy logic  

   **Why it matters**:  
   Streamlines prior authorization and coding workflows for providers and payors, reducing manual effort and increasing transparency.

### **🩻 Creating X-ray Knowledge Stores** 

In radiology workflows, X-ray reports and image metadata contain valuable clinical insights—but these are often underutilized. Traditionally, they’re stored as static entries in PACS systems or loosely connected databases. The goal of this use case is to turn those X-ray reports into a searchable, intelligent asset that clinicians can explore and interact with in meaningful ways.

<img src="..\utils\images\Turning Xrays into Knowledge.png" style="display: block; margin: 20px auto; border-radius: 15px; max-width: 80%; height: auto;" alt="Turning X-rays into Actionable Knowledge" />

*This diagram illustrates a full retrieval pipeline where radiology reports are uploaded, enriched through foundational models, embedded, and indexed. The output powers an AI-driven web app for similarity search and decision support.*

   - [🧾 Notebook - Creating X-rays Knowledge Stores](usecases/usecase-02-creating-x-rays-knowledge-stores.ipynb)  
   Turn imaging reports and metadata into a searchable knowledge base. This includes:  
     - Leveraging push APIs with custom event-driven indexing pipeline triggered on new X-ray uploads  
     - Generating embeddings using Microsoft Healthcare foundation models  
     - Providing an AI-powered front-end for **X-ray similarity search**  

   **Why it matters**:  
   Supports clinical decision-making by retrieving similar past cases, aiding diagnosis and treatment planning with contextual relevance.


### 🧠 Retrieval is the New Intelligence Layer

These use cases prove that **retrieval-first thinking** can dramatically improve how clinicians and staff interact with AI systems. Instead of relying on traditional databases or manual lookups, information is **embedded, indexed, and ready for semantic access**—just like asking a colleague, but faster and more consistent.

Together, these pipelines form the foundation for **agentic and multimodal workflows** where AI copilots can navigate policies and imaging with equal intelligence.

> Ready to build your own knowledge store?  
> These use cases serve as blueprints for building high-value, search-powered healthcare applications using Azure AI Search.
> You can customize and extend them to support other clinical domains such as pathology reports, EHR notes, surgical records, and more.
> Explore the notebooks, extend the pipelines, and join the healthcare AI transformation—one index at a time.

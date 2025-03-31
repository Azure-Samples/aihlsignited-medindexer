import json
import os
import pickle
from io import BytesIO
from typing import Any, Dict, Union

import numpy as np
import requests
from azure.search.documents.models import VectorizedQuery

import pandas as pd
import yaml
from azure.storage.blob import ContentSettings
from matplotlib import pyplot as plt
from PIL import Image as PILImage
from src.utils.io import get_blob_sas_url, read_dicom_to_array, normalize_image_to_uint8, numpy_to_image_bytearray
from src.utils.ml_logging import get_logger

def display_precision_table(avg_precision):
    """
    Given a dictionary of average precision scores per label, create and display a table.
    Example of avg_precision:
      {
          'cardiomegaly': {1: 0.80, 3: 0.65, 5: 0.60},
          'no findings': {1: 0.90, 3: 0.70, 5: 0.65},
          ...
      }
    """
    # Convert dictionary to DataFrame; labels will be the rows.
    df = pd.DataFrame(avg_precision).T
    # Sort the columns to be in the order of k=1,3,5
    df = df[[1, 3, 5]]
    # Rename columns for better readability
    df.columns = ['Precision@1', 'Precision@3', 'Precision@5']

    # Create a matplotlib figure to display the table
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values,
                     rowLabels=df.index,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.show()


def evaluate_precision_at_k(search_client, documents, ks=[1, 3, 5]):
    labels = ['Cardiomegaly', 'No Finding',
              'Pleural Effusion', 'Atelectasis',
              'Support Devices']
    # Initialize a dictionary to store all precision scores per label and k
    precision_scores = {label: {k: [] for k in ks} for label in labels}

    for doc in documents:
        query_label = doc.get("label_category")
        if query_label not in labels:
            continue  # Skip documents that are not part of our evaluation labels

        query_vector = doc['image_vector']
        # Use maximum k + 1 (to account for the query itself possibly appearing in results)
        max_k = max(ks)
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=max_k + 1,  # extra one in case the query is returned
            fields="image_vector"
        )

        # Perform the vector search
        search_results = list(search_client.search(search_text=None, vector_queries=[vector_query]))

        # Filter out the source document from the results
        filtered_results = [r for r in search_results if r.get("name") != doc.get("name")]

        # For each desired k, compute precision@k
        for k in ks:
            top_k = filtered_results[:k]
            # Count how many of the top k results have the same label as the query
            relevant_count = sum(1 for result in top_k if result.get("label_category") == query_label)
            precision = relevant_count / k
            precision_scores[query_label][k].append(precision)

    # Calculate the average precision@k per label
    avg_precision = {}
    for label, scores in precision_scores.items():
        avg_precision[label] = {k: round(np.mean(scores[k]), 2) if scores[k] else 0.0 for k in ks}

    return avg_precision


def display_source_and_image_search_results(documents, doc_index: int, search_client, k: int = 5):
    """
    Given a document index from the documents list, this function:
      1. Retrieves the source DICOM image using its blob URL, converts it to an image,
         and displays it.
      2. Uses the document's embedding vector to perform a vector search in Azure Cognitive Search.
      3. Displays the source image and the top k similar images (excluding the source),
         annotated with rank, label, and label_category.

    The DICOM-to-image conversion uses:
       - read_dicom_to_array to read the DICOM bytes,
       - normalize_image_to_uint8 to normalize the pixel values,
       - numpy_to_image_bytearray to convert the array to a PNG byte stream.
    """
    # Select the source document
    source_doc = documents[doc_index]
    source_blob_url = source_doc['blob_url']
    source_vector = source_doc['image_vector']
    source_label_category = source_doc.get("label_category", "Unknown Category")

    # Generate a SAS URL for the source DICOM blob
    sas_source_url = get_blob_sas_url(source_blob_url)

    # Download the DICOM bytes from blob
    response = requests.get(sas_source_url)
    if response.status_code == 200:
        # Use your helper function to read the DICOM bytes to a NumPy array
        dicom_array = read_dicom_to_array(response.content, engine="sitk")
        # Normalize the array to uint8 (you may adjust parameters as needed)
        normalized_array = normalize_image_to_uint8(dicom_array)
        # Convert the normalized NumPy array to image bytes (PNG format)
        img_bytes = numpy_to_image_bytearray(normalized_array, format="PNG")
        # Open the image with PIL
        source_img = PILImage.open(BytesIO(img_bytes))
    else:
        print(f"Failed to retrieve source image. Status code: {response.status_code}")
        return

    # Build a vector query using the source document's vector
    vector_query = VectorizedQuery(
        vector=source_vector,
        k_nearest_neighbors=k+1,
        fields="image_vector"
    )

    # Execute the vector search
    search_results = search_client.search(search_text=None, vector_queries=[vector_query])
    result_docs = [doc for doc in search_results]

    # Filter out the source document by comparing the normalized 'name' field
    filtered_results = [doc for doc in result_docs if doc.get("name") != source_doc["name"]]
    top_results = filtered_results[:k]

    # Create a matplotlib figure with subplots for source and result images
    total_images = 1 + len(top_results)
    fig, axes = plt.subplots(1, total_images, figsize=(4 * total_images, 4))
    if total_images == 1:
        axes = [axes]

    # Display the source image in the first subplot
    axes[0].imshow(source_img, cmap="gray")
    axes[0].set_title(f"Source Image\nCategory: {source_label_category}")
    axes[0].axis("off")

    # Display each result: download its DICOM, process, and show with annotations
    for i, doc in enumerate(top_results):
        result_blob_url = doc.get("blob_url")
        sas_url = get_blob_sas_url(result_blob_url)
        r = requests.get(sas_url)
        if r.status_code == 200:
            dicom_array_res = read_dicom_to_array(r.content, engine="sitk")
            norm_array_res = normalize_image_to_uint8(dicom_array_res)
            img_bytes_res = numpy_to_image_bytearray(norm_array_res, format="PNG")
            result_img = PILImage.open(BytesIO(img_bytes_res))
        else:
            print(f"Failed to retrieve image for result {i + 1}. Status code: {r.status_code}")
            continue

        label = doc.get("label")
        label_category = doc.get("label_category")
        axes[i + 1].imshow(result_img, cmap="gray")
        axes[i + 1].set_title(f"Rank {i + 1}\n{label_category}\n{label}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()

def display_source_and_text_search_results(text_search, search_client, aoai_client, k: int = 5):
    """
    Given a document index from the documents list, this function:
      1. Retrieves the source DICOM image using its blob URL, converts it to an image,
         and displays it.
      2. Uses the document's embedding vector to perform a vector search in Azure Cognitive Search.
      3. Displays the source image and the top k similar images (excluding the source),
         annotated with rank, label, and label_category.

    The DICOM-to-image conversion uses:
       - read_dicom_to_array to read the DICOM bytes,
       - normalize_image_to_uint8 to normalize the pixel values,
       - numpy_to_image_bytearray to convert the array to a PNG byte stream.
    """
    # Select the source document
    text_vector_response = aoai_client.generate_embedding(text_search)
    source_vector = json.loads(text_vector_response)['data'][0]['embedding']

    # Build a vector query using the source document's vector
    vector_query = VectorizedQuery(
        vector=source_vector,
        k_nearest_neighbors=k,
        fields="text_vector"
    )

    # Execute the vector search
    search_results = search_client.search(search_text=None, vector_queries=[vector_query])
    result_docs = [doc for doc in search_results]

    # Filter out the source document by comparing the normalized 'name' field
    filtered_results = [doc for doc in result_docs]
    top_results = filtered_results[:k]

    # Create a matplotlib figure with subplots for source and result images
    total_images = 1 + len(top_results)
    fig, axes = plt.subplots(1, total_images, figsize=(4 * total_images, 4))
    if total_images == 1:
        axes = [axes]

    # Display the source image in the first subplot
    axes[0].set_title(f"Source Search\nText: {text_search}")
    axes[0].axis("off")

    # Display each result: download its DICOM, process, and show with annotations
    for i, doc in enumerate(top_results):
        result_blob_url = doc.get("blob_url")
        sas_url = get_blob_sas_url(result_blob_url)
        r = requests.get(sas_url)
        if r.status_code == 200:
            dicom_array_res = read_dicom_to_array(r.content, engine="sitk")
            norm_array_res = normalize_image_to_uint8(dicom_array_res)
            img_bytes_res = numpy_to_image_bytearray(norm_array_res, format="PNG")
            result_img = PILImage.open(BytesIO(img_bytes_res))
        else:
            print(f"Failed to retrieve image for result {i + 1}. Status code: {r.status_code}")
            continue

        label = doc.get("label")
        label_category = doc.get("label_category")
        axes[i + 1].imshow(result_img, cmap="gray")
        axes[i + 1].set_title(f"Rank {i + 1}\n{label_category}\n{label}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()


def get_best_image_by_precision_at_5(search_client, documents, k=5):
    """
    Iterate over the documents, perform a vector search for each, and compute precision@5.
    Returns the document (image) with the highest precision@5 score and its precision.
    """
    best_precision = -1.0
    best_doc = None
    # You can use your evaluation labels to restrict the set if needed
    eval_labels = ['Cardiomegaly', 'No Finding', 'Pleural Effusion', 'Atelectasis', 'Support Devices']

    for doc in documents:
        query_label = doc.get("label_category")
        if query_label not in eval_labels:
            continue

        query_vector = doc['image_vector']
        # Request one extra result to account for the query itself possibly appearing in the results
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=k + 1,
            fields="image_vector"
        )

        # Perform the vector search
        search_results = list(search_client.search(search_text=None, vector_queries=[vector_query]))
        # Remove the source document
        filtered_results = [r for r in search_results if r.get("name") != doc.get("name")]

        # Take the top k results
        top_k = filtered_results[:k]
        # Calculate how many of the top k have the same label as the query document
        relevant_count = sum(1 for result in top_k if result.get("label_category") == query_label)
        precision = relevant_count / k

        # Keep track of the best-performing document
        if precision > best_precision:
            best_precision = precision
            best_doc = doc

    return best_doc, best_precision
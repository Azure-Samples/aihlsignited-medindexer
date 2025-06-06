import base64
import io
import os
import tempfile
import warnings
from datetime import datetime, timedelta
from io import BytesIO
import glob
import re

from typing import Union

import nibabel as nib
import numpy as np
import pydicom
import requests
import SimpleITK as sitk
import torch
from PIL import Image
from azure.storage.blob import ContentSettings, BlobClient, generate_blob_sas, BlobSasPermissions

from tqdm import tqdm
from collections.abc import Iterable

import magic

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def read_dicom_to_array(dicom_input, engine="sitk"):
    """Reads a DICOM file or bytes and returns the image array using the specified engine."""
    if engine not in ["pydicom", "sitk"]:
        raise ValueError("Unsupported engine. Use 'pydicom' or 'sitk'.")

    if isinstance(dicom_input, bytes) and engine == "sitk":
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as temp_file:
            temp_file.write(dicom_input)
            temp_file.flush()
            dicom_input = temp_file.name

    if engine == "sitk":
        return _read_dicom_sitk(dicom_input)
    elif engine == "pydicom":
        return _read_dicom_pydicom(dicom_input)


def _read_dicom_sitk(dicom_input, squeeze=True, suppress_warnings=False):
    try:
        if suppress_warnings:
            original_warning_state = sitk.ProcessObject_GetGlobalWarningDisplay()
            sitk.ProcessObject_SetGlobalWarningDisplay(False)
        image = sitk.ReadImage(dicom_input)
        image_array = sitk.GetArrayFromImage(image).astype(np.float32)

        if squeeze:
            if not suppress_warnings and image_array.shape[0] > 1:
                warnings.warn(
                    f"Squeezing the first dimension of size {image_array.shape[0]}"
                )
            image_array = image_array[0, :, :]

        return image_array
    finally:
        if suppress_warnings:
            sitk.ProcessObject_SetGlobalWarningDisplay(original_warning_state)


def _read_dicom_pydicom(dicom_input):
    ds = pydicom.dcmread(dicom_input)
    rescale_slope = getattr(ds, "RescaleSlope", 1)
    rescale_intercept = getattr(ds, "RescaleIntercept", 0)
    image_array = ds.pixel_array * rescale_slope + rescale_intercept
    return image_array


def normalize_image_to_uint8(image_array, window=None, percentiles=None, min_max=None):
    """Normalize a DICOM image array to uint8 format.

    Args:
        image_array (np.ndarray): The input image array.
        window (tuple, optional): A tuple (window_center, window_width) for windowing.
        percentiles (tuple or float, optional): A tuple (low_percentile, high_percentile) or a single float for percentile normalization.
        min_max (tuple, optional): A tuple (min_val, max_val) for min-max normalization. If None, use the image array's min and max.

    Returns:
        np.ndarray: The normalized image array in uint8 format.
    """
    # Ensure only one of the optional parameters is provided
    if sum([window is not None, percentiles is not None, min_max is not None]) > 1:
        raise ValueError(
            "Only one of 'window', 'percentiles', or 'min_max' must be specified."
        )

    if window:
        window_center, window_width = window
        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2
    elif percentiles:
        if isinstance(percentiles, float):
            low_percentile, high_percentile = percentiles, 1 - percentiles
        else:
            low_percentile, high_percentile = percentiles
        min_val = np.percentile(image_array, low_percentile * 100)
        max_val = np.percentile(image_array, high_percentile * 100)
    elif min_max is not None and isinstance(min_max, Iterable) and len(min_max) == 2:
        min_val, max_val = min_max
    else:
        min_val = np.min(image_array)
        max_val = np.max(image_array)

    image_array = np.clip(image_array, min_val, max_val)
    image_array = (image_array - min_val) / (max_val - min_val) * 255.0

    return image_array.astype(np.uint8)


def numpy_to_image_bytearray(image_array: np.ndarray, format: str = "PNG") -> bytes:
    """Convert a NumPy array to an image byte array."""
    byte_io = BytesIO()
    pil_image = Image.fromarray(image_array)
    if pil_image.mode == "L":
        pil_image = pil_image.convert("RGB")
    pil_image.save(byte_io, format=format)
    return byte_io.getvalue()


def read_rgb_to_array(image_path):
    """Reads an RGB image and returns resized pixel data as a BytesIO buffer."""
    image_array = np.array(Image.open(image_path))
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    return image_array


def read_image_to_array(input_data: Union[bytes, str]) -> np.ndarray:
    """Reads an image from a file path or bytes and returns it as a numpy array."""
    if isinstance(input_data, bytes):
        return np.array(Image.open(BytesIO(input_data)))
    elif isinstance(input_data, str):
        return np.array(Image.open(input_data))
    else:
        raise ValueError("Input must be a file path (str) or file bytes.")


def read_file_to_bytes(input_data: Union[bytes, str]) -> bytes:
    """Reads a file from a file path or returns file bytes directly."""
    if isinstance(input_data, bytes):
        return input_data
    elif isinstance(input_data, str):
        with open(input_data, "rb") as f:
            return f.read()
    else:
        raise ValueError("Input must be a file path (str) or file bytes.")


def read_nifti(file_path):
    """
    Loads a NIfTI file and returns the volumetric data as a NumPy array.

    Parameters:
    - file_path (str): Path to the NIfTI file (.nii or .nii.gz).

    Returns:
    - numpy.ndarray: 3D array representing the NIfTI file's data.
    """
    nifti_image = nib.load(file_path)
    nifti_data = nifti_image.get_fdata()
    return nifti_data


def read_nifti_2d(image_path, slice_idx=0, HW_index=(0, 1), channel_idx=None):
    """Reads a NIFTI file and returns pixel data as a BytesIO buffer."""
    image_array = read_nifti(image_path)

    if HW_index != (0, 1):
        image_array = np.moveaxis(image_array, HW_index, (0, 1))

    if channel_idx is None:
        image_array = image_array[:, :, slice_idx]
    else:
        image_array = image_array[:, :, slice_idx, channel_idx]

    return image_array


def convert_volume_to_slices(volume, output_dir, filename_prefix):
    """
    Converts a 3D volume into 2D slice images and saves them as PNG files.

    Parameters:
    - volume (numpy.ndarray): 3D array representing the volumetric data.
    - output_dir (str): Directory where slice images will be saved.
    - filename_prefix (str): Prefix for each saved image file.

    Each slice is rotated 90° clockwise and flipped horizontally before saving.
    """
    for i in range(volume.shape[2]):
        slice_img = volume[:, :, i]
        slice_img = slice_img.astype(np.uint8)
        slice_img = (
            Image.fromarray(slice_img)
            .transpose(Image.ROTATE_90)
            .transpose(Image.FLIP_LEFT_RIGHT)
        )
        slice_img.save(os.path.join(output_dir, f"{filename_prefix}_slice{i}.png"))

def upload_dicom_to_blob(file_path: str, container_client, folder: str = "raw") -> str:
    """
    Uploads a file to Blob Storage into a specified folder.
    If the blob already exists, the function returns its URL without re-uploading.
    """
    file_name = os.path.basename(file_path)
    # Create a blob name that places the file under the "raw" folder
    blob_name = f"{folder}/{file_name}"
    blob_client = container_client.get_blob_client(blob_name)

    # Check if the blob already exists; if so, skip upload
    if blob_client.exists():
        return blob_client.url

    with open(file_path, "rb") as data:
        blob_client.upload_blob(
            data,
            overwrite=True,
            content_settings=ContentSettings(content_type="application/dicom")
        )
    return blob_client.url


def get_blob_sas_url(blob_url: str, expiry_hours: int = 1) -> str:
    """
    Given a blob URL, generate and return a SAS URL that appends a read-only token.
    The account key must be set in the environment variable 'AZURE_STORAGE_ACCOUNT_KEY'.
    """
    account_key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
    if not account_key:
        raise ValueError("AZURE_STORAGE_ACCOUNT_KEY environment variable is not set.")

    blob_client = BlobClient.from_blob_url(blob_url)
    sas_token = generate_blob_sas(
        account_name=blob_client.account_name,
        container_name=blob_client.container_name,
        blob_name=blob_client.blob_name,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=expiry_hours)
    )
    return blob_client.url + "?" + sas_token
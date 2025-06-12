import shutil
import zipfile
from pathlib import Path

import streamlit as st


def clear_uploads_dir(upload_dir: Path) -> None:
    """
    Remove and recreate the uploads directory so it’s empty.
    """
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)


def save_uploaded_files(uploaded_files, upload_dir: Path) -> Path:
    """
    Save uploaded .txt/.md/.pdf/.docx files (or ZIPs) into upload_dir.
    If a ZIP is uploaded, extract its contents and then delete the ZIP.
    Returns the Path to the folder with all extracted/saved files.
    """
    upload_dir.mkdir(parents=True, exist_ok=True)

    for uploaded_file in uploaded_files:
        fname = uploaded_file.name
        target_path = upload_dir / fname

        # Save the raw file
        with open(target_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # If it’s a ZIP, extract and remove it
        if fname.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(target_path, "r") as z:
                    z.extractall(upload_dir)
            except zipfile.BadZipFile:
                st.error(f"🚨 Failed to extract ZIP file: {fname}")
            finally:
                target_path.unlink()

    return upload_dir

import os
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path=r"C:\Users\Naseeka\Downloads\mathwriting-2024\mathwriting-2024",
    repo_id="Ntsako12/Mathenatics-handwriting-dataset",
    repo_type="dataset",
)
import urllib.request
import os
import zipfile

def download_movielens_small(base_path="data"):
    """Download MovieLens 100K dataset"""
    url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_filename = "ml-100k.zip"
    
    os.makedirs(base_path, exist_ok=True)
    zip_path = os.path.join(base_path, zip_filename)
    
    if not os.path.exists(zip_path):
        print(f"Downloading MovieLens dataset to {zip_path}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete!")
    else:
        print("Dataset zip already exists.")
    
    # Extract the zip file
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(base_path)
    
    print("Dataset extracted successfully!")

if __name__ == "__main__":
    download_movielens_small()

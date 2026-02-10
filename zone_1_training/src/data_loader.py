import pandas as pd
import urllib.request
import zipfile
from pathlib import Path

def load_retail_data():
    """
    Load the Online Retail II dataset from UCI Machine Learning Repository.
    
    Downloads and extracts the dataset if not already present locally.
    
    Returns:
        pd.DataFrame: The Online Retail II dataset, or None if download fails.
    """
    # Get the project root directory (FYP/)
    project_root = Path(__file__).resolve().parent.parent  # src/ -> cmab_data_pipeline/ -> FYP/
    data_dir = project_root /"data" / "raw"
    zip_path = data_dir / "online_retail_ii.zip"
    excel_path = data_dir / "online_retail_II.xlsx"
    
    # Create directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if not excel_path.exists():
        print("Downloading dataset...")
        zip_url = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
        try:
            urllib.request.urlretrieve(zip_url, zip_path)
            
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
                
            zip_path.unlink()  # Remove ZIP after extraction
            print("Dataset extracted successfully!")
        except Exception as e:
            print(f"Download failed: {e}")
            return None
    
    print("Loading dataset...")
    try:
        df = pd.read_excel(excel_path)
        print(f"Dataset loaded successfully! Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

if __name__ == "__main__":
    # Test the function when running directly
    df = load_retail_data()
    if df is not None:
        print(df.head())
        print(df.info())
import kagglehub

# Download latest version
path = kagglehub.dataset_download("jisongxiao/synthetic-fraud-dataset-medium")

print("Path to dataset files:", path)
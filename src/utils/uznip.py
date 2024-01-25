import zipfile

def unzip_file(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Example usage
zip_file_path = 'processed.zip'
extract_to_path = 'data/'

unzip_file(zip_file_path, extract_to_path)

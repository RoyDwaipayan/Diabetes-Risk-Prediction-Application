import gdown
import joblib

# URL from Google Drive Shareable Link (change to the "uc" format)
def download_model():
    url = "https://drive.google.com/drive/u/1/folders/1ehTsbtzeYVe0vslo3WSDhayj8HgWRDfb"
    output = "model.pkl"
    gdown.download(url, output, quiet=False)

    model = joblib.load(output)

    return model
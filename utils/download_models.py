import gdown


def download_model(model_name: str) -> None:
    """
    download model from google drive

    Parameters
    ----------
    model_name : str
        model to download(clip_image_encoder, distilbert-base-uncased, classifier)
    """
    # folder with models: https://drive.google.com/drive/folders/1-vNTWBpRgsf6HXsdd4ZTJgR7hlOprl6q

    base_url = "https://drive.google.com/uc?id="

    ids = {
        "clip_image_encoder": "1xf_i009SeIeswjZWVCIt7xkMWbVtAnCc",
        "distilbert-base-uncased": "1yElZ7y77c7HWNmCIpsqkV5H9c8PZRZE8",
        "classifier": "1M2xbzKV9ahso8Y1qTnBd2aW48hofWNu2"
    }

    if model_name not in ids.keys():
        raise ValueError("Invalid model name")

    output = f"models/models_onnx/{model_name}.onnx"
    gdown.download(base_url + ids[model_name], output, quiet=False)

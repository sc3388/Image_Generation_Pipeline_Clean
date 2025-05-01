import requests
from PIL import Image
from io import BytesIO

CREDENTIALS_PATH = "/content/drive/MyDrive/Image_Generation_Pipeline_code/credentials/google_keys.txt"

def load_api_key(path=CREDENTIALS_PATH, label="GPT"):
    key = None
    with open(path, "r") as f:
        for line in f:
            if line.startswith(f"{label}="):
                key = line.strip().split("=", 1)[1]
                break
    if not key:
        raise ValueError(f"Missing {label} API key in credentials file.")
    return key

def load_google_credentials(path=CREDENTIALS_PATH):
    api_key, cx = None, None
    with open(path, "r") as f:
        for line in f:
            if line.startswith("API_KEY="):
                api_key = line.strip().split("=", 1)[1]
            if line.startswith("CX="):
                cx = line.strip().split("=", 1)[1]
    if not api_key or not cx:
        raise ValueError("Missing API_KEY or CX in credentials file.")
    return api_key, cx

def search_google_cse_images(keywords, max_results=10):
    api_key, cx = load_google_credentials()
    query = " ".join(keywords)
    url = "https://www.googleapis.com/customsearch/v1"

    params = {
        "q": query,
        "cx": cx,
        "key": api_key,
        "searchType": "image",
        "num": max_results,
        "safe": "active",
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "items" not in data or len(data["items"]) == 0:
        print("❌ No image results from Google CSE.")
        return []

    image_list = []
    for item in data["items"]:
        img_url = item["link"]
        try:
            img_bytes = requests.get(img_url, headers={"User-Agent": "Mozilla/5.0"}).content
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            image_list.append(img)
        except Exception as e:
            print(f"⚠️ Failed to load image: {img_url} — {e}")

    print(f"✅ Retrieved {len(image_list)} images from Google CSE")
    return image_list



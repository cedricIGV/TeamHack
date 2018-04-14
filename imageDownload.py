from google_images_download import google_images_download

def main():
    response = google_images_download.googleimagesdownload()
    response.download({"keywords": "dog", "limit": 1000})

if __name__ == "__main__":
    main():

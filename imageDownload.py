from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()
response.download({"keywords": "dog", "limit": 1000})

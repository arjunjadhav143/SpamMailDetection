import nltk

print("Forcing a fresh download of NLTK 'stopwords' and 'punkt'...")
nltk.download('stopwords', force=True)
nltk.download('punkt', force=True)
print("Download complete.")
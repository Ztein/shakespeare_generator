import requests
import os

def download_shakespeare():
    """Download Shakespeare's complete works from Project Gutenberg"""
    # Check if file already exists
    data_file = 'data/shakespeare.txt'
    if os.path.exists(data_file):
        # Read from existing file
        print("Using existing file")
        with open(data_file, 'r', encoding='utf-8') as f:
            return f.read()
    print("File does not exist")
  
    # If file doesn't exist, download it
    url = "https://www.gutenberg.org/files/100/100-0.txt"
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Download the file
    print("Downloading Shakespeare's works...")
    response = requests.get(url)
    text = response.text
    
    # Basic preprocessing
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    
    clean_text = text[start_idx:end_idx]
    
    # Save the processed text
    with open(data_file, 'w', encoding='utf-8') as f:
        f.write(clean_text)
    
    print("Download complete and file saved locally.")
    return clean_text

if __name__ == "__main__":
    download_shakespeare() 
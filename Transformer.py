import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils.data as dl
from collections import Counter
import io
import urllib.request
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

print("Step 1: Loading WikiText-2 dataset...")

# Download WikiText-2 directly
def download_wikitext2():
    url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    filename = "wikitext-2-train.txt"
    
    # Check if file already exists
    import os
    if os.path.exists(filename):
        print(f"Using existing file: {filename}")
        return filename
    
    try:
        print("Downloading WikiText-2 training data...")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded to {filename}")
        return filename
    except Exception as e:
        print(f"Error downloading: {e}")
        return None

# Download and load data
filename = download_wikitext2()

if filename:
    with open(filename, 'r', encoding='utf-8') as f:
        text_lines = f.readlines()
    print(f"Loaded {len(text_lines)} lines from WikiText-2")
    print(f"First line preview: {text_lines[1][:100]}")
else:
    print("Using simple example data instead")
    text_lines = [
        "the cat sat on the mat",
        "the dog ran in the park", 
        "the cat played with the ball",
        "the dog sat on the chair",
        "the cat ran in the house",
        "the dog played with the toy"
    ]
    print(f"Using {len(text_lines)} example sentences")
    
def simple_tokenize(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

all_tokens = []
for line in text_lines:
    all_tokens = []
for line in text_lines:
    tokens = simple_tokenize(line)
    if tokens:  # Skip empty lines
        all_tokens.extend(tokens)
# print(all_tokens)

vocab = list(set(all_tokens))
print('len', {len(vocab)})

# Create reverse mapping: number → word
idx_to_word = {idx: word for word, idx in enumerate(vocab)}
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
def idx_to_string(idx_list):
    """Convert a list of indices to a string of words"""
    return ' '.join([idx_to_word[idx] for idx in idx_list])

def string_to_idx(text):
    """Convert a string to a list of indices"""
    tokens = simple_tokenize(text)
    return [word_to_idx[word] for word in tokens]


# Test the mapping both ways
print("\nTesting mappings:")
print(string_to_idx('the cat sat on the mat'))
print(string_to_idx('the'))

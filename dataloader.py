from tqdm import tqdm
import json

import lm_dataformat
import numpy as np
import nltk

# nltk.download('punkt')
nltk_sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

pile_mapper = { "stackexchange":"StackExchange", 
                    "wikipedia":"Wikipedia (en)", 
                    "cc":"Pile-CC", 
                    "github":"Github", 
                    "pubmed_abstracts":"PubMed Abstracts", 
                    "openwebtext2":"OpenWebText2", 
                    "freelaw":"FreeLaw", 
                    "math":"DM Mathematics", 
                    "nih":"NIH ExPorter", 
                    "uspto":"USPTO Backgrounds", 
                    "hackernews":"HackerNews", 
                    "enron":'Enron Emails',
                    "books3": 'Books3',
                    "pubmed_central": 'PubMed Central',
                    "gutenberg":'Gutenberg (PG-19)',
                    "arxiv":'ArXiv',
                    "bookcorpus2":'BookCorpus2',
                    "opensubtitles":'OpenSubtitles',
                    "youtubesubtitles":'YoutubeSubtitles',
                    "ubuntu":'Ubuntu IRC',
                    "europarl":'EuroParl',
                    "philpapers":'PhilPapers'}

def split_paragraph(paragraph, max_sentences = 10):
    sentences = nltk_sentence_tokenizer.tokenize(paragraph)
    new_paragraphs = []
    for i in range(0, len(sentences), max_sentences):
        new_para = " ".join(sentences[i:i + max_sentences])
        new_paragraphs.append(new_para)
    return new_paragraphs

def generate_pile_zst(subset, num_samples=5000, split = "val"):
    if subset.startswith("pile_"):
        subset = subset[5:]
    file_path = f"/data/the_pile/{split}.jsonl.zst"
    subset_key = pile_mapper[subset]
    texts = []
    num_docs = 0
    reader = lm_dataformat.Reader(file_path)
    for count, doc in enumerate(tqdm(reader.stream_data(get_meta=True))):
        if doc[1]['pile_set_name'] == subset_key:
            if len(doc[0].split(" ")) < 10:
                continue
            texts.append(doc[0])
            num_docs += 1
        if num_docs >= num_samples:
            break
    return texts

def generate_pile_jsonl(subset, num_samples=5000):
    if subset.startswith("pile_"):
        subset = subset[5:]
    file_path = "/data/the_pile/combined.jsonl"
    subset_key = pile_mapper[subset]
    texts = []
    num_texts = 0
    with open(file_path, 'r', encoding="utf-8") as json_file:
        for line in json_file:
            json_data = json.loads(line)
            if 'text' in json_data:
                if json_data['meta']['pile_set_name'] == subset_key:
                    if len(json_data['text'].split(" ")) < 800:
                        continue
                    texts.append(json_data['text'])
                    num_texts += 1
                    if num_texts == num_samples:
                        break
    return texts

def generate_c4(num_samples=500):
    # trove mount dataset/C4_subset@1.0.0 ./data
    file = "data/C4_subset-1.0.0/data/raw/c4-train.00000-of-01024.json"
    texts = []
    num_texts = 0
    with open(file, 'r', encoding="utf-8") as json_file:
        for line in json_file:
            json_data = json.loads(line)
            if 'text' in json_data:
                texts.append(json_data['text'])
                num_texts += 1
            if num_texts == num_samples:
                break
    return texts

def split_long_texts_by_paragraph(texts, num_samples):
    if len(texts) < num_samples:
        print(f"initial texts {len(texts)} were less than num_samples {num_samples}. Further splitting")
        #split the sentences at every 1000 characters
        required_from_each = 2*(num_samples//len(texts) + 1)
        new_texts = []
        for text in texts:
                new_texts += split_paragraph(text, max_sentences=3)[:required_from_each]

        texts = new_texts
    
    print(f"Length of texts {len(texts)}")
    return texts

def split_long_texts(texts, num_samples, seq_length, tokenizer = None):
    '''
    This function splits long texts into smaller texts of length seq_length
    1. Concatenate all the texts together
    2. Convert everything to tokens
    3. divide into chunks of seq_length
    4. Convert back to text
    5. return the list of texts
    '''
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")
    
    #concatenate all the texts
    all_text = " ".join(texts)
    #tokenize
    tokens = tokenizer.encode(all_text, return_tensors="pt")[0]
    #divide into chunks
    chunk_length = seq_length
    num_chunks = len(tokens)//chunk_length
    new_texts = []

    for i in range(num_chunks):
        chunk = tokens[i*chunk_length:(i+1)*chunk_length]
        text = tokenizer.decode(chunk)
        new_texts.append(text)
    
    # randomize the order and return only num_samples
    np.random.seed(11)
    np.random.shuffle(new_texts)
    new_texts = new_texts[:num_samples]

    return new_texts

def load_data(dataset_name, split, num_samples = 1000, seq_length = 512):
    if  "enron" in dataset_name:
        seq_length = 64
    if "nih" in dataset_name:
        seq_length = 64
    if "pubmed_abstracts" in dataset_name:
        seq_length = 32

    if split == "train":
        texts = generate_pile_jsonl(dataset_name, num_samples=num_samples*5)
        texts = split_long_texts(texts, num_samples,seq_length)
    else:
        assert split == "val"
        texts = generate_pile_zst(dataset_name, num_samples=num_samples*5)
        texts = split_long_texts(texts, num_samples,seq_length)
    print (f"Loaded {len(texts)} samples from {dataset_name} {split}")
    return texts

#!/bin/sh
cd ..

for dataset in stackexchange wikipedia cc github pubmed_abstracts openwebtext2 freelaw math nih uspto hackernews enron books3 pubmed_central gutenberg arxiv bookcorpus2 opensubtitles youtubesubtitles ubuntu europarl philpapers
do
    echo "dataset: $dataset"
    python data_creator.py --dataset_name $dataset 
done
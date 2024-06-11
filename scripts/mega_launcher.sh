
for dataset in stackexchange wikipedia cc github pubmed_abstracts openwebtext2 freelaw math nih uspto hackernews enron books3 pubmed_central gutenberg arxiv bookcorpus2 opensubtitles youtubesubtitles ubuntu europarl philpapers
do
    for outliers in "mean+p-value" "mean" "p-value" #"clip" "zero" "keep" "randomize" #
    do
        for normalize in "combined" "train" "no"
        do
            for features in "all" "selected"
            do
                for false_positive in 1 0
                do
                    for model in "EleutherAI/pythia-12b-deduped" "EleutherAI/pythia-12b" "EleutherAI/pythia-6.9b-deduped" "EleutherAI/pythia-6.9b" "EleutherAI/pythia-1.3b-deduped" "EleutherAI/pythia-1.3b" "EleutherAI/pythia-410m-deduped" "EleutherAI/pythia-410m"
                    do
                        # num_samples=500 if false_positive=1 else 1000
                        if [ $false_positive -eq 1 ]
                        then
                            num_samples=500
                        else
                            num_samples=1000
                        fi
                        sbatch launcher.sh $model $outliers $normalize $features $false_positive $num_samples $dataset 
                    done
                    wait
                done
            done
            
        done
    done
done
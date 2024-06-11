# launch all the models on the inference dataset
# "kernelmachine/silo-pdswby-1.3b",
# "roneneldan/TinyStories-33M",
# "roneneldan/TinyStories-1M",
# "microsoft/phi-1_5",
# "microsoft/phi-1",

# for model_name in "roneneldan/TinyStories-33M" "roneneldan/TinyStories-1M" #"microsoft/phi-1_5" 
# for model_name in "EleutherAI/pythia-410m-deduped" "EleutherAI/pythia-1.3b-deduped" "EleutherAI/pythia-6.9b-deduped" 
# for model_name in "kernelmachine/silo-pdswby-1.3b" #(need different git repo for this model)
num_jobs=0
for model_name in "EleutherAI/pythia-410m" 
    do
    if [ $model_name = "EleutherAI/pythia-6.9b" ]
    then
        batch_size=8
    else
        batch_size=32
    fi
    # batch_size=1 if model_name = "EleutherAI/pythia-12b"
    if [ $model_name = "EleutherAI/pythia-12b-deduped" ]
    then
        batch_size=1
    fi

    for dataset in bookcorpus2 opensubtitles youtubesubtitles ubuntu europarl philpapers pubmed_abstracts math nih  enron stackexchange wikipedia cc github openwebtext2 freelaw uspto hackernews  books3 pubmed_central gutenberg arxiv #
        do
        for split_name in "train" "val"
        do
            num_jobs=$((num_jobs+1))
            # wait if 8 jobs submitted
            if [ $num_jobs -eq 24 ]
            then
                echo "waiting for 8 jobs to complete"
                wait
                sleep 100s
                # check squeue if any process is running by user pratyus2
                while [ $(squeue -u pratyus2 | wc -l) -gt 1 ]
                do
                    echo "waiting for 8 jobs to complete"
                    sleep 10s
                done
                num_jobs=0
            fi
            sbatch di_launcher_individual.sh $model_name $split_name 0 $batch_size $dataset
        done
        # sbatch di_launcher_b.sh $model_name $split_name 0 $batch_size
        # sbatch di_launcher_c.sh $model_name $split_name 0 $batch_size
        # sbatch di_launcher_d.sh $model_name $split_name 0 $batch_size
        # sbatch di_launcher_e.sh $model_name $split_name 0 $batch_size
        # sbatch di_launcher_f.sh $model_name $split_name 0 $batch_size
    done
done
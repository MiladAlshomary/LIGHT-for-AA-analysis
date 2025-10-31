
### Training-Environment
```
conda create -n wg-env python==3.12 -y
conda activate wg-env
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install numpy scikit-learn tqdm transformers memory_profiler sentence_transformers convokit
```

OR, if you want to not use latest versions and use our frozen setup:

```
conda create -n wg-env python==3.12 -y
conda activate wg-env
pip install -r requirements.txt
```

### Experiment
This hypothesis is also tested on the Wegmann baseline, where we create an extension of Wegmann, which computes partial losses from all its intermediate layers, rather than relying on only the final layer’s output. By leveraging hidden representations across the network’s depth, it captures both low-level and high-level stylistic cues. This multi-layer approach yields more robust, comprehensive authorship attribution signals than the single-layer Wegmann model.


Data: 
- Reddit: For training, we take triplet-sets comprising (author_1-text1, author_1-text2, author_2-text1) serving as (anchor, same, different) for Triplet-loss-objective for the Reddit-MUD dataset.
- Wegmann: We take their default conversation data used for their training, and load a cleaned out version (their version has newlines in single line tsv rows, which breaks default tsv format). By cleaning, we mean using escape characters to replace \n with \\\n, and similar other whitespace which otherwise would break TSV loading - their default Wegmann model's data format


### Training and Eval

`./train_evaluate.sh`

### Generate new Reddit data
The key is `--generate-data`, the others are just required parameters, but once the data is generated, the code exits without training. This is intentional behavior to allow data to not get overwritten during new model training. So, when data gets generated models don't get trained, and we keep same data for all model training using Reddit data for fair metric comparisons.

`time python main.py --backbone modernbert --data reddit --mode single --generate-data`

## TGFormer: Towards Temporal Graph Transformer with Auto-Correlation Mechanism

### Requirements

Dependencies (with python >= 3.7):

```{bash}
pandas==1.1.0
torch==1.6.0
scikit_learn==0.23.1
```

### Dataset and Preprocessing

#### Download the public data
Download the sample datasets (eg. wikipedia and reddit) from
[here](http://snap.stanford.edu/jodie/) and store their csv files in a folder named
```data/```.

#### Preprocess the data
We use the dense `npy` format to save the features in binary format. If edge features or nodes 
features are absent, they will be replaced by a vector of zeros. 
```{bash}
python utils/preprocess_data.py --data wikipedia --bipartite
python utils/preprocess_data.py --data reddit --bipartite
```

## Model Training
```shell
python main.py

```

## Periodic Dependencies Learning experiment
<p align="center">
<img src=".\pdf\Periodic_Dependencies.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 1.</b> Visualization of learned periodic dependencies. For clearness, we select the top-9 time delay sizes $\left\{\delta_1,\cdots,\delta_9, \right\}$ of Auto-Correlation and mark them in raw series (red lines). For self-attentions, top-9 similar points with respect to the last time step (red stars) are also marked by yellow lines.
</p>

## Long-term Dependencies Learning experiment
<p align="center">
<img src=".\pdf\Long-term_Dependencies.jpg" height = "250" alt="" align=center />
<br><br>
<b>Figure 2.</b> Performance of different methods on Reddit and LastFM with varying historical lengths $L$.
</p>

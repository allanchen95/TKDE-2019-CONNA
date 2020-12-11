# CONNA

This is tensorflow implementations of CONNA in the algorithm-level.





### Requirements

>- Ubuntu 16.04
>- Python 3.6.8
>- Tensorflow-gpu == '1.11.0'
>- A single NVIDIA Tesla V100





### Data Preprocessing

Before running the code, you need to  preprocess the raw data as follows, take OAG-WhoIsWho for example:

>+ **Prepare the essential data**: Download data from the link of OAG-WhoIsWho into the folder of *OAG_WhoIsWho_data*.
>
>
>
>+ **Generate word embedding**:  Run the *preprocessing.py* in the folder of *generate word2vec* step by step to obtain the word embeddings in each paper (Notbly, we have already sampled some training and testing paper data from OAG-WhoIsWho, more specifically, the version of na-v1,  and save it in the folder of *OAG_WhoIsWho_data*, so you must download the data files, from na-v1, that contained the paper information of our sampled data from OAG-WhoIsWho.);
>





## Running

>+ **Train the Ranking Module**: Run the *ranking_main.py* to train the ranking module of CONNA and save the checkpoints in the folder of *saved_ranking_model* ; 
>
> 
>
>+ **Train the Classificaion Module**: Run the *classifier_main.py* to load the ranking module from *saved_ranking_model*, and further reinforce both the two modules with their feedback. Specifically,  we first implement it with REINFORCE algorithm, and find its performance is the same as retraining the right cases, i.e., the accurately predicted cases, so for simplicity, we just retrain the right cases.







## Data Source

All experimental data is available.

OAG-WhoIsWho: (link: https://www.aminer.cn/whoiswho)

KDD-Cup: (link:https://pan.baidu.com/s/10RV3Xrn12t9TRZz2yc0Gyw  passwd: w8yv)

Note: The results might be slightly inconsistent induced by the different hyper-parameters or data samplings.


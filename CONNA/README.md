# CONNA

This is tensorflow implementations of CONNA in the algorithm-level.



**Note**: For more and cleaner name disambiguation data, we highly recommend *WhoIsWho*, **the world largest manually-labeled name disambiguation benchmark**, which contains nearly **1,000,000** papers and can be easily accessed in its dataset page, 

*WhoIsWho*: (link: https://www.aminer.cn/whoiswho)



### Requirements

>- Ubuntu 16.04
>- Python 3.6.8
>- Tensorflow-gpu == '1.11.0'
>- A single NVIDIA Tesla V100



### Data Preprocessing

Before running the code, you need to  preprocess the raw data as follows, take *WhoIsWho* for example:

>+ **Prepare the essential data**: Download data from the link of *WhoIsWho* into the folder of *WhoIsWho_data*, (Notably, we have already sampled some training and testing paper data from *WhoIsWho*, more specifically, the version of na-v1, and save it in the folder of *WhoIsWho_data*.) 
>
>   + *train/test_author_pub_index_profile.json*, Serves as the **existing author profiles**, which is organized as a three-level dictionary, i.e., name --- author_id --- papers,
>
>   + *train/test_author_pub_index_test.json* serves as **the ground truth of the papers that need to be assigned to the author profiles**.
>
>    (Note: the paper-id is composed of original paper-id and the index of name to be disambiguated, like the paper id "3UDRYR4J-2", "3UDRYR4J" is the original paper id which can be found in the conna_pub_dict, and "2" means the 3rd author need to be disambiguated. More data construction details can be found in the page of *WhoIsWho*.) 
>
>
>
>+ **Generate word embedding**:  Download the essential paper attribute file [conna_pub_dict.json](https://lfs.aminer.cn/misc/ND-data/conna_pub_dict.json), then run the *preprocessing.py* in the folder of *generate word2vec* step by step to obtain the word embeddings in each paper. You will get following files after this step,
>     + *pub_feature.ids*: Key-value dict., where the key is paper id, value is corresponding paper attribute token id,
>     + *author_emb.array*: Contains author embeddings with corresponding author token id,
>     + *word_emb.array*: Contains word embeddings with corresponding word token id.



## Running

>+ **Train the Ranking Module**: Run the *ranking_main.py* to train the ranking module of CONNA and save the checkpoints in the folder of *saved_ranking_model* ; 
>
> 
>
>+ **Train the Classificaion Module**: Run the *classifier_main.py* to load the ranking module from *saved_ranking_model*, and further reinforce both the two modules with their feedback. Specifically,  we first implement it with REINFORCE algorithm, and find its performance is the same as retraining the right cases, i.e., the accurately predicted cases, so for simplicity, we just retrain the right cases.



Note: The results might be slightly inconsistent induced by the different hyper-parameters or data samplings.



### Checkpoints

You may omit training process by downloading corresponding model checkpoints.

+ **Ranking Model**: Download the ranking model checkpoints into *saved_ranking_model*, then you can validate the ranking performance (Link: https://pan.baidu.com/s/1_QBpTX41_6xOX-AOQ881Ag   passwd: 1dg1) , based on which you can further train the classification model.
+ **Classification Model**:



## Data Source

KDD-Cup: (link:https://pan.baidu.com/s/10RV3Xrn12t9TRZz2yc0Gyw  passwd: w8yv)






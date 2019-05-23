# topic_reranking
This repository contains the source code used for the paper 'Re-Ranking Words to Improve Interpretability of Automatically Generated Topics' which can be found at https://arxiv.org/abs/1903.12542


# Steps:
1. Load and Preprocess data.
2. Train LDA model on the processed data.
3. Display the learned topics with their default order.
4. Rerank topic words using one of the ranking methods.
5. Display the re-ranked topics.




# Files
* **data.txt**: contains the sample data extracted from [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/) which is just used to run the code. __Note__: this is not the dataset used in the paper as those datasets are not available for public distribute.  
* **reranking_main.py** : conatins loading the data, preprocessing the data, learning topics for the data using Gensim and finally re-ranking the topics terms to improve thier interpretability.
* **methods.py**: conatins all the methods used in reranking_main.py.

# Running the code
* Clone or Download this project
* Set up the parameters in reranking_main.py
* Excute reranking_main.py 


# Output 
* The code will display:
    * The learned latent topics with their words ranked using the original method "Rank_orig". 
    * The topics with thier words re-ranked using one of the methods described in the paper.
    * The average coherence of the topics is also shown. 
    




# topic_reranking
This repository contains the source code used for the paper:<br>
<br>
**Re-Ranking Words to Improve Interpretability of Automatically Generated Topics** (2019) Areej Alokaili, Nikolaos Aletras and Mark Stevenson in *Proceedings of the 13th International Conference on Computational Semantics - Long Papers*, pp43-54, Gothenburg, Sweden.<br>
<br>
[https://www.aclweb.org/anthology/W19-0404](https://www.aclweb.org/anthology/W19-0404)
 




# Steps:
The code executes the following steps: 

1. Load and preprocess data.
2. Train an LDA model on the processed data.
3. Display the topics learned with their default order.
4. Rerank topic words using one of the ranking methods.
5. Display the re-ranked topics.




# Files
* **data.txt**: contains sample input data extracted from [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/). __Note__: this is not the dataset used in the paper as those datasets are not available for public distribution.  
* **reranking_main.py** : loads and preprocesses the data, learns topics using Gensim and finally re-ranking the topics terms to improve their interpretability.
* **methods.py**: contains all methods used by reranking_main.py.

# Running the code
* Clone or Download this project
* Set up the parameters in reranking_main.py
* Excute reranking_main.py 


# Output 
* The code will display:
    * The learned latent topics with their words ranked using the original method "Rank_orig". 
    * The topics with thier words re-ranked using one of the methods described in the paper.
    * The average coherence of the topics is also shown. 
    




# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:18:48 2019

@author: areej
"""
import re
from decimal import Decimal
from nltk.corpus import stopwords
import numpy as np
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer

import math
########################## data prep methods ##########################
def load_data(data_path):
    raw_texts=[]
    with open('data.txt','r') as f:
        for line in f:
            raw_texts.append(line)
    return raw_texts

def clean_data(raw_texts):
    #lower case all words
    pro_texts = [text.lower() for text in raw_texts]
    
    # remove special characters and numbers
    pro_texts = [re.sub(r"[^a-zA-Z]+", ' ', text) for text in pro_texts]
    
    # remove stopwords
    stopwords_list = set(stopwords.words('english')) 
    
    pro_texts=[[x for x in text.split(' ') if x not in stopwords_list] for text in pro_texts]
    
    
    # remove short words < 2 characters
    pro_texts=[[x for x in text if len(x) > 2] for text in pro_texts]
    
    
    
    return pro_texts

def get_dict(pro_texts):
    # create a dictionary 
    dictionary = Dictionary(pro_texts)
    
    # filter out words with frequency< 5 in a document and filter out words apprear in more than %30 of the documents
    dictionary.filter_extremes(no_below=5, no_above=0.3, keep_n=None)
    #remove gaps in ids after the filter
    dictionary.compactify()
    
    #create bow of the data
    corpus = [dictionary.doc2bow(text) for text in pro_texts]
    
    return dictionary, corpus


########################## ranking support methods ##########################
def flip_matrix(matrix, dictionary, NUMTOPICS):
    try:
        term_topic_matrix= np.empty(shape=(len(dictionary),NUMTOPICS))
        column=0
        for line in matrix:
            row=0    
            for term_prob in line:
                #print(term_prob)
                term_topic_matrix[row,column]=term_prob 
                row+=1
            column+=1
    except:
        print("error occured ")
                

    return term_topic_matrix

def flip_to_topic_terms(term_topic_matrix, NUMTOPICS,dictionary):
    
    topic_term_matrix= np.empty(shape=(NUMTOPICS,len(dictionary)))
    i=0
    for line in term_topic_matrix:
        topic_num=0    
        for topic_prob in line:
            topic_term_matrix[topic_num,i]=topic_prob 
            topic_num+=1
        i+=1
                

    return topic_term_matrix
    
def calculate_product(line, NUMTOPICS):
    data_float= [float(i) for i in line]
    
    
    product=Decimal(data_float[0])
    x=1
    for num in data_float[1:]:
        product*=Decimal(num)
        
        x+=1
    
    return (pow(product, Decimal(1/NUMTOPICS)))

def calculate_sum(line):
    
    data_float= [float(i) for i in line]
    if (sum(data_float) == 0):
        print(line)
        sys.exit(0)
    return sum(data_float)
# recevies:
#      a probability matrix with rows coresposeds to topics and columns coresponds for disctionary terms.
#      dimensions (NUMTOPICS x dictionaryLength)
#      Example of one row: [0.00618062 0.00028241 0.00030084 ... 0.         0.         0.        ]
# returns:
#      Matrix of tuples (word_index, probability) of words with highest probability.
#      dimensions  (NUMTOPICS x NUMTERMS)
#      Example of one row: [(1080, 0.068920371331339), (1450, 0.06586989934376174), (442, 0.05922795804750381), ....]
def create_new_topics(topic_term_array,NUMTOPICS, NUMTERMS):
    
    
    topics=[[0 for x in range(NUMTERMS)] for y in range(NUMTOPICS)]
    for topic in range (0, NUMTOPICS):
        row= topic_term_array[topic]
        ind = np.argpartition(row,-NUMTERMS)[-NUMTERMS:]
        #print("index are ", ind , "values are ", row[ind])
        sorted_ind=ind[np.argsort(row[ind])[::-1]]
        

        #print("index sorted are ", sorted_ind , "values are ", row[sorted_ind])
        for terms in range (0, NUMTERMS):
            topics[topic][terms]=((sorted_ind[terms], row[sorted_ind[terms]]))
    
    return topics
###########################################
# recevies:
#      matrix of tuples (word_index, probability)
#      with dimensions  (NUMTOPICS x NUMTERMS )
#      Example of one row: [(1080, 0.068920371331339), (1450, 0.06586989934376174), (442, 0.05922795804750381),...]

# returns:
#      matrix of tuples (word, probability)
#      with dimensions  (NUMTOPICS x NUMTERMS )
#      Example of one row: [('london', 0.06892), ('fast', 0.06587), ('events', 0.059228), ...]
###########################################
def get_all_topics_reweighted_with_matrix(topic_term_array,  NUMTOPICS, NUMTERMS, dictionary):
    topics=[]
    for topic_num in range(0, NUMTOPICS):
            
            list1= get_list_in_term_prob_format(topic_term_array[topic_num][:NUMTERMS], NUMTERMS, dictionary) 

            topics.append(list1)
    return topics       
            
            
def get_list_in_term_prob_format(topics_terms, num_terms, dictionary):
    
    temp = [0 for x in range(num_terms)] 
    i=0
    for (term_id, prob) in topics_terms:
        temp[i]= dictionary[int(term_id)], float("{:10f}".format(float(prob)))
            
        i+=1
    return temp


########################## ranking methods ##########################
def rank_orig (NUMTOPICS, NUMTERMS,ldamodel, dictionary):
    topics=[]
    for topic_num in range(NUMTOPICS):
        topic = ldamodel.show_topic(topic_num, topn=NUMTERMS)
        topics.append(topic)
    return topics

def rank_norm (topics_norm, NUMTOPICS, NUMTERMS,ldamodel, dictionary):
    print("************** Start Rank_norm re-weighting **************")
    
    # flip the topic_terms matrix to get term_topics_matrix
    term_topics_matrix = flip_matrix(topics_norm, dictionary, NUMTOPICS)
    print("flip done ", term_topics_matrix.shape)

    new_topics=[]
    
           
    term_ind = 0 
   
    for line in term_topics_matrix:
        new_line=[]
        sum_probabilities=calculate_sum(line)
        #print(str(term_ind),": Sum ", sum_probabilities)
        for term in line:
            operand1=float(term)
            
            new_weight=operand1/sum_probabilities
            
             # print("divide %s by %s = %s " %(operand1, sum_probabilities, new_weight))
                
            new_line.append(new_weight)

        term_ind=term_ind+1
        new_topics.append(new_line)
        
        
    new_topics=flip_to_topic_terms(new_topics, NUMTOPICS, dictionary)
    
    return new_topics
            
    

def rank_tfidf(topics_norm, NUMTOPICS, NUMTERMS,ldamodel, dictionary):
    print("************** Start Rank_tfidf re-weighting **************")

    
    # flip the topic_terms matrix to get term_topics_matrix
    term_topics_matrix = flip_matrix(topics_norm, dictionary, NUMTOPICS)
    #print("flip done ", term_topics_matrix.shape)
    new_topics=[]
    
    
    terms = 0 
    for line in term_topics_matrix:
        new_line=[]
        
        # term's probability acress all topics
        #print("line ", line)
        prod=calculate_product(line, NUMTOPICS)
        #print('term ', terms , "line ", line)
        if prod == 0:
            print('term ', terms , "line ", line)
            sys.exit(0)
        #print("prod ", prod)
        for term in line:
            operand1=Decimal(term)
            divide=operand1/prod
            operand2=Decimal(math.log10(divide))
            new_weight=operand1*operand2
            new_line.append(new_weight)
        
        
        terms=terms+1
        #print("new_line ", new_line)
        new_topics.append(new_line)
        

    #flip matrix back to topic_terms
    new_topics = flip_to_topic_terms(new_topics, NUMTOPICS, dictionary)
    
    
    return new_topics
def compute_idf(dictionary, pro_texts):
    pro_texts= [" ".join(text) for text in pro_texts]
    tf = TfidfVectorizer(use_idf=True, lowercase=True, vocabulary=dictionary.values())
    #tfidf_matrix =  tf.fit_transform([content for file, content in texts])
    tf.fit_transform(pro_texts)
    idf= tf.idf_
    dictionary_idf = dict(zip(tf.get_feature_names(), idf))
   
    return dictionary_idf

def rank_idf(topics_norm, NUMTOPICS, NUMTERMS,ldamodel, dictionary, pro_texts):
    
    print("************** Start Rank_idf re-weighting **************")
    
    term_topics_matrix = flip_matrix(topics_norm, dictionary, NUMTOPICS)
    
    
    
    new_topics=[]
    word2idf_dict= compute_idf(dictionary, pro_texts)
           
    term_ind = 0 
   
    for line in term_topics_matrix:
        new_line=[]
        #print("term_topics row with "+str(len(line))+" topics -> ", line)
        for termid in line:
            operand1=float(termid)
            termvalue= dictionary[term_ind]            
            operand2= word2idf_dict[termvalue] 
            new_weight=operand1*operand2
            new_line.append(new_weight)
        
        term_ind=term_ind+1
           
        
        new_topics.append(new_line)
       
    
    new_topics = flip_to_topic_terms(new_topics, NUMTOPICS, dictionary)
    
    return new_topics

    
    
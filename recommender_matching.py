import numpy as np
import torch
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
tqdm.pandas()
import time
import numexpr as ne
from sentence_transformers import SentenceTransformer
import streamlit as st
import pickle


class Recommender_matching_segel():
    """
    This class will perform end to end recommendations.
    it hold preprocessing:
      - cluster the information of the Publications into one long text in a form of: ['title']+ '[SEP]'+ ['description'] + '[SEP]'+ ['authkeywords']
      - cluster the information of the Grants into one long text in a form of: ['title']  + '[SEP]'+ ['tags'] + '[SEP]'+ ['keywords']
    Fit:
      -Embedding the publications into a torch.Tensor column inplace of pubs_df
      -Embedding the grants into a torch.Tensor column inplace of grants df

    Predict - diffrent kind of predictions, asked by user:
      - predict for each researcher , top n recommendations
      - creates a dataframe of top general recommendations, considering a given threshold

    Save* - saving features for all tables

    """

    def __init__(self, model, device, senior_df=False, pubs_df=None):
        self.DIR = 'K:/Shared drives/RPI-Dan Peled HIPAA/Seder/Funds_recommendations/all/'
        self.model = model  # pre-trained model as an input
        self.senior_df = pd.read_csv("segel_thin.csv")
        self.fit_action = False if pubs_df is None else True
        self.device = device  # CUDA or cpu
        self.fit_time = 0
        self.predict_time = 0
        with open('segel_bahir_embd.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        f.close()
        with open('segel_bahir_wrn.pkl', 'rb') as f:
            self.wrn = pickle.load(f)
        f.close()
        self.pubs_df = pd.DataFrame(loaded_dict.items(), columns=['Name', 'embedded'])
        self.pubs_df['Name'] = self.pubs_df['Name'].str.lower()

    def predict(self, text, Th=80, Number=0):
        t = time.time()
        self.recommended = pd.DataFrame(
            columns=['Text_Matched', 'Researcher', 'match score', 'faculties', 'circle', 'mail'])
        self.text_embd = self.embed_text(text)
        for name in tqdm(self.pubs_df.Name.unique()):
            rel_df = self.pubs_df[self.pubs_df.Name == name].reset_index(drop=True)
            vals = self.senior_df[self.senior_df.ScopusName == name].values[0]
            author_embeddings = rel_df.embedded.values[0]
            sim = self.cosine_sim(author_embeddings, self.text_embd)
            if Number > 0:
                self.recommended.loc[-1] = [text[0:10] + '...', name, sim, vals[1], vals[2], vals[3]];
                self.recommended.reset_index(drop=True, inplace=True)
            elif sim * 100 >= Th:  # add to general recommendation table if score is higher than decided threshold (default is 90% score)
                self.recommended.loc[-1] = [text[0:10] + '...', name, sim, vals[1], vals[2], vals[3]];
                self.recommended.reset_index(drop=True, inplace=True)
        if Number > 0:
            self.recommended = self.recommended.loc[self.recommended['match score'].nlargest(Number).index]
        self.recommended['match score'] = (
            ((self.recommended['match score'] * 100).apply(np.round)).apply(lambda x: int(x))).apply(
            lambda x: str(x) + '%')
        self.recommended = self.recommended.sort_values(by='match score', ascending=False)
        self.predict_time = (time.time() - t) / 60  # record mintues, not seconds
        print(f'Time taken to Predict:{self.predict_time:.0f} minutes')
        return self.recommended.reset_index(drop=True)

    def embed_text(self, x):
        '''
        Embbed the self stacked grants relevant text into a numpy array
        '''
        return self.model.encode(x, convert_to_tensor=True, device=self.device).cpu().detach().numpy().reshape(1, -1)

    def cosine_sim(self, array1, array2):
        sumyy = np.einsum('ij,ij->i', array2, array2)
        sumxx = np.einsum('ij,ij->i', array1, array1)[:, None]
        sumxy = array1.dot(array2.T)
        sqrt_sumxx = ne.evaluate('sqrt(sumxx)')
        sqrt_sumyy = ne.evaluate('sqrt(sumyy)')
        return ne.evaluate('(sumxy/sqrt_sumxx)/sqrt_sumyy')[0][0]


class Recommender_matching():
    """
    This class will perform end to end recommendations.
    it hold preprocessing:
      - cluster the information of the Publications into one long text in a form of: ['title']+ '[SEP]'+ ['description'] + '[SEP]'+ ['authkeywords']
      - cluster the information of the Grants into one long text in a form of: ['title']  + '[SEP]'+ ['tags'] + '[SEP]'+ ['keywords']
    Fit:
      -Embedding the publications into a torch.Tensor column inplace of pubs_df
      -Embedding the grants into a torch.Tensor column inplace of grants df

    Predict - diffrent kind of predictions, asked by user:
      - predict for each researcher , top n recommendations
      - creates a dataframe of top general recommendations, considering a given threshold
      
    Save* - saving features for all tables 

    """
    def __init__(self,model,device,senior_df =False,pubs_df=None):
        self.DIR = 'K:/Shared drives/RPI-Dan Peled HIPAA/Seder/Funds_recommendations/all/'
        self.model=model #pre-trained model as an input
        self.fit_action = False if pubs_df is None else True
        if not self.fit_action:
            # self.pubs_df= pd.read_csv("K:\Shared drives\RPI-Dan Peled HIPAA\Seder\horizon_recommendations\pubs_df_fit.csv")
            self.pubs_df = pd.read_csv("pubs_df_fit.csv")
        else:
            self.pubs_df=pubs_df.copy() #publications dataframe
        self.senior_df=senior_df
        if senior_df :
            # senior_df = pd.read_excel("K:/Shared drives/RPI-Dan Peled HIPAA/Seder/Funds_recommendations/full_segel_list_for_rec.xlsx")
            senior_df = pd.read_excel("full_segel_list_for_rec.xlsx")
            self.pubs_df = pd.merge(left=self.pubs_df,right=senior_df[['ScopusName','circle']],how='outer',left_on=['Name'],right_on=['ScopusName']).dropna(subset=['description']).drop_duplicates(subset=['title', 'coverDate', 'publicationName', 'description', 'authkeywords', 'scopus_id', 'Name']).drop(labels=['ScopusName_y','ScopusName_x'],axis = 1)
#             self.pubs_df = pd.merge(left=self.pubs_df,right=senior_df[['ScopusName','faculties','circle']],how='outer',left_on=['Name'],right_on=['ScopusName']).dropna(subset=['description']).drop_duplicates(subset=['title', 'coverDate', 'publicationName', 'description', 'authkeywords', 'scopus_id', 'Name'])
            self.pubs_df['circle']= self.pubs_df['circle'].fillna('')
            self.pubs_df= self.pubs_df[self.pubs_df['Name'].isin(senior_df['ScopusName'])].copy() #only senior staff publications
            self.DIR = 'K:/Shared drives/RPI-Dan Peled HIPAA/Seder/Funds_recommendations/seniors/'

        self.device = device #CUDA or cpu 
        self.fit_time = 0
        self.predict_time = 0

        #apply pre-processing on data before fit-predict
        self.preprocess()
        self.fit()
    def preprocess(self):
#         self.grants['unified'] = self.grants.apply(self.unify_text_grants,axis = 1)
        if self.fit_action:
            self.pubs_df['unified'] = self.pubs_df.apply(self.unify_text_pubs,axis = 1)
        else:
            self.pubs_df['embedded']=self.pubs_df['embedded'].progress_apply(lambda x: np.array(literal_eval(x)).reshape(1,-1))
        self.pubs_df=self.pubs_df.dropna(subset=['Name']).reset_index(drop=True)

    def fit(self):
        t = time.time()
        if self.fit_action:
            self.pubs_df['embedded'] = self.pubs_df['unified'].progress_apply(self.embed_text)
        self.fit_time =(time.time() -t)/60  #record mintues, not seconds
        print(f'Time taken to Fit:{self.fit_time:.0f} minutes')

    def predict(self,text,Th = 80,Number =0):
        t = time.time()
        self.recommended = pd.DataFrame(columns = ['Text_Matched','Researcher','match score','faculties','circle','warning'])
        self.text_embd = self.embed_text(text)
        for name in tqdm(self.pubs_df.Name.unique()):
          rel_df = self.pubs_df[self.pubs_df.Name == name].reset_index(drop =True)
          warn = np.nan if self.pubs_df[self.pubs_df.Name == name].shape[0] >4 else 'Fit based on less than 5 papers'
          warn_all = '' if self.pubs_df[self.pubs_df.Name == name].shape[0] >4 else 'Fit based on less than 5 papers'
          author_embeddings = np.expand_dims(rel_df['embedded'].values, axis=1).mean()
          sim = self.cosine_sim(author_embeddings,self.text_embd)
          if Number >0:
           self.recommended.loc[-1] = [text[0:10]+'...',name,sim,rel_df['faculties'].iloc[0],rel_df['circle'].iloc[0],warn_all] ; self.recommended.reset_index(drop = True,inplace = True)      
          elif sim*100 >= Th: #add to general recommendation table if score is higher than decided threshold (default is 90% score)
           self.recommended.loc[-1] = [text[0:10]+'...',name,sim,rel_df['faculties'].iloc[0],rel_df['circle'].iloc[0],warn_all] ; self.recommended.reset_index(drop = True,inplace = True)
        if Number >0:
         self.recommended= self.recommended.loc[self.recommended['match score'].nlargest(Number).index]
        self.recommended['match score'] =  (((self.recommended['match score']*100).apply(np.round)).apply(lambda x:int(x))).apply(lambda x: str(x)+'%')
        self.recommended= self.recommended.sort_values(by='match score',ascending=False)
        self.predict_time =(time.time() -t)/60  #record mintues, not seconds
        print(f'Time taken to Predict:{self.predict_time:.0f} minutes')
        return self.recommended.reset_index(drop=True)

    def unify_text_pubs(self,x):
        '''
        Creates a columns with all publications relevant text stacked into one line
        '''
        return x['title']  + '[SEP]'+ x['description'] + '[SEP]'+ x['authkeywords']

    def embed_text(self,x):
        '''
        Embbed the self stacked grants relevant text into a numpy array 
        '''
        return self.model.encode(x, convert_to_tensor=True,device= self.device).cpu().detach().numpy().reshape(1,-1)

    def save_best_recommendations(self,topic = ''):
        self.recommended['match score'] =  (((self.recommended['match score']*100).apply(np.round)).apply(lambda x:int(x))).apply(lambda x: str(x)+'%')
        self.recommended= self.recommended.sort_values(by='match score',ascending=False)
        path = self.DIR+f'Best_Recommendations_For_{topic}.xlsx'
        self.recommended.to_excel(path,index = False)  
        
    def cosine_sim(self,array1, array2):
        sumyy = np.einsum('ij,ij->i',array2,array2)
        sumxx = np.einsum('ij,ij->i',array1,array1)[:,None]
        sumxy = array1.dot(array2.T)
        sqrt_sumxx = ne.evaluate('sqrt(sumxx)')
        sqrt_sumyy = ne.evaluate('sqrt(sumyy)')
        return ne.evaluate('(sumxy/sqrt_sumxx)/sqrt_sumyy')[0][0]



if 'rec' not in st.session_state:
    st.session_state['rec']= Recommender_matching_segel(SentenceTransformer('allenai-specter'),torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
st.title('Matching search engine based on publications')
st.subheader('Developed by Research Authority - University of Haifa')
title = st.text_input('Enter a Piece of text to find a match')
threshold = st.text_input('Enter a Threshold for matching scores')
st.write(f'Presenting only researchers with more than {threshold}% match:')
st.dataframe(data=st.session_state['rec'].predict(title, Th=int(threshold)), width=None, height=None)
#



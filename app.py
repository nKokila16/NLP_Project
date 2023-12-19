import pandas as pd
import streamlit as st
import spacy
import re
import PyPDF2
from PyPDF2 import PdfReader
from spacy import displacy
from preprocess import *
import os
from wordcloud import WordCloud
import matplotlib.pyplot as  plt
from textblob import TextBlob
import sqlite3

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
custom_nlp=spacy.load("en_ner_bionlp13cg_md")
nlp=spacy.load("en_ner_bc5cdr_md")
s_nlp=spacy.load("en_core_web_sm")


def main():
     st.header("Named Entity Recognition")
     st.sidebar.header("Choose your file")
     task=st.sidebar.file_uploader("Enter the file:")
     reader = PyPDF2.PdfReader(task)
     text = ' '
     for page in reader.pages:
          text+=page.extract_text()+ "\n"
     extra_words = re.sub("[^A-Za-z" "]+", " ",text).lower()
     docx=nlp(extra_words)
     Dataframe=[(ent.text,ent.label_) for ent in docx.ents]
     doc=custom_nlp(extra_words)
     df1=[(ent.text,ent.label_) for ent in docx.ents]
     uniue_char = []
     for c in Dataframe:
           if not c in uniue_char:
                 uniue_char.append(c)
     unique = []
     for c in df1:
           if not c in unique:
                 unique.append(c)

     st.sidebar.subheader("Search for word")
     st.markdown("Search for the word in PDF and insert into Dataframe for download and feature reference")


     final=pd.DataFrame(uniue_char,columns=["name","entity_type"])
     conn=sqlite3.connect('mydatabase.db')
     c=conn.cursor()
     c.execute('CREATE TABLE IF NOT EXISTS nlp(name text,entity_type text)')
     conn.commit()
     final.to_sql('nlp',conn,if_exists='replace',index=False)
     def view_all_data():
          c.execute('SELECT * FROM nlp')
          data = c.fetchall()
          return data
     task=st.sidebar.text_input("Enter Disease/Chemical name:")
     Entity_type=st.sidebar.text_input("Enter entity type:")
     c.execute('INSERT INTO nlp(name,entity_type) VALUES(?,?)',(task,Entity_type))
     conn.commit()
     result = view_all_data()
     clean_df = pd.DataFrame(result,columns=["Names","Entity_type"])
     with st.expander("View Database:"):
           st.table(clean_df)
     with st.expander("View Custom Updated Enitites"):
          final1=pd.DataFrame(unique,columns=["Names","Entity_type"])
          st.table(final1)

     @st.cache
     def convert_df_to_csv(df):
           return df.to_csv().encode('utf-8')
     st.download_button(
          label="Download data as CSV",
          data=convert_df_to_csv(clean_df),
          file_name='New_dataa.csv',
          mime='text/csv',)
     
     st.subheader("Sentiment analysis")
     blob = TextBlob(extra_words)
     with st.expander("Show Sentiment Analysis:"):
          st.write("Polarity score:",blob.sentiment_assessments.polarity)
          st.markdown("Polarity is a float value within the range [-1.0 to 1.0] where 0 indicates neutral, +1 indicates a very positive sentiment and -1 represents a very negative sentiment.")
          st.write("Subjectivity score:",blob.sentiment_assessments.subjectivity)
          st.markdown("Subjectivity is a float value within the range [0.0 to 1.0] where 0.0 is very objective and 1.0 is very subjective. Subjective sentence expresses some personal feelings, views, beliefs, opinions, allegations, desires, beliefs, suspicions, and speculations where as Objective sentences are factual.")
          st.write(blob.sentiment_assessments.assessments)

     st.subheader("Word Cloud")
     st.set_option('deprecation.showPyplotGlobalUse', False)
     if extra_words:
          w=WordCloud().generate(extra_words)
          plt.imshow(w)
          plt.axis('off')
          with st.expander("Show Word Cloud"):
               st.pyplot()
     st.subheader("Entity Visualizer")
     doc=nlp(extra_words)
     colors = { "DISEASE": "pink",
          "CHEMICAL": "orange",
          "CHEMICALS": "lightblue",
          }
     options = {"ents": ["DISEASE",
          "CHEMICAL",
          "CHEMICALS",
          ],
          "colors": colors
          }
     HTML=displacy.render(doc, style='ent',options=options)
     with st.expander("Show Entity Visualisation"):
          st.write(HTML,unsafe_allow_html=True)

     


     
     
          
              
if __name__ == '__main__':
	main()

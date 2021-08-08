from youtube_transcript_api import YouTubeTranscriptApi
import re
import string
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from spacy import displacy
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

import streamlit as st
import spacy_streamlit
spacy_model = "en_core_web_sm" 
import spacy
nlp = spacy.load(spacy_model)

from PIL import Image

###### UTILS ######

STOPWORDS.add('')
wordnet_lemmatizer = WordNetLemmatizer()

def find_words_time_range(start,end,df):
    # INPUTS:
    # start: minutos de inicio do range desejado
    # end: minutos de fim do range desejado
    # df: dicionário com lista de palavras em cada frame
    # OUTPUTS:
    # lista de palavras no range solicitado
    
    start = start
    end = end
    return df[(df['start']>=start)&(df['end']<=end)]['text'].sum()

@st.cache
def load_image(img):
	im = Image.open(img)
	return im

def main():
	st.title('Desafio 1STi')

	menu = ['Home','Análises Exploratórias 1','Análises Exploratórias 2','NER']
	choice = st.sidebar.selectbox('Menu',menu,)

	if choice == 'Home':
		st.markdown('# O desafio')
		st.markdown('O desafio de Data Science da 1STI teve os seguintes requisitos:')
		st.markdown('1. Obter a transcrição de um vídeo do youtube')
		st.markdown('2. Obter insights a partir de análise exploratória')
		st.markdown('3. Aplicar um modelo de reconhecimento de entidades pré-treinado utilizando a transcrição como input')
		st.markdown('4. [EXTRA] Prototipar a solução utilizando a biblioteca streamlit')

		st.markdown('# Objetivo')
		st.markdown('Meu objetivo durante o desafio foi entender e estudar as ferramentas e bibliotecas utilizadas em NLP,\
		 criar visualizações e análises que fizessem sentido para a geração de insights e expor via interface gráfica de maneira a criar uma ferramenta \
		 facil de utilizar')

		st.markdown('# Principais dificuldades')
		st.markdown('Os principais desafios foram:')
		st.markdown('1.  Encontrar vídeos que fizessem sentido aplicar um modelo NER, pois percebi que pelo objetivo do modelo alguns vídeos não gerariam insights relevantes')
		st.markdown('2. Aprender detalhes de funcionamento da biblioteca Streamlit e montar uma interface que fizesse sentido com o meu objetivo')

	elif choice == 'Análises Exploratórias 1':
		st.subheader('Análises de palavras em todo o vídeo')
		raw_docx = st.text_area('Insira a parte final da URL do vídeo a ser analisado','fC9da6eqaqg')
		if st.button("Enter"):
			srt = YouTubeTranscriptApi.get_transcript(raw_docx)
			### Pre processing texto inteiro ###
		
			# Juntando strings em uma
			txt_raw = " ".join([item['text'] for item in srt])
			st.markdown('# Texto transcrito')
			st.write(txt_raw)

			# Normalizando
			txt_raw = re.sub('\t|\n',' ',txt_raw)
			txt = re.sub('“|”',' ',txt_raw)
			txt = txt_raw.lower()
			txt = txt.translate(str.maketrans('', '', string.punctuation.replace("'",'')))
			txt = re.sub(r'(^|\s)\d+($|\s)',' ',txt)
			txt = txt.replace("’","'")

			# Lista de palavras sem as stopwords
			list_words = [wordnet_lemmatizer.lemmatize(word,pos= 'v') for word in txt.split(' ') if not word in STOPWORDS]

			### Pre processing texto no tempo ###
			# Adicionando a um DataFrame
			srt_time_processing = pd.DataFrame(srt)

			# Adicionando coluna de "end"
			srt_time_processing['end'] = srt_time_processing['start'] + srt_time_processing['duration']

			# Normalização das strings
			srt_time_processing['text'] = srt_time_processing['text'].str.lower()
			srt_time_processing['text'] = srt_time_processing['text'].apply(lambda t: t.translate(str.maketrans('', '', string.punctuation.replace("'",''))))
			srt_time_processing['text'] = srt_time_processing['text'].str.replace('\t|\n|“|”',' ',regex=True)
			srt_time_processing['text'] = srt_time_processing['text'].str.replace('’',"'")
			srt_time_processing['text'] = srt_time_processing['text'].str.replace(r'(^|\s)\d+($|\s)',' ',regex = True)

			# Retirando stop words e formando lista de palarvras
			srt_time_processing['text'] = srt_time_processing['text'].apply(lambda t: [wordnet_lemmatizer.lemmatize(word,pos= 'v') for word in t.split(' ') if not word in STOPWORDS])

			####### Contagem de palavras ##########
			st.markdown('# Contagem de palavras')

			vectorizer = CountVectorizer()
			X = vectorizer.fit_transform([' '.join(list_words)])

			# Printando
			st.write(pd.Series(index = vectorizer.get_feature_names(),
					data = X.toarray()[0]).sort_values(ascending = False).to_frame(name='Contagem'))

			# Gerando figura de bigramas
			wordcloud = WordCloud(stopwords=STOPWORDS,
								background_color="black",
								width=1600, height=800).generate(' '.join(list_words))

			# Plot
			fig, ax = plt.subplots(figsize=(10,6))
			ax.imshow(wordcloud, interpolation='bilinear')
			ax.set_axis_off()
			st.pyplot(fig)

			
			###### Contando bigramas ######
			st.markdown('# Contagem de bigramas')
			vectorizer = CountVectorizer(ngram_range = (2,2))
			X = vectorizer.fit_transform([' '.join(list_words)])

			st.write(pd.Series(index = vectorizer.get_feature_names(),
						data = X.toarray()[0]).sort_values(ascending = False).to_frame(name='Contagem'))

			# Gerando figura de palavras
			wordcloud = WordCloud(stopwords=STOPWORDS,
								background_color="black",
								collocation_threshold = 3,
								width=1600, height=800).generate(' '.join(list_words))

			# Plot
			fig, ax = plt.subplots(figsize=(10,6))
			ax.imshow(wordcloud, interpolation='bilinear')
			ax.set_axis_off()
			st.pyplot(fig)

			##### Palavras por segundo ######
			st.markdown('# Palavras por segundo no tempo de vídeo')

			# Contagem de "velocidade"
			srt_time_processing['quantity'] = srt_time_processing['text'].apply(lambda x: len(x))
			srt_time_processing['quantity per sec'] = (srt_time_processing['quantity'])/(srt_time_processing['duration'])

			#Plot
			fig, ax = plt.subplots()
			ax.plot(srt_time_processing['start'],srt_time_processing['quantity per sec'])
			ax.set_xlabel("Tempo do vídeo (em segundos)")
			ax.set_ylabel("Quantidade de palavras por segundo")
			st.pyplot(fig)

	elif choice == 'Análises Exploratórias 2':
		st.subheader('Análise de palavras em um range de tempo')
		# Buscando vídeo no Youtube
		raw_docx = st.text_area('Insira a parte final da URL do vídeo a ser analisado','fC9da6eqaqg')
		srt = YouTubeTranscriptApi.get_transcript(raw_docx)
		# Definindo range
		values = st.slider('Select a range of values',0.0, srt[-1]['start']+srt[-1]['duration'], (0.0, srt[-1]['start']+srt[-1]['duration']))
		
		if st.button("Enter"):

			## Preprocessing ##

			# Adicionando a um DataFrame
			srt_time_processing = pd.DataFrame(srt)

			# Adicionando coluna de "end"
			srt_time_processing['end'] = srt_time_processing['start'] + srt_time_processing['duration']

			# Normalização das strings
			srt_time_processing['text'] = srt_time_processing['text'].str.lower()
			srt_time_processing['text'] = srt_time_processing['text'].apply(lambda t: t.translate(str.maketrans('', '', string.punctuation.replace("'",''))))
			srt_time_processing['text'] = srt_time_processing['text'].str.replace('\t|\n|“|”',' ',regex=True)
			srt_time_processing['text'] = srt_time_processing['text'].str.replace('’',"'")
			srt_time_processing['text'] = srt_time_processing['text'].str.replace(r'(^|\s)\d+($|\s)',' ',regex = True)

			# Retirando stop words e formando lista de palarvras
			srt_time_processing['text'] = srt_time_processing['text'].apply(lambda t: [wordnet_lemmatizer.lemmatize(word,pos= 'v') for word in t.split(' ') if not word in STOPWORDS])

			# Contagens
			list_words_range = find_words_time_range(values[0],values[1],srt_time_processing)
			####### Contagem de palavras ##########
			st.markdown('# Contagem de palavras')

			vectorizer = CountVectorizer()
			X = vectorizer.fit_transform([' '.join(list_words_range)])

			# Plotando 
			st.write(pd.Series(index = vectorizer.get_feature_names(),
					data = X.toarray()[0]).sort_values(ascending = False).to_frame(name = 'Contagem'))

			# Gerando figura de palavras
			wordcloud = WordCloud(stopwords=STOPWORDS,
								background_color="black",
								width=1600, height=800).generate(' '.join(list_words_range))

			# Plot
			fig, ax = plt.subplots(figsize=(10,6))
			ax.imshow(wordcloud, interpolation='bilinear')
			ax.set_axis_off()
			st.pyplot(fig)

			
			###### Contando bigramas ######
			st.markdown('# Contagem de bigramas')
			vectorizer = CountVectorizer(ngram_range = (2,2))
			X = vectorizer.fit_transform([' '.join(list_words_range)])

			st.write(pd.Series(index = vectorizer.get_feature_names(),
						data = X.toarray()[0]).sort_values(ascending = False).to_frame(name='Contagem'))

			# Gerando figura de palavras
			wordcloud = WordCloud(stopwords=STOPWORDS,
								background_color="black",
								collocation_threshold = 3,
								width=1600, height=800).generate(' '.join(list_words_range))

			# Plot
			fig, ax = plt.subplots(figsize=(10,6))
			ax.imshow(wordcloud, interpolation='bilinear')
			ax.set_axis_off()
			st.pyplot(fig)

	elif choice == 'NER':
		st.subheader('Named Entity Recognizer')
		raw_docx = st.text_area('Insira a parte final da URL do vídeo a ser analisado','fC9da6eqaqg')
		if st.button("Enter"):
			srt = YouTubeTranscriptApi.get_transcript(raw_docx)
			# Juntando strings em uma
			txt_raw = " ".join([item['text'] for item in srt])
			docx = nlp(txt_raw)
			# NER
			spacy_streamlit.visualize_ner(docx, labels=nlp.get_pipe("ner").labels)



if __name__ == '__main__':
	main()
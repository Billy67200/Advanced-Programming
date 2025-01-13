#!/usr/bin/env python
# coding: utf-8

# ## LIBRARY

# In[1]:


from youtube_transcript_api import YouTubeTranscriptApi
import requests
from bs4 import BeautifulSoup
import openai
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import re, csv
from googleapiclient.discovery import build
from pytube import YouTube
from youtube_comment_downloader import YoutubeCommentDownloader
from textblob import TextBlob
import seaborn as sns
from colorama import init, Back, Fore, Style
from IPython.display import display, Markdown
import assemblyai as aai
import pandas as pd 



# ## CLEF API OPENAI ----------------------------------------------------------------------

# In[2]:


openai.api_key = "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"


# # ETAPE 1 : LIEN, BEAUTIFULSOUP, TITRE, ET TRANSCRIPT ------------------------

# In[3]:


url = "https://www.youtube.com/watch?v=5wzoQhitmdc"


# In[4]:


page = requests.get(url)
print(url)
soup = BeautifulSoup(page.text, "html.parser")
title = soup.title.text


# In[5]:


video_id = url.replace("https://youtu.be/", "") #dans le cas ou le lien de la vidéo est sous cette forme
video_id = url.replace("https://www.youtube.com/watch?v=", "") #dans le cas ou le lien de la vidéo est sous cette forme, comme c'est le cas ici
print(video_id)


# In[6]:


transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])


# In[7]:


print(transcript)


# ## SUMMARY AVEC GPT 3.5 CLASSIQUE ETC... : -------------------------------------------------

# In[8]:


output = "" 
for x in transcript:
 sentence = x['text'] #prend le text de x
 output = f'{output} {sentence}\n' #n permet de faire un saut de ligne, on prend la partie du text qui se situe dans sentence et on le met dans output puis saut de ligne etc..

response = openai.ChatCompletion.create(
 model="gpt-3.5-turbo", #modèle de gpt (base)
 messages=[
  {"role": "system", "content": "You are a journalist"}, #configure le comportement du modèle en définissant le contexte ou les règles générales
  {"role": "user", "content": "write a summary between 80 and 120 words"}, #founi les instructions
  {"role": "user", "content": output} #ce que l'utilisateur veut que le modèle traite ou répond.
 ]
)
summary = response.choices[0].message["content"]


# In[9]:


print(title)
print(video_id)
print(summary)
#print(tag)
#print(output)


# In[10]:


#from unidecode import unidecode

#title = unidecode(title)
#video_id = unidecode(video_id)
#output = unidecode(output)
#summary = unidecode(summary)


# In[11]:


#with open("youtube.html", "w", encoding="utf-8") as file :
#    file.write(f'<h1>{title} {video_id}</h1>')
#    file.write(f'<h2>Summary:</h2> <p>{summary}</p>')


# In[12]:


#file = open("youtube.html", "a") #inutile
#file.write(f'<h2>Summary:</h2> <p>{summary}</p>')
#file.write(f'<p><strong>Tags: </strong> {tag}</p>') #inutile belek
#file.write(f'<h3>Full Transcript:</h3>') #inutile belek


# ## FIN ETAPE SUMMARY -----------------------------------

# ## 1) CHAPITRAGE METHODE 1

# In[13]:


API_KEY = 'AIzaSyBwT7XyaS7E_RhxD32zXkRApp2lI3jwnd0'

def get_video_id(url):
    # extract video id from the URL
    video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    return video_id_match.group(1) if video_id_match else None

def get_video_title(video_id):
    # build the youTube service
    youtube = build('youtube', 'v3', developerKey=API_KEY)

    # fetch the video details
    request = youtube.videos().list(
        part='snippet',
        id=video_id
    )
    response = request.execute()

    # extract the title
    title = response['items'][0]['snippet']['title'] if response['items'] else 'Unknown Title'
    return title

def get_video_transcript(video_id, language='en'):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        return transcript
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def save_to_csv(title, transcript, filename):
    transcript_data = [{'start': entry['start'], 'text': entry['text']} for entry in transcript]
    df = pd.DataFrame(transcript_data)
    df.to_csv(filename, index=False)

    # save the title separately
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Title:', title])

def main():
    url = input('Enter YouTube URL: ')
    video_id = get_video_id(url)

    if not video_id:
        print('Invalid YouTube URL.')
        return

    title = get_video_title(video_id)
    transcript = get_video_transcript(video_id)

    if not transcript:
        print('No transcript available for this video.')
        return

    filename = f"{video_id}_transcript.csv"
    save_to_csv(title, transcript, filename)
    print(f'Transcript saved to {filename}')

if __name__ == '__main__':
    main()


# #### Analyse thématique et génération de chapitres à partir du transcript vidéo

# In[14]:


display(Markdown("### 1. Chargement du Dataset ###"))
transcript_df = pd.read_csv("XVzrEmVrkj4_transcript.csv")
display(transcript_df.head())

# Conversion des temps en numérique
transcript_df['start'] = pd.to_numeric(transcript_df['start'], errors='coerce')

display(Markdown("### 2. Aperçu du Dataset ###"))
display(Markdown("#### Dataset Overview ####"))
display(Markdown("```\n" + str(transcript_df.info()) + "\n```"))
display(Markdown("#### Basic Statistics ####"))
display(Markdown("```\n" + str(transcript_df.describe()) + "\n```"))

display(Markdown("### 3. Distribution des Longueurs de Texte ###"))
transcript_df['text_length'] = transcript_df['text'].apply(len)
plt.figure(figsize=(10, 5))
plt.hist(transcript_df['text_length'], bins=50, color='blue', alpha=0.7)
plt.title('Distribution of Text Lengths')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.show()

# Mots les plus fréquents
display(Markdown("### 4. Mots les Plus Fréquents ###"))
vectorizer = CountVectorizer(stop_words='english')
word_counts = vectorizer.fit_transform(transcript_df['text'])
word_counts_df = pd.DataFrame(word_counts.toarray(), columns=vectorizer.get_feature_names_out())
common_words = word_counts_df.sum().sort_values(ascending=False).head(20)
plt.figure(figsize=(10, 5))
common_words.plot(kind='bar', color='green', alpha=0.7)
plt.title('Top 20 Common Words')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()

display(Markdown("### 5. Modélisation des Sujets avec NMF ###"))
n_features = 1000
n_topics = 10
n_top_words = 10

tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
tf = tf_vectorizer.fit_transform(transcript_df['text'])
nmf = NMF(n_components=n_topics, random_state=42).fit(tf)
tf_feature_names = tf_vectorizer.get_feature_names_out()

def display_topics(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        topics.append(" ".join(topic_words))
    return topics

topics = display_topics(nmf, tf_feature_names, n_top_words)
display(Markdown("#### Identified Topics ####"))
for i, topic in enumerate(topics):
    display(Markdown(f"**Topic {i + 1}:** {topic}"))

display(Markdown("### 6. Distribution des Sujets Dominants ###"))
topic_distribution = nmf.transform(tf)

# Ajuster la longueur en supprimant les lignes en trop
topic_distribution_trimmed = topic_distribution[:len(transcript_df)]

# Calcul du sujet dominant pour chaque segment de texte
transcript_df['dominant_topic'] = topic_distribution_trimmed.argmax(axis=1)

# Analyse des coupures logiques
display(Markdown("### 7. Identification des Coupures Logiques ###"))
logical_breaks = []

for i in range(1, len(transcript_df)):
    if transcript_df['dominant_topic'].iloc[i] != transcript_df['dominant_topic'].iloc[i - 1]:
        logical_breaks.append(transcript_df['start'].iloc[i])

# Consolidation des coupures en chapitres plus larges
threshold = 60  # en secondes
display(Markdown("### 8. Consolidation des Coupures en Chapitres ###"))
consolidated_breaks = []
last_break = None

for break_point in logical_breaks:
    if last_break is None or break_point - last_break >= threshold:
        consolidated_breaks.append(break_point)
        last_break = break_point

# Fusion des coupures consécutives avec le même sujet dominant
final_chapters = []
last_chapter = (consolidated_breaks[0], transcript_df['dominant_topic'][0])

for break_point in consolidated_breaks[1:]:
    current_topic = transcript_df[transcript_df['start'] == break_point]['dominant_topic'].values[0]
    if current_topic == last_chapter[1]:
        last_chapter = (last_chapter[0], current_topic)
    else:
        final_chapters.append(last_chapter)
        last_chapter = (break_point, current_topic)

final_chapters.append(last_chapter)  # Ajout du dernier chapitre

# Conversion des chapitres en format horaire
display(Markdown("### 9. Conversion des Chapitres en Format Horaire ###"))
chapter_points = []
chapter_names = []

for i, (break_point, topic_idx) in enumerate(final_chapters):
    chapter_time = pd.to_datetime(break_point, unit='s').strftime('%H:%M:%S')
    chapter_points.append(chapter_time)

    # Récupération du texte pour le nom du chapitre
    chapter_text = transcript_df[(transcript_df['start'] >= break_point) & (transcript_df['dominant_topic'] == topic_idx)]['text'].str.cat(sep=' ')

    # Extraction de phrases clés pour créer un nom de chapitre
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3)
    tfidf_matrix = vectorizer.fit_transform([chapter_text])
    feature_names = vectorizer.get_feature_names_out()
    chapter_name = " ".join(feature_names)

    chapter_names.append(f"Chapter {i+1}: {chapter_name}")

# Affichage des points de chapitres finaux avec les noms
display(Markdown("### 10. Points de Chapitres Final avec Noms ###"))
for time, name in zip(chapter_points, chapter_names):
    display(Markdown(f"**{time}** - {name}"))


# ## 2) CHAPITRAGE METHODE 2 :

# In[15]:


def print_time(search_word,time):
    # calculate the accurate time according to the video's duration

    phrases = ["Top 10", "Top 9", "Top 8", "Top 7","Top 6","Top 5","Top 4","Top 3","Top 2","Top 1"]

# Créer une liste pour stocker les résultats, sans ça, difficile d'afficher le résumé avant le chapitrage car la fonction renvoie un print
    result = []

    for idx, t in enumerate(time):
        hours = int(t // 3600)
        min = int((t // 60) % 60)
        sec = int(t % 60)
        phrase = phrases[idx % len(phrases)]  # Sélectionner une phrase différente pour chaque timer
        result.append(f"{hours:02d}:{min:02d}:{sec:02d} : {phrase}")
        # Retourner la liste de résultats
    return result

    


# In[16]:


#print(f"{hours:02d}:{min:02d}:{sec:02d} : {phrase}")


# In[17]:


data = [t['text'] for t in transcript]
data = [re.sub(r"[^a-zA-Z0–9-ışğöüçiIŞĞÖÜÇİ ]", "", line) for line in data]


# In[18]:


sds = 'numéro'.replace('é', 'e')


# In[19]:


transcript


# search_word = ["numéro","numero", "Numéro", "Numero"]
# time = []
# for i, line in enumerate(data):
#     if mot in line:
#         start_time = transcript[i]['start']
#         time.append(start_time)
# 
# print_time(mot, time)

# In[20]:


# Parcours des lignes de la transcription
search_word = ["Number",
               "numero 9", "Numéro", "number", "place"]
phrases = ["Top 10", "Top 9", "Top 8", "Top 7","Top 6","Top 5","Top 4","Top 3","Top 2","Top 1"]
time = []
for i, line in enumerate(data):
    # Vérifier si l'un des mots de la liste 'mot' est présent dans la ligne
    if any(word in line for word in search_word):
        start_time = transcript[i]['start']
        time.append(start_time)
#des qu'il y a une occurence d'un mot de 'search word' dans le transcript, on prend  son start et on le met dans la liste time
    


# In[21]:


chapitrage = print_time(search_word, time)


# In[22]:


chapitrage


# ## FIN ----------------------------------------

# ## CHAPITRAGE : METHODE 3

# In[23]:


### Etape b diviser les sous-titres en sections logiques basées sur les sujets abordés.

## Nous allons d'abord diviser les sous-titres en segments raisonnables que nous pourrions analyser 
# par exemple chaque 100 mots).

def split_transcript(transcript, chunk_size=100):
    chunks = []
    current_chunk = ""
    current_start = None
    
    for entry in transcript: #entry=x
        if current_start is None:
            current_start = entry['start']
        
        current_chunk += " " + entry['text'] #rajoute/append à current chunk entry['text'], c'est comme si on faisait current shank = current shank + ...
        
        if len(current_chunk.split()) >= chunk_size:
            chunks.append({
                'start': current_start,
                'text': current_chunk.strip()
            })
            current_chunk = ""
            current_start = None
    
    if current_chunk:
        chunks.append({
            'start': current_start,
            'text': current_chunk.strip()
        })
    
    #return chunks

chunks = split_transcript(transcript)



# In[24]:


# Crée un client AssemblyAI
aai.settings.api_key = 'a7f24c416b4c4058af2460e896e56ee1'

# Si tu as un fichier audio local -> changer de vidéo, les top 10 = bad

audio_url = '/Users/vubilly/Downloads/You wont believe the door designs on these 10 classic cars Curious List.mp3' #2 chap
#audio_url = '/Users/vubilly/Downloads/Top 10 Vertical Take-off and Landing(VTOL) Warplanes Curious List.mp3' #-> 4 chap

transcriber = aai.Transcriber(
    config = aai.TranscriptionConfig(auto_chapters=True)
)
transcription = transcriber.transcribe(audio_url)
# Afficher la transcription
print(transcription.text, end='\n\n')


# In[25]:


for chapter in transcription.chapters: 
  print(f"Start: {chapter.start}, End: {chapter.end}") 
  print(f"Summary: {chapter.summary}")
  print(f"Healine: {chapter.headline}")
  print(f"Gist: {chapter.gist}")


# In[26]:


def ms_to_hms(start): #start en milliseconde
    s, ms = divmod(start, 1000) #divion puis reste
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return h, m, s


# In[27]:


def create_timestamps(chapters):
    last_hour = ms_to_hms(chapters[-1].start)[0] #Convertit le temps de début du dernier chapitre (en millisecondes) en heures, 
    #minutes et secondes puis extrait le nombre d’heures du dernier chapitre pour savoir si le temps total est inférieur à 1h ou pas
    time_format = "{m:02d}:{s:02d}" if last_hour == 0 else "{h:02d}:{m:02d}:{s:02d}"

    lines = []
    for idx, chapter in enumerate(chapters):
        # first YouTube timestamp must be at zero
        h, m, s = (0, 0, 0) if idx == 0 else ms_to_hms(chapter.start) #Si le chapitre est le premier (idx=0) le timeskip commence à zéro
        lines.append(f"{time_format.format(h=h, m=m, s=s)} {chapter.headline}")

    return "\n".join(lines) #Toutes les lignes formatées sont combinées en une chaîne unique, séparée par des sauts de ligne.


# In[28]:


timestamp_lines = create_timestamps(transcription.chapters)
print(timestamp_lines)
#premier test de chapitrage, avec des titres qui se basent sur ce qu'il dit


# In[29]:


#On peut utiliser openAI afin de reformuler les titres


# In[30]:


response = openai.ChatCompletion.create(
 model="gpt-3.5-turbo", #modèle de gpt (base)
 messages=[
  {"role": "system", "content": "You are a journalist."}, #configure le comportement du modèle en définissant le contexte ou les règles générales
  {"role": "user", "content": "reformulate the titles that appear in ‘timestamp_lines’ that I'm sending you below so that you get nicer titles, retains the form of timestamp_lines, but with times"}, #founi les instructions
  {"role": "user", "content": timestamp_lines} #ce que l'utilisateur veut que le modèle traite ou répond.
 ]
)
summary2 = response.choices[0].message["content"]


# In[31]:


print(summary2)


# In[32]:


#FAQ
output = "" 
for x in transcript:
 sentence = x['text'] #prend le text de x
 output = f'{output} {sentence}\n' #n permet de faire un saut de ligne, on prend la partie du text qui se situe dans sentence et on le met dans output puis saut de ligne etc.. on garde que le texte

faq = openai.ChatCompletion.create(
 model="gpt-3.5-turbo", #modèle de gpt (base)
 messages=[
  {"role": "system", "content": "You are a journalist."}, #configure le comportement du modèle en définissant le contexte ou les règles générales
  {"role": "user", "content": "Realize a FAQ of 5 pertinent questions/answers of the content I send you below"}, #founi les instructions
  {"role": "user", "content": output} #ce que l'utilisateur veut que le modèle traite ou répond.
 ]
)
FAQ2 = faq.choices[0].message["content"]


# In[33]:


print(FAQ2)


# ## ETAPE : ANALYSE DES SENTIMENTS

# In[34]:


display(Markdown("### 1. Récupérer l'ID de la vidéo et les commentaires ###"))
url = "https://www.youtube.com/watch?v=5wzoQhitmdc"
video = YouTube(url)
video_id = video.video_id

# Initialiser le downloader
downloader = YoutubeCommentDownloader()
comments = downloader.get_comments(video_id)

# Stocker les commentaires et analyser le sentiment
comment_texts = []
sentiments = []

for comment in comments:
    comment_texts.append(comment['text'])
    analysis = TextBlob(comment['text'])
    polarity = analysis.sentiment.polarity
    sentiments.append(polarity)

# Déterminer le type de sentiment pour chaque commentaire
sentiment_labels = ['Positif' if s > 0 else 'Négatif' if s < 0 else 'Neutre' for s in sentiments]

display(Markdown("### 2. Créer un DataFrame avec les Données ###"))
df = pd.DataFrame({
    "Comment": comment_texts,
    "Sentiment_Score": sentiments,
    "Sentiment_Label": sentiment_labels
})

display(Markdown("### 3. Enregistrer les Données dans un Fichier CSV ###"))
df.to_csv("youtube_comments_sentiment.csv", index=False)

display(Markdown("### 4. Visualisation des Données ###"))
plt.figure(figsize=(14, 6))

# Graphique 1: Répartition des sentiments
plt.subplot(1, 2, 1)
sns.countplot(data=df, x="Sentiment_Label", palette="coolwarm")
plt.title("Répartition des Sentiments")
plt.xlabel("Type de Sentiment")
plt.ylabel("Nombre de Commentaires")

# Graphique 2: Distribution des scores de sentiment
plt.subplot(1, 2, 2)
sns.histplot(df["Sentiment_Score"], kde=True, color="skyblue")
plt.title("Distribution des Scores de Sentiment")
plt.xlabel("Score de Sentiment")
plt.ylabel("Fréquence")

# Afficher les graphiques
plt.tight_layout()
plt.show()

display(Markdown("### 5. Analyse du Sentiment Général des Commentaires ###"))
average_sentiment = df["Sentiment_Score"].mean()
sentiment_summary = "positive" if average_sentiment > 0 else "negative" if average_sentiment < 0 else "neutral"

# Objet contenant le message à poster
comment_to_post = {
    "message": f"The general opinion of the comments on this video is {sentiment_summary}."
}

# Affichage du commentaire pour vérification
display(Markdown("### 6. Résumé des Résultats ###"))
print(comment_to_post["message"])
print("Score de sentiment moyen :", average_sentiment)
print("\nÉchantillon de commentaires analysés :")
print(df.head())


# ## FIN

# ## ETAPE FINALE : POSTER DANS LA SECTION COMMENTAIRE

# In[35]:


import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By


# In[36]:


from selenium.webdriver.support.ui import WebDriverWait


# In[37]:


# Variables
video_url = "https://www.youtube.com/watch?v=5wzoQhitmdc"  # Remplacer par l'URL de la vidéo
comment_text = f"""General opinion in the comments :

{comment_to_post["message"]}

Summary of the video :

{summary}

FAQ:

{FAQ2}

Watchtime :
{chr(10).join(chapitrage)}
"""

email = "Test.advancedprogramming67@gmail.com"
password = "!fjref454A6F45!#*54FR" 


# In[38]:


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def post_comment_on_youtube(video_url, comment_text, email, password):
    driver = webdriver.Firefox()
    wait = WebDriverWait(driver, 20)

    # Ouvrir la page de connexion Google
    driver.get("https://accounts.google.com/signin/v2/identifier")

    # Étape 1 : Entrer l'adresse email
    try:
        email_field = wait.until(EC.visibility_of_element_located((By.ID, "identifierId")))
        email_field.send_keys(email)
        email_field.send_keys(Keys.RETURN)
        print("Email saisi et touche Entrée pressée")
    except Exception as e:
        print("Erreur lors de la saisie de l'email :", e)
        driver.quit()
        return

    time.sleep(2)

    # Étape 2 : Entrer le mot de passe
    try:
        password_field = wait.until(EC.visibility_of_element_located((By.XPATH, "//input[@type='password']")))
        password_field.send_keys(password)
        password_field.send_keys(Keys.RETURN)
    except Exception as e:
        print("Erreur lors de la saisie du mot de passe :", e)
        driver.quit()
        return

        # Étape 3 : Accepter les cookies si nécessaire
    try:
        time.sleep(15)
        
        # Identifier le conteneur de la modale de cookies
        cookie_modal = wait.until(EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'modal') or contains(@class, 'cookie')]")))

        # Scroller dans la modale des cookies pour atteindre le bouton "Tout accepter"
        driver.execute_script("arguments[0].scrollBy(0, 500);", cookie_modal)
        time.sleep(1)  # Pause pour que le navigateur prenne en compte le scroll

        # Cliquer sur le bouton "Tout accepter" dans le modal de cookies
        accept_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(),'Tout accepter')]")))
        accept_button.click()
        print("Cookies acceptés.")
    except Exception as e:
        print("Aucune page de cookies ou bouton 'Accepter' non trouvé :", e)


    # Étape 4 : Ouvrir la vidéo YouTube
    try:
        driver.get(video_url)
        time.sleep(5)
    except Exception as e:
        print("Erreur lors de l'ouverture de la vidéo :", e)
        driver.quit()
        return

    # Étape 5 : Faire défiler la page légèrement
    try:
        driver.execute_script("window.scrollBy(0, 500);")
        time.sleep(2)
        print("Page déroulée légèrement jusqu'aux commentaires.")
    except Exception as e:
        print("Erreur lors du défilement de la page :", e)

    # Étape 6 : Rechercher et poster le commentaire
    try:
        # Trouver la zone de commentaire et la cliquer pour l'activer
        comment_box = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "ytd-comment-simplebox-renderer #placeholder-area")))
        comment_box.click()  # Cliquer sur la zone pour activer la saisie
        print("Zone de commentaire activée.")

        # Rechercher la zone éditable pour taper le commentaire
        editable_area = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "ytd-comment-simplebox-renderer #contenteditable-root")))
        editable_area.send_keys(comment_text)
        time.sleep(1)  # Pause pour être sûr que le texte est bien entré

        # Trouver et cliquer sur le bouton pour publier le commentaire
        submit_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//ytd-button-renderer[@id='submit-button']//button")))
        submit_button.click()
        print("Commentaire posté avec succès.")
    except Exception as e:
        print("Erreur lors de la publication du commentaire :", e)
    finally:
        time.sleep(7)
        driver.quit()


# In[39]:


# Poster le commentaire
post_comment_on_youtube(video_url, comment_text, email, password)


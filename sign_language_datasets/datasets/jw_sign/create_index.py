"""Helper file to crawl The Jehovah Witness Sign Language Website and create an up-to-date index of the dataset"""

!pip3 install requests-html
import pandas as pd
import urllib
import pickle
import gzip
import json
import subprocess
from requests_html import HTMLSession
from urllib.request import urlopen
from urllib.error import HTTPError

#tks = json.loads(open("tkl.json", "r").read()) #'en': ...
s2t = json.loads(open("s2t.json", "r").read()) #'JML': 'en'
text_dict = json.loads(open("text_dict.json", "r").read())
valid_signs = json.loads(open("valid_signs.json", "r").read())

#TODO add an image of the english names and their initials
user_sign = "XXX" #TODO a way to take this input from the user
if user_sign in valid_signs:
  sign_language_ini = [user_sign]
elif user_sign == "ALL":
  sign_language_ini = valid_signs
else: 
  print("Invalid Sign")
  # TODO a line should be here to end the script     

  
# text scrapping  
text_lang = s2t[user_sign]  
text_info = text_dict[text_lang]  
sign_lang_urls = [text_info]
session = HTMLSession()
#sl_text = dict() #sign_language texts i.e. annotations. each key represent a text/spoken language. 
base = "https://www.jw.org"
#g = 0 # just to keep track
t1=1  # these times are essential to make sure the site doesn't receive too many requests at once. 
for tlang in sign_lang_urls:
  tl_dict = dict() #text language dictionary. 
  sl_url = tlang["base_url"]
  books = session.get(sl_url).html.find('.booksContainer')[0].find('a[href]')
  for bk in books:
    session = HTMLSession()      
    bk_url = base + bk.attrs['href']
    chapters = session.get(bk_url).html.find('a.chapter')      
    for ch in chapters:
      ch_url = base + ch.attrs['href'] 
      verses = session.get(ch_url).html.find('div#bibleText')[0].find('span.verse')        
      for ver in verses:
        tl_dict[ver.attrs['id']] = ver.text
        #g = g + 1
        #print(f"verse {g} - {ver.attrs['id']}") 
    time.sleep(t1)
  #sl_text[tlang["lang_ini"]] = tl_dict
  print("Text done.")  
tks = set(tl_dict.keys()) 
    
    
# videos scrapping
index_data = [] 
# list which contains all the videos and associated information
shester = 1 
for slang in sign_language_ini:
  slang_videos = []
  for bnum in range(1,67):
    new_video = dict()
    try:
      link = "https://b.jw-cdn.org/apis/pub-media/GETPUBMEDIALINKS?pub=nwt&langwritten=" + slang + "&txtCMSLang=" + slang + "&booknum=" + str(bnum) + "&alllangs&output=json&fileformat=MP4%2CM4V%2C3GP"
      response = urlopen(link)
    except HTTPError as err:
      continue    
    all_data_json = json.loads(response.read())
    data_json = all_data_json["files"][slang]
    video_types = list(data_json.keys())
    for vt in video_types:
      for vid in data_json[vt]:
        if (vid["label"] == "720p" 
            and vid["frameRate"] == 29.97
            and vid["duration"] != 0.0):   
          vid_url = vid["file"]["url"]
          vid_name = vid_url.split('/')[-1].split('.')[0]
          command = f"ffprobe -i {vid_url} -print_format default -show_chapters -loglevel error -v quiet -of json"
          result = subprocess.run(command.split(), capture_output=True)        
          if result.returncode == 0:
              output = result.stdout.decode('utf-8')
              data = json.loads(output)
          else:
            continue               
          try:
            ver_all = data["chapters"]          
          except:
            continue
          if len(ver_all)==0:
            continue
          print(f"Video {shester}")
          shester = shester + 1
          for ver in ver_all:
            verse_dict = {}
            ver_name = ver["tags"]["title"]
            count_colon = ver_name.count(":")
            name_split = vid_name.split("_") 
            # ["0-Pub","1-Bknumb","2-Bkname","3-SignLang","4-Chaptnumb","5-Quality"]
            try:
              if count_colon == 1:
                versenumb = ver_name.rsplit(':', 1)[1]
                chaptnumb = name_split[4]
              elif count_colon == 0:
                if name_split[1] in ["31","57","63","64","65"] :
                  versenumb = ver_name.rsplit(' ', 1)[1]
                  chaptnumb = "1"
                else:
                  continue
              else: #count_colon 2,3,4,... 
                continue
            except:
              pass           
            verse_dict["video_url"] = vid_url
            verse_dict["video_name"] = vid_name
            verse_dict["verse_lang"] = name_split[3]
            verse_dict["verse_name"] = ver_name  #"Mic. 6:8"
            verse_dict["verse_start"]= ver["start_time"]
            verse_dict["verse_end"] = ver["end_time"]
            verse_dict["duration"] = float(ver["end_time"]) - float(ver["start_time"])
            verse_dict["verse_unique"] = name_split[3] + " " + ver_name # "ISL Mic. 6:8"
            try:          
              id = "v" + str(int(name_split[1])) + str("{:03d}".format(int(chaptnumb))) + str("{:03d}".format(int(versenumb))) #"v33006008"
            except:
              continue
            if id in tks:
              verse_dict["verseID"] = id
              verse_dict["verse_text"] = tl_dict[id] #TODO clean text
              index_data.append(verse_dict)
              
# remove all duplications inclusive
df = pd.DataFrame.from_dict(index_data)
df.drop_duplicates(subset="verse_unique", keep=False, inplace=True)
index_data = df.to_dict('records')              
              

# store
jsonString = json.dumps(index_data)
jsonFile = open(f"{user_sign}.json", "w")
jsonFile.write(jsonString)
jsonFile.close()  

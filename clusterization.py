from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 
from keras.applications.vgg16 import VGG16 
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd


with open('data_480p.json', 'r') as f:
    data = json.loads(f.read())
            
df = pd.DataFrame.from_dict(data)
tstamps = list(set(df.Timestamps))

df['img'] = ''
df = df.drop('PolygonVerts')

vidcap = cv2.VideoCapture('test.mp4')


for stamp in tstamps:
    vidcap.set(cv2.CAP_PROP_POS_MSEC,stamp)
    success, frame = vidcap.read()

    num_obj = len(df.loc[df['Timestamp'] == stamp])
    for i in num_obj:
        bbox = df.loc[df['Timestamp'] == stamp].loc[i].BoundingBoxes
        df.loc[df['Timestamp'] == stamp].loc[i]['img'] = np.asarray(frame[bbox[0]:bbox[2], bbox[1]:bbox[3]], dtype = float64)


model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

def extract_features(file, model):
    img = file.reshape(224,224,3)
    img = np.array(img) 
    reshaped_img = img.reshape(1,224,224,3) 
    imgx = preprocess_input(reshaped_img)
    features = model.predict(imgx, use_multiprocessing=True)
    return features
   
data = {}

for i in len(df):
    data[df.loc[i].index] = extract_features(df.loc[i].img)



filenames = np.array(list(data.keys()))

feat = np.array(list(data.values()))

feat = feat.reshape(-1,4096)

pca = PCA(n_components=100, random_state=22)
pca.fit(feat)
x = pca.transform(feat)

kmeans = KMeans(n_clusters=len(unique_labels),n_jobs=-1, random_state=22)
kmeans.fit(x)

groups = {}
for file, cluster in zip(filenames,kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)
     
def view_cluster(cluster):
    plt.figure(figsize = (25,25));
    files = groups[cluster]   
    for index, file in enumerate(files):
        plt.subplot(10,10,index+1);
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')
        

rss = []
list_k = list(range(3, 50))

for k in list_k:
    km = KMeans(n_clusters=k, random_state=22, n_jobs=-1)
    km.fit(x)    
    rss.append(km.inertia_)

plt.figure(figsize=(6, 6))
plt.plot(list_k, rss)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distance');
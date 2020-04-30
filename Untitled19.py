#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import requests # library to handle requests
from pandas.io.json import json_normalize


# In[2]:


import pandas as pd
url='https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'

df=pd.read_html(url, header = 0)[0]
df.head()


# In[3]:


df.columns = ['Postal Code', 'Borough', 'Neighborhood']


# In[4]:


df=df[df['Borough']!='Not assigned']


# In[5]:


df['Neighborhood']=df['Neighborhood'].str.replace('/', ',')


# In[6]:


df.shape


# In[7]:


File = 'http://cocl.us/Geospatial_data'

df2=pd.read_csv('http://cocl.us/Geospatial_data')
df2.head()


# In[8]:


df5 =pd.merge(df, df2, on = 'Postal Code', how ='left')


# In[9]:


df5.head()


# In[10]:


from geopy.geocoders import Nominatim
address = 'Toronto, Canada'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto, Canada are {}, {}.'.format(latitude, longitude))


# In[11]:


#!conda install -c conda-forge folium=0.5.0 --yes 
import folium


# In[12]:


map_Toronto = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, label in zip(df5['Latitude'], df5['Longitude'], df5['Neighborhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_Toronto)  
    
map_Toronto


# In[13]:


CLIENT_ID = 'Q1PQW351R32NOHSNDUW1VHOAGRIF1WRPTCHXZOZVGNH2GPTM' # your Foursquare ID
CLIENT_SECRET = 'TNMPBDB1CV2YNQURUHMAHYPFAU2VMLD0N3VCFX0IKHH51CYX' # your Foursquare Secret
VERSION = '20200429' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[14]:


df5.loc[0, 'Neighborhood']


# In[15]:


neighborhood_latitude=df5.loc[0,'Latitude']
neighborhood_longitude = df5.loc[0,'Longitude']
neighborhood_longitude


# In[16]:



LIMIT = 100

radius = 500

url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)
url


# In[17]:


import requests # library to handle requests
results = requests.get(url).json()
results


# In[18]:


# function that extracts the category of the venue
def get_category_type(row):
    categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[19]:


venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()


# In[20]:


print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# In[21]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])


    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[22]:


Toronto_venues = getNearbyVenues(df5['Neighborhood'], df5['Latitude'],df5['Longitude'])


# In[23]:


print(Toronto_venues.shape)
Toronto_venues.head()


# In[24]:


Toronto_venues.groupby('Neighborhood').count()


# In[25]:


print('There are {} uniques categories.'.format(len(Toronto_venues['Venue Category'].unique())))


# In[26]:


Toronto_venues['Venue Category'].unique()


# In[27]:


Toronto_onehot = pd.get_dummies(Toronto_venues[['Venue Category']], prefix="", prefix_sep="")
Toronto_onehot


# In[28]:


Toronto_onehot['Neighborhood1'] = Toronto_venues['Neighborhood'] 


# In[29]:


fixed_columns = [Toronto_onehot.columns[-1]] + list(Toronto_onehot.columns[:-1])
fixed_columns


# In[30]:


Toronto_onehot = Toronto_onehot[fixed_columns]


# In[31]:


Toronto_onehot.head()


# In[32]:


Toronto_onehot.shape


# In[33]:


Toronto_grouped= Toronto_onehot.groupby('Neighborhood1').mean().reset_index()
Toronto_grouped


# In[34]:


Toronto_grouped.shape


# In[35]:


num_top_venues = 5

for hood in Toronto_grouped['Neighborhood1']:
    print("----"+hood+"----")
    temp = Toronto_grouped[Toronto_grouped['Neighborhood1'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[36]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[37]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood1']

np.arange(num_top_venues)



# In[38]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = Toronto_grouped['Neighborhood1']

for ind in np.arange(Toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(Toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# In[39]:


from sklearn.cluster import KMeans
# set number of clusters
kclusters = 5

Toronto_grouped_clustering = Toronto_grouped.drop('Neighborhood1', 1)


# In[40]:


kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(Toronto_grouped_clustering)
kmeans.labels_[0:93] 


# In[41]:


Toronto_grouped_clustering


# In[42]:


neighborhoods_venues_sorted.insert(0, 'Cluster Labels',kmeans.labels_)
neighborhoods_venues_sorted


# In[43]:


Toronto_merged = df5
Toronto_merged = Toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')


# In[44]:


Toronto_merged.head(20)


# In[45]:


Toronto_merged.dropna(inplace = True)


# In[46]:


import matplotlib.cm as cm
import matplotlib.colors as colors

# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)


# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]


# add markers to the map

markers_colors = []
for lat, lon, poi, cluster in zip(Toronto_merged['Latitude'], Toronto_merged['Longitude'], Toronto_merged['Neighborhood'], Toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[int(cluster-1)],
        fill=True,
        fill_color=rainbow[int(cluster-1)],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[47]:


Toronto_merged.loc[Toronto_merged['Cluster Labels'] == 0, Toronto_merged.columns[[1] + list(range(5, Toronto_merged.shape[1]))]]


# In[48]:


list(range(5, Toronto_merged.shape[1]))


# In[49]:


Toronto_merged.columns[[1] + list(range(5, Toronto_merged.shape[1]))]


# In[50]:


Toronto_merged.shape[1]


# In[51]:


list(range(5,16))


# In[52]:


[[1]+list(range(5,16))]


# In[53]:


Toronto_merged.loc[Toronto_merged['Cluster Labels']==0, Toronto_merged.columns[list(range(5,Toronto_merged.shape[1]))]]


# In[54]:


Toronto_merged.loc[Toronto_merged['Cluster Labels']==1, Toronto_merged.columns[list(range(5, Toronto_merged.shape[1]))]]


# In[55]:


Toronto_merged.loc[Toronto_merged['Cluster Labels']==2, Toronto_merged.columns[list(range(5, Toronto_merged.shape[1]))]]


# In[56]:


Toronto_merged.loc[Toronto_merged['Cluster Labels']==3, Toronto_merged.columns[list(range(5, Toronto_merged.shape[1]))]]


# In[57]:


Toronto_merged.loc[Toronto_merged['Cluster Labels']==4, Toronto_merged.columns[list(range(5, Toronto_merged.shape[1]))]]


# In[58]:


def ClusterLabels(row):
    Cluster_Labels1 = row['Cluster Labels']
    if Cluster_Labels1==0:
        return 'Quite/Close to park'
    elif Cluster_Labels1 ==1:
        return 'Busy/Resturant Orientated'
    elif Cluster_Labels1 == 2:
        return 'Busy/Close to park'
    elif Cluster_Labels1 == 3:
        return 'Peaceful/Close to bar'
    else:
        return 'Quite/Sport Orientated'


# In[59]:


Toronto_merged.shape


# In[60]:


Toronto_merged['Cluster Labels'] = Toronto_merged.apply(ClusterLabels, axis = 1)


# In[61]:


Toronto_merged.head(20)


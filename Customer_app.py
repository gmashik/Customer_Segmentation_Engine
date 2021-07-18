import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from sklearn.cluster import KMeans
import plotly.figure_factory as ff
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import  DBSCAN
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
#
st.markdown("<h1 style='text-align: center; color: blue;'><b>Customer Segmentation Engine</b></h1>", unsafe_allow_html=True)
link = '[here](https://github.com/gmashik/Customer_Segmentation_Engine)'
st.markdown("For code and data click "+link, unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: red;'><b>3-Dimensional K-Means Clustering Analysis</b></h2>", unsafe_allow_html=True)
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
data = pd.read_csv('Mall_Customers.csv')
if st.button('Show Full Customer Data'):
    st.header('Full Customer Data') 
    ndat=data
    ndat=ndat.set_index('CustomerID')
    ndat=data.style.background_gradient(cmap = 'icefire')
    st.dataframe(ndat)
x2 = data.loc[:, ['Age','Annual Income (k$)','Spending Score (1-100)']].values
km = KMeans(n_clusters = 6, init = 'k-means++', max_iter = 300, n_init = 20, random_state = 0)
y_means = km.fit_predict(x2)
cen={"x":km.cluster_centers_[:,0],"y":km.cluster_centers_[:,1],"z":km.cluster_centers_[:,2]}
clusterdf=pd.DataFrame(cen)
layout = go.Layout(
    title = 'K Means Clustering using Age, Spending Score and Annual Income',
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    ),
    scene = dict(
            xaxis = dict(title  = 'Age'),
            yaxis = dict(title  = 'Annual Income'),
            zaxis = dict(title  = 'Spending Score')
        )
)

fig = go.Figure([go.Scatter3d(
    x= x2[y_means == 0, 0],
    y= x2[y_means == 0, 1],
    z= x2[y_means == 0, 2],
    mode='markers',
    name='Cluster_1',
     marker=dict(
        color = ["red"]*len(x2[y_means == 0, 0]), 
        size= 10,
        line=dict(
            color= ["red"]*len(x2[y_means == 0, 0]),
            width= 12
        ),
        
        opacity=0.8
     )
),
go.Scatter3d(
    x= x2[y_means == 1, 0],
    y= x2[y_means == 1, 1],#data['Spending Score (1-100)'],
    z= x2[y_means == 1, 2],
    mode='markers',
    name='Cluster_2',
     marker=dict(
        color = ["blue"]*len(x2[y_means == 1, 0]), 
        size= 10,
        line=dict(
            color=["blue"]*len(x2[y_means == 1, 0]),
            width= 12
        ),
        
        opacity=0.8
     )
),
go.Scatter3d(
    x= x2[y_means == 2, 0],
    y= x2[y_means == 2, 1],#data['Spending Score (1-100)'],
    z= x2[y_means == 2, 2],
    mode='markers',
    name='Cluster_3',
     marker=dict(
        color = ["yellow"]*len(x2[y_means == 2, 0]), 
        size= 10,
        line=dict(
            color= ["yellow"]*len(x2[y_means == 2, 0]),
            width= 12
        ),
        
        opacity=0.8
     )
),
go.Scatter3d(
    x= x2[y_means == 3, 0],
    y= x2[y_means == 3, 1],#data['Spending Score (1-100)'],
    z= x2[y_means == 3, 2],
    mode='markers',
    name='Cluster_4',
     marker=dict(
        color = ["orange"]*len(x2[y_means == 3, 0]), 
        size= 10,
        line=dict(
            color= ["orange"]*len(x2[y_means == 3, 0]),
            width= 12
        ),
        
        opacity=0.8
     )
),
go.Scatter3d(
    x= x2[y_means == 4, 0],
    y= x2[y_means == 4, 1],#data['Spending Score (1-100)'],
    z= x2[y_means == 4, 2],
    mode='markers',
    name='Cluster_6',
     marker=dict(
        color = ["cyan"]*len(x2[y_means == 4, 0]), 
        size= 10,
        line=dict(
            color= ["cyan"]*len(x2[y_means == 4, 0]),
            width= 12
        ),
        
        opacity=0.8
     )
),
go.Scatter3d(
    x= x2[y_means == 5, 0],
    y= x2[y_means == 5, 1],#data['Spending Score (1-100)'],
    z= x2[y_means == 5, 2],
    mode='markers',
    name='Cluster_5',
     marker=dict(
        color = ["limegreen"]*len(x2[y_means == 5, 0]), 
        size= 10,
        line=dict(
            color= ["limegreen"]*len(x2[y_means == 5, 0]),
            width= 12
        ),
        
        opacity=0.8
     )
),
go.Scatter3d(
    x= clusterdf['x'],
    y= clusterdf['y'],#data['Spending Score (1-100)'],
    z= clusterdf['z'],
    mode='markers',
    name="Centroids",
     marker=dict(
        color = [], 
        size= 10,
        line=dict(
            color= [],#data['labels'],,
            width= 1
        ),
        
        opacity=0.8
     )
)], layout = layout)

fig.update_layout(scene = dict(
                    xaxis = dict(
                         backgroundcolor="rgb(200, 200, 230)",
                         gridcolor="white",
                         showbackground=True,
                         zerolinecolor="white",),
                    yaxis = dict(
                        backgroundcolor="rgb(230, 200,230)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white"),
                    zaxis = dict(
                        backgroundcolor="rgb(230, 230,200)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",),),
                    width=900,
                    height=700,
                    margin=dict(
                    r=10, l=10,
                    b=10, t=40)
                  )
st.plotly_chart(fig)
st.write("**Summary of the observations of the above data clustering:**")
st.write("1. From the sihlouette score analysis and elbow method it can be found that we can fit \
           the data within 6 clusters. \n 2. The spending score for customers in cluster 1 is midium\
               where most of them are young and their income is low. \
                   \n 3. The spending score of the customers in cluster 4 is relatively high where their \
                   income is varying from mid to high and their maximum age is around 40. \n 4. The income of  the \
                       customers in cluster 1 and 2  are midium as well as their spending score. However, \
                people in cluster 1 are representing younger people where older customers are represented by \
                    cluster 2. \n 5. People of all ages with relatively low income and spending \
                        scores are repsenting by cluster 5. \n 6. Finally, cluster 3 consists of the customers \
                            having midium to high income with low spending score. The age range for this cluster is 18-60.")

st.write("We've shown how we choose the number of clusters for our **K-Means** algorithm by doing \
    silhouette score analysis and elbow method below. We have also shown the clustering analysis using **DBSCAN** algorithm in the end. ")
st.write("However, For this dataset it can be observed that **K-Means** produce more efficient result than **DBSCAN**.")
st.write("**So, for this dataset K-Means algorithm is preferred. The perfomance of DBSCAN algorithm is \
    not satisfactory since the data is not densly separated here. **")
l=[]
p=[2,3,4,5,6,7,8,9,10]
for i in p:
  km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 20, random_state = 0)
  km.fit_predict(x2)
  labels =km.labels_
  ts=silhouette_score(x2, labels, metric='euclidean')
  #print(f"For {i} number of cluster silhouette score is {ts}")
  l.append(ts)
fig12 = go.Figure()
fig12.add_trace(go.Scatter(x=[i for i in range(2, 11)], y=l,
                    mode='lines+markers',
                    name='Elbow method'))

fig12.add_shape(type="circle",
    xref="x", yref="y",
    x0=p[l.index(max(l))]-.2, y0=max(l)-max(l)*.03, x1=p[l.index(max(l))]+.2, y1=max(l)+max(l)*.03,
    line_color="Red",
)
fig12.update_layout(
    title="Silhouette score for different number of clusters",
    xaxis_title="Number_of_Cluster",
    yaxis_title="silhouette_score",
    width=800,
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)
st.header("Silhouette score visualization for different number of clusters")
st.plotly_chart(fig12)
image1 = Image.open('3dsihl.png')
st.image(image1, caption='')
# fig,ax=plt.subplots(3,2,figsize=(15,8))
# for i in [2,3,4,5,6,7]:
#   km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 101)
#   q,mod=divmod(i,2)
#   visualaizer=SilhouetteVisualizer(km,color='yellowbricks',ax=ax[q-1][mod])
#   visualaizer.fit(x2)

st.write("We've got highest Silhouette score for 6 clusters. Similar result achieved from elbow method")
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    km.fit(x2)
    wcss.append(km.inertia_)
fig13 = go.Figure()
fig13.add_trace(go.Scatter(x=[i for i in range(1, 11)], y=wcss,
                    mode='lines+markers',
                    name='Elbow method'))

fig13.add_shape(type="circle",
    xref="x", yref="y",
    x0=5.8, y0=wcss[5]-wcss[5]*.15, x1=6.2, y1=wcss[5]+wcss[5]*.15,
    line_color="Red",
)
fig13.update_layout(
    title="Elbow Method",
    xaxis_title="Number_of_Cluster",
    yaxis_title="WCSS",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)
st.plotly_chart(fig13)
st.markdown("<h2 style='text-align: center; color: red;'><b>2-Dimensional Clustering Analysis using Age and Spending Score</b></h2>", unsafe_allow_html=True)
x1 = data.loc[:, ['Age','Spending Score (1-100)']].values
km = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(x1)
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x1[y_means == 0, 0], y=x1[y_means == 0, 1],
    name='Midium_spending_younger_customer',
    mode='markers',
    marker_color='rgba(152, 0, 0, .8)'
))

fig.add_trace(go.Scatter(
    x=x1[y_means == 1, 0], y=x1[y_means == 1, 1],
    name='High_spending_younger',
    mode='markers',
    marker_color='rgba(255, 0, 13, .9)'
))
fig.add_trace(go.Scatter(
    x=x1[y_means == 2, 0], y=x1[y_means == 2, 1],
    name='Midium_to_low_all_age',
    mode='markers',
    marker_color='rgba(55, 182, 19, .9)'
))
fig.add_trace(go.Scatter(
    x=x1[y_means == 3, 0], y=x1[y_means == 3, 1],
    name='Midium_spending_not_young',
    mode='markers',
    marker_color='rgba(255, 182, 193, .9)'
))
fig.add_trace(go.Scatter(
    x=km.cluster_centers_[:,0], y=km.cluster_centers_[:, 1],
    name='Centroids',
    mode='markers',
    marker_color='royalblue')
)

fig.update_traces(mode='markers', marker_line_width=2, marker_size=10)
fig.update_layout(title='K-Means Clustering using Age and Spending Score',
                  xaxis_title="Age",
                  yaxis_title="Spending Score",
                  yaxis_zeroline=False, xaxis_zeroline=False,
                  width=700)
st.plotly_chart(fig)
st.write("** When Age and spending score is taken we've found from the sihlouette score analysis and elbow method that \
          4 clusters can be a good fit for this data.**")
st.header("Silhouette score visualization for different number of clusters")
image2 = Image.open('shil2dagsp.png')
st.image(image2, caption='')
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    km.fit(x1)
    wcss.append(km.inertia_)
fig12 = go.Figure()
fig12.add_trace(go.Scatter(x=[i for i in range(1, 11)], y=wcss,
                    mode='lines+markers',
                    name='Elbow method'))

fig12.add_shape(type="circle",
    xref="x", yref="y",
    x0=3.8, y0=wcss[3]-wcss[3]*.15, x1=4.2, y1=wcss[3]+wcss[3]*.15,
    line_color="Red",
)
fig12.update_layout(
    title="Elbow Method",
    xaxis_title="Number_of_Cluster",
    yaxis_title="WCSS",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)
st.plotly_chart(fig12)
st.markdown("<h2 style='text-align: center; color: red;'><b>2-Dimensional Clustering Analysis using Annual Income and Spending Score</b></h2>", unsafe_allow_html=True)
x = data.loc[:, ['Annual Income (k$)','Spending Score (1-100)']].values
km = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 20, random_state = 0)
y_means = km.fit_predict(x)
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x[y_means == 0, 0], y=x[y_means == 0, 1],
    name='High_spend_low_spend',
    mode='markers',
    marker_color='rgba(152, 0, 0, .8)'
))

fig.add_trace(go.Scatter(
    x=x[y_means == 1, 0], y=x[y_means == 1, 1],
    name='General_Category',
    mode='markers',
    marker_color='rgba(255, 0, 13, .9)'
))
fig.add_trace(go.Scatter(
    x=x[y_means == 2, 0], y=x[y_means == 2, 1],
    name='High_income_high_spend',
    mode='markers',
    marker_color='rgba(55, 182, 19, .9)'
))
fig.add_trace(go.Scatter(
    x=x[y_means == 3, 0], y=x[y_means == 3, 1],
    name='Regular',
    mode='markers',
    marker_color='rgba(255, 182, 193, .9)'
))
fig.add_trace(go.Scatter(
    x=x[y_means == 4, 0], y=x[y_means == 4, 1],
    name='Careful',
    mode='markers',
    marker_color='rgba(205, 0, 193, 66.9)'
))
fig.add_trace(go.Scatter(
    x=km.cluster_centers_[:,0], y=km.cluster_centers_[:, 1],
    name='Centroids',
    mode='markers',
    marker_color='royalblue')
)
# Set options common to all traces with fig.update_traces
fig.update_traces(mode='markers', marker_line_width=2, marker_size=10)
fig.update_layout(title='K Means Clustering between Annual Income and Spending Score',
                  xaxis_title="Annual income",
                  yaxis_title="Spending Score",
                  yaxis_zeroline=False, xaxis_zeroline=False)
st.plotly_chart(fig)
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    km.fit(x)
    wcss.append(km.inertia_)
fig12 = go.Figure()
fig12.add_trace(go.Scatter(x=[i for i in range(1, 11)], y=wcss,
                    mode='lines+markers',
                    name='Elbow method'))

fig12.add_shape(type="circle",
    xref="x", yref="y",
    x0=4.8, y0=wcss[4]-wcss[4]*.15, x1=5.2, y1=wcss[4]+wcss[4]*.15,
    line_color="Red",
)
fig12.update_layout(
    title="Elbow Method",
    xaxis_title="Number_of_Cluster",
    yaxis_title="WCSS",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)
st.header("Silhouette score Visualization for different number of clusters")
image3 = Image.open('shil2daisp.png')
st.image(image3, caption='')
st.plotly_chart(fig12)
st.write("** When anual income and spending score is considered we've found from the sihlouette score analysis and elbow method that \
          5 clusters can be a good fit for this data. In the next section we will show some exploratory data analysis in the next section.**")
st.markdown("<h2 style='text-align: center; color: red;'><b>Exploratory Data Analysis </b></h2>", unsafe_allow_html=True)
st.write("**Data Description:**")
st.dataframe(data.drop('CustomerID',axis=1).describe())
st.write("**Male vs Female customer percentage:**")
figmf = go.Figure(data=[go.Pie(labels=["Female","Male"], values=data["Gender"].value_counts(), textinfo='label+percent'
                            )])
st.plotly_chart(figmf)
sex=['Male', 'Female']
st.write("**We have more female customers than male customers in our dataset.**")
st.write("** Age and Annual Income distribution:** ")
image4 = Image.open('agindis.png')
st.image(image4, caption='')
hist_data1 = [[data["Age"].iloc[i]  for i in range(len(data["Age"])) if data["Gender"].iloc[i]=="Male"],[data["Age"].iloc[i]  for i in range(len(data["Age"])) if data["Gender"].iloc[i]=="Female"]]
group_labels = ['Male','Female'] # name of the dataset

fig4445 = ff.create_distplot(hist_data1, group_labels)
fig4445.update_layout(title_text='Male vs Female Age distribution')
st.write("**Male vs Female Age distribution:**")
st.plotly_chart(fig4445)
hist_data2 = [[data['Annual Income (k$)'].iloc[i]  for i in range(len(data['Annual Income (k$)'])) if data["Gender"].iloc[i]=="Male"],[data['Annual Income (k$)'].iloc[i]  for i in range(len(data['Annual Income (k$)'])) if data["Gender"].iloc[i]=="Female"]]
group_labels = ['Male','Female'] # name of the dataset

fig5 = ff.create_distplot(hist_data2, group_labels)
fig5.update_layout(title_text='Male vs Female Annual Income distribution')
st.write("**Male vs Female Annual Income distribution:**")
st.plotly_chart(fig5)
fig8 = go.Figure(data=[
               go.Violin(x=data['Gender'][data['Gender'] == "Male"],
                             y=data['Annual Income (k$)'][data['Gender'] == "Male"],
                             name="Male",
                             box_visible=True,
                             meanline_visible=True),
              go.Violin(x=data['Gender'][data['Gender'] == "Female"],
                             y=data['Annual Income (k$)'][data['Gender'] == "Female"],
                             name="Female",
                             box_visible=True,
                             meanline_visible=True,
                        line_color='red')
                               
])
fig8.update_layout(title_text='Male vs Female Annual Income')
st.write("**Violin plot of Male vs Female Annual Income distribution**")
st.plotly_chart(fig8)
st.write("** Spending Score distribution:** ")
image5 = Image.open('spendsdist.png')
st.image(image5, caption='')
hist_data3 = [[data['Spending Score (1-100)'].iloc[i]  for i in range(len(data['Spending Score (1-100)'])) if data["Gender"].iloc[i]=="Male"],[data['Spending Score (1-100)'].iloc[i]  for i in range(len(data['Spending Score (1-100)'])) if data["Gender"].iloc[i]=="Female"]]
group_labels = ['Male','Female'] # name of the dataset

fig6 = ff.create_distplot(hist_data3, group_labels)
fig6.update_layout(title_text='Male vs Female Spending Score distribution')
st.write("**Male vs Female Spending Score distribution:** ")
st.plotly_chart(fig6)
st.write("**Violin plot of Male vs Female Spending Score **")
fig9 = go.Figure(data=[
               go.Violin(x=data['Gender'][data['Gender'] == "Male"],
                             y=data['Spending Score (1-100)'][data['Gender'] == "Male"],
                             name="Male",
                             box_visible=True,
                             meanline_visible=True,
                         line_color="royalblue"),
              go.Violin(x=data['Gender'][data['Gender'] == "Female"],
                             y=data['Spending Score (1-100)'][data['Gender'] == "Female"],
                             name="Female",
                             box_visible=True,
                             meanline_visible=True,
                        line_color='yellowgreen')
                               
])
fig9.update_layout(title_text='Male vs Female spending score')
st.plotly_chart(fig9)
st.write("** Male vs Female Spending (Boxen Plot):** ")
image8 = Image.open('boxenp.png')
st.image(image8, caption='')
st.write("** From the above bivariate analysis it can be seen that most female customers spending scores are \
    slightly higher than the male customers even the annul income trend is opposite i.e. most male customers \
        income range is slightly higher than female customers. Now if we closely look at the barplot below. ** ")

fig112 = go.Figure(data=[
    go.Bar(name='Male vs Female average spending', x=sex, y=data["Spending Score (1-100)"].groupby(by=data['Gender'],axis=0).mean()),
    go.Bar(name='Male vs Female average income', x=sex, y=data["Annual Income (k$)"].groupby(by=data['Gender'],axis=0).mean()),
    go.Bar(name='Male vs Female max spending score', x=sex, y=data["Spending Score (1-100)"].groupby(by=data['Gender'],axis=0).max())
])

fig112.update_layout(barmode='group')
st.plotly_chart(fig112)
st.write("**We can see that the average spending score and maximum spending score are slightly high for the male customers. \
    However, the average income of the female customers are slightly high.**")
st.write("**Recommendations:** \n We can use this clustering in multiple way. Using the K-Means \
    algorithm we've found 6 cluster. We can see a a good number of people from age group 20-40 having good spending\
        score. So more focus can be helpful to attact them more. On the other hand, overall sales can be increased by \
            focusing on the people of age around 40-60. More analysis needs to be performed for low income customers in \
                cluster 4 to increase the overall sale. It is also observed that the spending score is slighly higher \
                    for most of the female customers. So, more attention for them can be helpful to increase the overall sales. ")
x = data.loc[:, ['Age','Annual Income (k$)','Spending Score (1-100)']].values
dbscan=DBSCAN(eps=14,min_samples=6)
y_means=dbscan.fit_predict(x)

layout = go.Layout(
    title = 'DBSCAN Clustering using Age, Spending Score and Annual income',
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    ),
    scene = dict(
            xaxis = dict(title  = 'Age'),
            yaxis = dict(title  = 'Annual Income'),
            zaxis = dict(title  = 'Spending Score')
        )
)

fig = go.Figure([go.Scatter3d(
    x= x[y_means == -1, 0],
    y= x[y_means == -1, 1],
    z= x[y_means == -1, 2],
    mode='markers',
    name='Cluster_1',
     marker=dict(
        color = ["red"]*len(x[y_means == -1, 0]), 
        size= 10,
        line=dict(
            color= ["red"]*len(x[y_means == -1, 0]),
            width= 12
        ),
        
        opacity=0.8
     )
),
go.Scatter3d(
    x= x[y_means == 0, 0],
    y= x[y_means == 0, 1],#data['Spending Score (1-100)'],
    z= x[y_means == 0, 2],
    mode='markers',
    name='Cluster_2',
     marker=dict(
        color = ["blue"]*len(x[y_means == 0, 0]), 
        size= 10,
        line=dict(
            color=["blue"]*len(x[y_means == 0, 0]),
            width= 12
        ),
        
        opacity=0.8
     )
),
go.Scatter3d(
    x= x[y_means == 1, 0],
    y= x[y_means == 1, 1],#data['Spending Score (1-100)'],
    z= x[y_means == 1, 2],
    mode='markers',
    name='Cluster_3',
     marker=dict(
        color = ["yellow"]*len(x[y_means == 1, 0]), 
        size= 10,
        line=dict(
            color= ["yellow"]*len(x[y_means == 1, 0]),
            width= 12
        ),
        
        opacity=0.8
     )
),
go.Scatter3d(
    x= x[y_means == 2, 0],
    y= x[y_means == 2, 1],#data['Spending Score (1-100)'],
    z= x[y_means == 2, 2],
    mode='markers',
    name='Cluster_4',
     marker=dict(
        color = ["orange"]*len(x[y_means == 2, 0]), 
        size= 10,
        line=dict(
            color= ["orange"]*len(x[y_means == 2, 0]),
            width= 12
        ),
        
        opacity=0.8
     )
),
go.Scatter3d(
    x= x[y_means == 3, 0],
    y= x[y_means == 3, 1],#data['Spending Score (1-100)'],
    z= x[y_means == 3, 2],
    mode='markers',
    name='Cluster_5',
     marker=dict(
        color = ["cyan"]*len(x[y_means == 3, 0]), 
        size= 10,
        line=dict(
            color= ["cyan"]*len(x[y_means == 3, 0]),
            width= 12
        ),
        
        opacity=0.8
     )
),
# go.Scatter3d(
#    x= x[y_means == 4, 0],
#     y= x[y_means == 4, 1],#data['Spending Score (1-100)'],
#     z= x[y_means == 4, 2],
#     mode='markers',
#     name="Cluster_6",
#      marker=dict(
#         color =["violet"]*len(x[y_means == 4, 0]), 
#         size= 10,
#         line=dict(
#             color= ["violet"]*len(x[y_means == 4, 0]),
#             width= 1
#         ),
        
#         opacity=0.8
#      )
# ),
# go.Scatter3d(
#    x= x[y_means == 5, 0],
#     y= x[y_means == 5, 1],#data['Spending Score (1-100)'],
#     z= x[y_means == 5, 2],
#     mode='markers',
#     name="Cluster_7",
#      marker=dict(
#         color = [], 
#         size= 10,
#         line=dict(
#             color=[],
#             width= 1
#         ),
        
#         opacity=0.8
#      ))
], layout = layout)

fig.update_layout(scene = dict(
                    xaxis = dict(
                         backgroundcolor="rgb(200, 200, 230)",
                         gridcolor="white",
                         showbackground=True,
                         zerolinecolor="white",),
                    yaxis = dict(
                        backgroundcolor="rgb(230, 200,230)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white"),
                    zaxis = dict(
                        backgroundcolor="rgb(230, 230,200)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",),),
                    width=900,
                    height=700,
                    margin=dict(
                    r=10, l=10,
                    b=10, t=40)
                  )
st.markdown("<h2 style='text-align: center; color: red;'><b>3-Dimensional DBSCAN Clustering Analysis</b></h2>", unsafe_allow_html=True)
st.plotly_chart(fig)
st.write("**Summary of clustering using DBSCAN algorithm: **")
st.write("1. Customers can be divided into 5 different clusters with some ambiguity since data is not \
     densely separated.  \n 2. Cluster 2 includes all aged  \
    customers with  midium income and midium to high spending score. \
    \n 3. Cluster 3 consists of elderly customers with low income and low spending score.\
    \n 4. Cluster 4 represents the younger customers with midium to relatively high income and high spending score where \
        cluster 5 does hold opposite for the same income group represented by cluster 4. \
    ")
st.write("For the epsilon value we observed the figure below.")
silhouette = []
l=[i for i in range (6,16)]
for i in l:
  dbscan=DBSCAN(eps=i,min_samples=6)
  dbscan.fit(x)
  score = silhouette_score(x, dbscan.labels_)
  silhouette.append(score)
figl = go.Figure()
figl.add_trace(go.Scatter(x=l, y=silhouette,
                    mode='lines+markers',
                    name='Silhouette Score analysis for different epsilon in DBSCAN clustering'))
st.write("**Silhouette Score analysis for different epsilon in DBSCAN clustering**")
st.plotly_chart(figl)
st.write("We used **epsilon=14** since we've got high Silhouette Score for this value and finaly got 5 clusters using \
    DBSCAN clustering algorithm. ")


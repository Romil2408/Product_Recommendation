import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile
import urllib.request
import streamlit as st
from sklearn.metrics import pairwise_distances

st.set_option('deprecation.showfileUploaderEncoding', False)

fashion_df = pd.read_csv("data/fashion.csv")
boys_extracted_features = np.load('Boys_ResNet_features.npy')
boys_Productids = np.load('Boys_ResNet_feature_product_ids.npy')
girls_extracted_features = np.load('Girls_ResNet_features.npy')
girls_Productids = np.load('Girls_ResNet_feature_product_ids.npy')
men_extracted_features = np.load('Men_ResNet_features.npy')
men_Productids = np.load('Men_ResNet_feature_product_ids.npy')
women_extracted_features = np.load('Women_ResNet_features.npy')
women_Productids = np.load('Women_ResNet_feature_product_ids.npy')
fashion_df["ProductId"] = fashion_df["ProductId"].astype(str)

def get_categories_by_gender(gender):
    categories = list(fashion_df[fashion_df['Gender'] == gender]['SubCategory'].unique())
    return categories

def get_product_titles_by_subcategory(subcategory):
    product_titles = list(fashion_df[fashion_df['SubCategory'] == subcategory]['ProductTitle'].unique())
    return product_titles

def get_product_id_by_title(product_title):
    product_id = fashion_df[fashion_df['ProductTitle'] == product_title]['ProductId'].values[0]
    return product_id

def get_product_ids_by_gender(gender):
    if gender == "Boys":
        return boys_Productids
    elif gender == "Girls":
        return girls_Productids
    elif gender == "Men":
        return men_Productids
    elif gender == "Women":
        return women_Productids

def get_similar_products_cnn(product_title, num_results):
    product_id = get_product_id_by_title(product_title)
    gender = fashion_df[fashion_df['ProductId'] == product_id]['Gender'].values[0]
    extracted_features = None
    Productids = None
    if gender == "Boys":
        extracted_features = boys_extracted_features
        Productids = boys_Productids
    elif gender == "Girls":
        extracted_features = girls_extracted_features
        Productids = girls_Productids
    elif gender == "Men":
        extracted_features = men_extracted_features
        Productids = men_Productids
    elif gender == "Women":
        extracted_features = women_extracted_features
        Productids = women_Productids

    Productids = list(Productids)
    doc_id = Productids.index(product_id)
    pairwise_dist = pairwise_distances(extracted_features, extracted_features[doc_id].reshape(1,-1))
    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    pdists  = np.sort(pairwise_dist.flatten())[0:num_results]
    st.write("""
         #### Input Item details
         """)
    ip_row = fashion_df[['ImageURL','ProductTitle']].loc[fashion_df['ProductId']==Productids[indices[0]]]
    for indx, row in ip_row.iterrows():
        image = Image.open(urllib.request.urlopen(row['ImageURL']))
        image = image.resize((224,224))
        st.image(image)
        st.write(f"Product Title: {row['ProductTitle']}")
    st.write("""
         #### Top Recommended items
         """)
    for i in range(1,len(indices)):
        rows = fashion_df[['ImageURL','ProductTitle']].loc[fashion_df['ProductId']==Productids[indices[i]]]
        for indx, row in rows.iterrows():
            image = Image.open(urllib.request.urlopen(row['ImageURL']))
            image = image.resize((224, 224))
            st.image(image)
            st.write(f"Product Title: {row['ProductTitle']}")
            st.write("-------------------------------------------------------")



st.title("Fashion Recommendation System")

st.sidebar.title("Filters")
gender = st.sidebar.selectbox("Select Gender", ["Boys", "Girls", "Men", "Women"])
category = st.sidebar.selectbox("Select Category", get_categories_by_gender(gender))
product_title = st.sidebar.selectbox("Select Product", get_product_titles_by_subcategory(category))
btn = st.sidebar.button('Show Recommendation')

st.write(f"## {category} Recommendations for {gender}")
st.write("-------------------------------------------------------")
if btn:
    get_similar_products_cnn(product_title, 6)

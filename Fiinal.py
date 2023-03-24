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


def get_product_ids_by_gender(gender):
    if gender == "Boys":
        return boys_Productids
    elif gender == "Girls":
        return girls_Productids
    elif gender == "Men":
        return men_Productids
    elif gender == "Women":
        return women_Productids


def get_similar_products_cnn(product_id, num_results):
    product_id = str(product_id)
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
         #### input item details
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
            image = image.resize((224,224))
            st.image(image)
            st.write(f"Product Title: {row['ProductTitle']}")
            st.write(f"Euclidean Distance from input image: {pdists[i]}")


st.write("""
         ## Visual Similarity based Recommendation
         """
         )

gender = st.selectbox("Select gender", ["Boys", "Girls", "Men", "Women"])
Productids = get_product_ids_by_gender(gender)
product_id = st.selectbox(
"Select a product",
options=Productids
)

if st.button('Show Recommendation'):
    get_similar_products_cnn(product_id, 6)

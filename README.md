# **Somm**


Som·me·lier /ˌsəməlˈyā/: a wine steward 


https://github.com/VishyFishy02/Somm_Wine_Reccomendation/assets/57776597/a1d6fdde-168e-4b31-a7cd-3855976d4251

In this project we built a wine recommendation engine to help users search for wines that fit their tastes.

## Deliverables
When the user enters a free-form search phrase that describes wine flavors or characteristics, Somm will return two outputs:

1. Top 5 wines that best match the query
2. Top 3 styles (e.g. Red Blend from Italy) whose flavor profile are closest to the query

## Who does this project benefit?

- Wine buyers:
    * [Wine is very confusing - huge variety + lack of consistent categorization](https://www.vox.com/the-goods/2020/3/4/21152752/understanding-wine-complicated-learning-education)
    * Wine distribution is fragmented → not easy to find the same wine at another store → suggests a wine style

- Wine sellers: 
    * Use Somm as a recommender for an online store
    * Use Somm as a procurement tool that fit consumers’ demand

While other wine recommendation apps (Vivino, Delectable) do so based on wine information such as price, region, type of wine etc., the innovation of Somm is that it uses reviews of wine flavors to find the wine that best matches a user's query.


## Table of Contents
- [Data and EDA](#data-and-eda)
- [Modelling](#modelling)
- [Installation and Usage](#installation-and-usage)


### Data and EDA
We obtained the [dataset](https://www.kaggle.com/datasets/zynicide/wine-reviews) from Kaggle, it contains 130k rows of wine reviews, together with information on price, region, grape variety etc. The data was scraped from [WineEnthusiast](https://wineenthusiast.com/?s=&drink_type=wine) during the week of June 15th, 2017.

- We cleaned the data and preprocessed to have a reliable dataset, while reducing imbalance between wine styles (important for classification models)
- Main tools: pandas, matplotlib, seaborn, NLP tools (spaCy)

### Modelling

We start with converting unstructured text data in wine reviews to high dimensional vectors. These text embeddings capture the semantic meaning of each review in a vector.

Then we consider two main modelling approaches:

- **Classification**: 
    - Use various classification algorithms to classify wines by styles based on text embeddings of wine reviews, then filter for 5 best-fit wines among 3 most likely styles.
 
- **Retrieval Augmented Generation (RAG) pipline**: 
    - Combine similarity search, reranking, and LLM (with memory) to return 5 best-fit wines. 
    - Top 3 most popular styles are also obtained from the reranked wines.

Comparing these two approaches against a baseline KNN search model, RAG is the best performer for both deliverables!


|Model| Top 5 Wine Recs<br>Mean Relevance<br>(1 to 5) | Top 3 Style Recs<br>Mean Similarity<br>(-1 to 1)|
|---|:---:|:---:|
|Baseline KNN Search| 4.49 | 0.8274 |
|Classification (XGBoost) | 4.34 | 0.8261 |
|RAG| 4.84 | 0.8269 |


- Main tools: OpenAI Text Embedding Model (ada-002), SciKit-Learn, XGBoost, LangChain, Cohere Rerank, Pinecone, LLM (OpenAI gpt-4-1106-preview).


### Installation and Usage

This project was developed during our Data Science Boot Camp Fall 2023 at the Erdos Institute.














# Book Recommendation System 

## Introduction
This is a book recommendation system based on the book rating data from [GoodReads_100k](https://www.kaggle.com/datasets/mdhamani/goodreads-books-100k/) dataset. The dataset contains 100k book. 

## Data Cleaning & Model Training

[```recommendation_data_cleaning.ipynb```](https://www.kaggle.com/code/sanjeevkumar78/book-recommendation/notebook) is used to clean the data. The data is cleaned by removing the books with less than 50 ratings and users with less than 50 ratings. After running ```.ipynb``` file, It works TF-IDF Vectorizer and Cosine Similarity to find the similarity between books. The model is saved as ```cosine_sim_desc.pkl``` in model folder and ```final_data.csv``` also in model folder it contains the data after cleaning (25151 Books).

## Web App
```app.py``` is used to run the web app. The web app is created using Streamlit. 

## Technologies Used
1. Python
2. Pandas
3. Numpy
4. Scikit-learn
5. Streamlit

## How to run the app
1. Clone the repository
2. Install the requirements using ```pip install -r requirements.txt```
3. Download the dataset from [GoodReads_100k](https://www.kaggle.com/datasets/mdhamani/goodreads-books-100k/) and place it in the dataset folder.
4. Run ```recommendation_data_cleaning.ipynb``` to clean the data and train the model.
5. Run ```app.py``` using ```streamlit run app.py``` It will open the web app in the browser.




## References
1. [GoodReads_100k](https://www.kaggle.com/datasets/mdhamani/goodreads-books-100k/)
2. [Streamlit](https://www.streamlit.io/)
3. [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
4. [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
5. [Scikit-learn](https://scikit-learn.org/stable/)
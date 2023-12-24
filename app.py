import streamlit as st
import pickle,csv
import pandas as pd

# Set the app title and favicon
st.set_page_config(page_title='Book Recommendation System', page_icon='📚', layout='wide')

# Function to load the pickled model


@st.cache_resource()
def load_models():
    cosine_sim_desc = pickle.load(open('model/cosine_sim_desc.pkl', 'rb'))
    final_data = pd.read_csv('model/final_data.csv')
    # final_data = pickle.load(open('model/final_data.pkl', 'rb'))
    st.success('Models loaded successfully!')
    return cosine_sim_desc, final_data

cosine_sim_desc, final_data = load_models()


# Get the list of book titles from the final_data DataFrame using pandas
options = final_data['Title'].values.tolist()

# print(options[:5]) # Output check


# Create the Streamlit app
def main():

    # Set the app title
    st.title('Book Recommendation System')

    # Add a dropdown to the main content
    selected_option = st.selectbox('Select an option', pd.Series(options).sort_values().unique())
    # Display the selected option
    st.write('You selected:', selected_option)

    def get_recommendations(book_title, cosine_sim_desc):

        # Check if the final_data DataFrame is empty
        if not final_data.empty:
            # Get the index of the book title
            idx = final_data[final_data['Title'] == book_title].index 
            # print(f"idx: {idx}") output check
            if len(idx) > 0:
                idx = idx[0]
                sim_scores = list(enumerate(cosine_sim_desc[idx]))

                # print(f"sim_scores: {sim_scores}") # output check
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1:6]

                # print(f"sim_scores top 10: {sim_scores}") # output check
                book_indices = [i[0] for i in sim_scores]

                # print(f"book_indices: {book_indices}") # output check

                return final_data[['Title', 'Image', 'Author','Pages']].iloc[book_indices] # return book title, image and author
            else:
                return "Book not found"
        else:
            return "No data available"
        
    # Display book recommendations
    st.subheader('Recommended Books')

    # Display book with image fetch from image url
    book = get_recommendations(selected_option, cosine_sim_desc)
    # align books in a row
    col1, col2, col3, col4, col5 = st.columns(5,gap='large')

    with col1:
        st.image(book.iloc[0, 1],caption=book.iloc[0,0], width=150)
        # st.write(book.iloc[0, 0])
        st.write(book.iloc[0, 2])
        st.write("Pages: ",book.iloc[0, 3])
    
    with col2:
        st.image(book.iloc[1, 1],caption=book.iloc[1,0], width=150)
        # st.write(book.iloc[1, 0])
        st.write(book.iloc[1, 2])
        st.write("Pages: ",book.iloc[1, 3])
    
    with col3:
        st.image(book.iloc[2, 1], caption=book.iloc[2,0],width=150)
        # st.write(book.iloc[2, 0])
        st.write(book.iloc[2, 2])
        st.write("Pages: ",book.iloc[2, 3])
    
    with col4:
        st.image(book.iloc[3, 1],caption=book.iloc[3,0], width=150)
        # st.write(book.iloc[3, 0])
        st.write(book.iloc[3, 2])
        st.write("Pages: ",book.iloc[3, 3])
    
    with col5:
        st.image(book.iloc[4, 1], caption=book.iloc[4,0],width=150)
        # st.write(book.iloc[4, 0])
        st.write(book.iloc[4, 2])
        st.write("Pages: ", book.iloc[4, 3])

    # Books Recommended in a column
    # for i in range(5):
    #     st.image(book.iloc[i, 1], width=150)
    #     st.write(book.iloc[i, 0])
    #     st.write(book.iloc[i, 2])
    #     st.write(book.iloc[i, 3])
    #     st.write('______')
if __name__ == '__main__':
    main()

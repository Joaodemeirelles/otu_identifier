
import pickle
import streamlit as st
from otu_identifier.encoders.dna_encoder import KmerEncoder
# loading the trained model
pickle_in = open('../catboost.pkl', 'rb')
classifier = pickle.load(pickle_in)

@st.cache()

# defining the function which will make the prediction using the data which the user inputs
def prediction(seq):
    seq_encoded = KmerEncoder(4).obtain_kmer_feature_for_one_sequence(
                        seq,
                        write_number_of_occurrences=False
                        )
    # Making predictions
    prediction = classifier.predict(seq_encoded)

    return prediction


# this is the main function in which we define our webpage
def main():
    # front end elements of the web page
    #html_temp = """
    #<div style ="background-color:red;padding:13px">
    #<h1 style ="color:black;text-align:center;">16S PREDICTOR</h1>
    #</div>
    #"""

    html_temp = """
    <div style ="padding:13px">
    <h1 style ="text-align:center;">16S PREDICTOR</h1>
    </div>
    """

    dna_fig = """
    <p align="center">
    <img src="https://media.giphy.com/media/3o7TKz2eMXx7dn95FS/giphy.gif" alt="animated" />
    </p>
    """

    # display the front end aspect
    st.markdown(dna_fig, unsafe_allow_html=True)
    st.markdown(html_temp, unsafe_allow_html=True)

    # following lines create boxes in which user can enter data required to make prediction
    seq = st.text_input('16S sequence',"")
    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        result = prediction(seq)
        st.success('Your 16S sequence is from {}'.format(result))

if __name__=='__main__':
    main()

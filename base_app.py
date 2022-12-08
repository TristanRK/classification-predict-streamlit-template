"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os


from PIL import Image
# Loading Image using PIL
im = Image.open('content/image.jpg')
ima = Image.open('content/hashtags.png')
# Adding Image to web app
st.set_page_config(page_title="base_app.py", page_icon = im)

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

# Data dependencies
import pandas as pd

st.image(
	"https://th.bing.com/th/id/R.ecbb577764245551f2eb3d68db207808?rik=7z5rKX4dMPIj0g&riu=http%3a%2f%2fworld.350.org%2fnz%2ffiles%2f2014%2f01%2fTwitter-350.jpg&ehk=JjB3BnuqD6vhFaspJVVLSwEKtbPCPc3rwUfywG77Rp4%3d&risl=&pid=ImgRaw&r=0",
	width = 650,
)
# Vectorizer
news_vectorizer = open("resources/Vectoriser.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")


# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("EnviroData")
	st.subheader("Climate change tweet classification")


	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = [ "Information", "Models"]
	
	selection = st.sidebar.selectbox("Choose Option", options)
	

	

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and sentiment labels")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

		st.subheader("Hashtags")

		st.image(ima,width = 500)


	if selection == "Models":
		st.subheader("Model Selection")
		selectmodel = st.radio(" ", options=["Log regression","SVM","lsvm"], horizontal=True)

		# Building out the predication page
		if selectmodel == "Log regression":
			st.info("Prediction with ML Models")
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Logreg.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				predicted = []
				if prediction == 1:
					predicted = "pro"
				if prediction == -1:
					predicted = "anti"
				if prediction == 0:
					predicted = "neutral"
				if prediction == 2:
					predicted = "news"

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				st.success("Text Categorized as: {}".format(predicted)) 

		if selectmodel == "SVM":
			st.info("Prediction with ML Models")
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/svm.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				predicted = []
				if prediction == 1:
					predicted = "pro"
				if prediction == -1:
					predicted = "anti"
				if prediction == 0:
					predicted = "neutral"
				if prediction == 2:
					predicted = "news"

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				st.success("Text Categorized as: {}".format(predicted)) 

		if selectmodel == "lsvm":
			st.info("Prediction with ML Models")
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/lsvm.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				predicted = []
				if prediction == 1:
					predicted = "pro"
				if prediction == -1:
					predicted = "anti"
				if prediction == 0:
					predicted = "neutral"
				if prediction == 2:
					predicted = "news"
			

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				st.success("Text Categorized as: {}".format(predicted)) 

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()

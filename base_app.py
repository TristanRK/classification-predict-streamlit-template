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
im = Image.open('content/Logo2.png')
ima = Image.open('content/hashtags.png')
# Adding Image to web app
st.set_page_config(page_title="base_app.py", page_icon = im, layout="wide")

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

# Data dependencies
import pandas as pd


col1, col2, col3 = st.columns(3)
with col1:
	st.write(" ")
with col2:
    st.image('content/Logo2.png')
with col3:
    st.write(" ")

st.markdown("<h1 style='text-align: center; color: black;'>EnviroData</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: left; color: black;'>About</h2>", unsafe_allow_html=True)


st.image(
	"https://th.bing.com/th/id/R.ecbb577764245551f2eb3d68db207808?rik=7z5rKX4dMPIj0g&riu=http%3a%2f%2fworld.350.org%2fnz%2ffiles%2f2014%2f01%2fTwitter-350.jpg&ehk=JjB3BnuqD6vhFaspJVVLSwEKtbPCPc3rwUfywG77Rp4%3d&risl=&pid=ImgRaw&r=0",
	width = 800,
)
# Vectorizer
news_vectorizer = open("resources/Vectoriser.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")
f1 = pd.read_csv("resources/f1.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages

	#######st.title("EnviroData")
	#######st.subheader("Climate change tweet classification")


	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["About", "Project Information", "Models" ]
	
	selection = st.sidebar.selectbox("Choose Option", options)
	

	

	# Building out the "Information" page
	if selection == "Project Information":
		st.info("Many companies are built around lessening one’s environmental impact or carbon footprint. They offer products and services that are environmentally friendly and sustainable, in line with their values and ideals. They would like to determine how people perceive climate change and whether or not they believe it is a real threat. This would add to their market research efforts in gauging how their product/service may be received. With this context, EDSA throwing a challeng with the task of creating a Machine Learning model that is able to classify whether or not a person believes in climate change, based on their novel tweet data. Providing an accurate and robust solution to this task gives companies access to a broad base of consumer sentiment, spanning multiple demographic and geographic categories - thus increasing their insights and informing future marketing strategies")
		# You can read a markdown file from supporting resources folder
		st.subheader("Where is our data from?")
		st.markdown("The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo. The dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43,943 tweets were collected. Each tweet is labelled as one of 4 classes, which are described below.")
		
		st.subheader("Classes")
		st.markdown("2 News: the tweet links to factual news about climate change")
		st.markdown("1 Pro: the tweet supports the belief of man-made climate change")
		st.markdown("0 Neutral: the tweet neither supports nor refutes the belief of man-made climate change")
		st.markdown("-1 Anti: the tweet does not believe in man-made climate change Variable definitions")

		st.subheader("Features")
		st.markdown("sentiment: Which class a tweet belongs in (refer to Class Description above)")
		st.markdown("message: Tweet body")
		st.markdown("tweetid: Twitter unique id")


		st.subheader("Raw data with sentiment labels")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

		st.subheader("Hashtags")
		
		#st.image("content/hashtags2.png",width = 800)
		st.image("content/hashtag.jpg")

	# Building out the "Information" page
	if selection == "About":
		st.info("EnviroData is an environmentally focused data science consultancy founded in 2022 by four innovative individuals. These four individuals are Farayi, Solomon, Tristan and Peakanyo. As a data science consultancy, we aim to provide insights of what makes the world tick and more importantly how environmental issues are dealt with in our modern society.")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Amazing People")

		st.markdown("Solomon Nwokoro")
		st.image("content/Solomon.jpg", width = 300,caption="Data Scientist")

		st.markdown("Peakanyo Mokone")
		st.image("content/Peakanyo.jpg", width = 300, caption = "Data Analyst")

		st.markdown("Farayi Myambo")
		st.image("content/Farayi.jpg", width = 300, caption= "Business Analyst")

		st.markdown("Tristan Krafft")
		st.image("content/Tristan.png", width = 300, caption="ML Engineer")



	if selection == "Models":
		st.subheader("Model Selection")
		selectmodel = st.radio(" ", options=["Logistic Regression","SVC","Linear SVC","RFC","SMOTE Linear SVC", "SMOTE Logistic Regression"], horizontal=True)

		# Building out the predication page
		if selectmodel == "Logistic Regression":
			st.info("Prediction with ML Models")
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/LRmodel.pkl"),"rb"))
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

		if selectmodel == "SVC":
			st.info("Prediction with ML Models")
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/SVCmodel.pkl"),"rb"))
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

		if selectmodel == "Linear SVC":
			st.info("Prediction with ML Models")
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/LinearSVC.pkl"),"rb"))
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


		if selectmodel == "RFC":
			st.info("Prediction with ML Models")
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/RFCmodel.pkl"),"rb"))
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

		if selectmodel == "SMOTE Linear SVC":
			st.info("Prediction with ML Models")
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/lsvc_smote.pkl"),"rb"))
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

		if selectmodel == "SMOTE Logistic Regression":
			st.info("Prediction with ML Models")
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/LR_smote.pkl"),"rb"))
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
		
		st.subheader("F1 scores")
		st.write(f1) # will write the df to the page
		st.write(" ")

		st.bar_chart(f1,x="Models", y="F1_score", width=400)


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()

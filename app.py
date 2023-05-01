import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import slack
from dotenv import load_dotenv
from flask import Flask
from nltk import stem
from nltk import tokenize
from sklearn.metrics.pairwise import cosine_similarity
from slackeventsapi import SlackEventAdapter

load_dotenv(Path() / ".env")

def load_model():
	model_path = Path() / "model"

	urls_file = model_path / "urls.csv"
	corpus_file = model_path / "corpus.csv"
	model_file = model_path / "model.pkl"

	urls = pd.read_csv(urls_file)
	corpus = pd.read_csv(corpus_file).set_index("url index")
	with open(model_file, "rb") as f:
		vectorizer, doc_term_matrix = pickle.load(f)

	def build_query_handler(urls, corpus, vectorizer, doc_term_matrix):

		def preprocess_query(query):
			query = re.sub(r"\w*\d\w*", "", query)
			query = re.sub(r"[^a-zA-Z ]", "", query)
			ps = stem.PorterStemmer()
			return " ".join(ps.stem(token) for token in tokenize.word_tokenize(query))

		def output_query_result(query, doc_indices):
			top_documents = corpus.iloc[doc_indices]

			# Find the url indices corresponding to the top documents.
			top_urls = urls.iloc[top_documents.index]

			output = [f"Showing top results for {query=}"]
			for i, (_, row) in enumerate(top_urls.iterrows(), start=1):
				output.append(f"{i}.")
				for entry in row:
					if isinstance(entry, str):
						output.append(entry)
			return "\n".join(output)

		def handle_query(query, k=5):
			query_vec = vectorizer.transform([preprocess_query(query)])

			similarity_vec = cosine_similarity(query_vec, doc_term_matrix).flatten()

			max_k_indices_unordered = np.argpartition(similarity_vec, -k)[-k:]
			max_k_indices = max_k_indices_unordered[np.argsort(similarity_vec[max_k_indices_unordered])]

			return output_query_result(query, max_k_indices)

		return handle_query

	query_handler = build_query_handler(urls, corpus, vectorizer, doc_term_matrix)

	return query_handler


# Bot
def main():
	query_handler = load_model()

	app = Flask(__name__)
	slack_event_adapter = SlackEventAdapter(os.environ["SIGNING_SECRET"], "/slack/events", app)

	client = slack.WebClient(token=os.environ["SLACK_TOKEN"])
	bot_id = client.api_call("auth.test")["user_id"]

	@slack_event_adapter.on("message")
	def message(payload):
		event = payload.get("event", {})
		user_id = event.get("user")

		if user_id == bot_id: return
		
		channel_id = event.get("channel")
		text = event.get("text")
		output = query_handler(text)
		client.chat_postMessage(channel=channel_id, text=output)

	app.run(debug=True)

if __name__ == "__main__":
	main()
    
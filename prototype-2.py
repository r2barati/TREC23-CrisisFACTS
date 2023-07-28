from transformers import pipeline

summarizer_object = pipeline("summarization", model = "facebook/bart-large-cnn")
 
#Here is a list of all the articles text strings
article_texts = []

text = " ".join(article_texts)

new_summary = summarizer_object(text, max_length = 120, min_length = 80)


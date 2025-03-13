# all in one code which actually works!         LOL
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data from .txt file
file_path = "chat_data.txt"  # Update with your actual file path

questions = []
answers = []

with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        if "\t" in line:  # Ensure proper format
            q, a = line.strip().split("\t")
            questions.append(q.lower())  # Store in lowercase
            answers.append(a)

# Convert to DataFrame
df = pd.DataFrame({"Question": questions, "Answer": answers})

# Vectorize the questions
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(df["Question"])

def chatbot_response(user_input):
    user_input = user_input.lower()
    user_vector = vectorizer.transform([user_input])  # Convert input to vector
    
    # Compute cosine similarity
    similarities = cosine_similarity(user_vector, question_vectors)
    
    # Get the best match index
    best_match_idx = similarities.argmax()
    
    # Return the corresponding answer
    return df.iloc[best_match_idx]["Answer"]

# Test the chatbot
while True:
    user_query = input("You: ")
    if user_query.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    response = chatbot_response(user_query)
    print(f"Chatbot: {response}")

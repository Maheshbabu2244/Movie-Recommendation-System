# Movie-Recommendation-System
A movie recommendation system is designed to suggest films to users based on their preferences. This system relies on historical data (in this case, from Netflix) and utilizes content-based filtering to suggest similar movies using natural language processing and machine learning techniques.

1. 📦 Importing the Dataset
The first step in any machine learning project is getting the data:

python
Copy
Edit
import kagglehub
ashfakyeafi_netflix_movies_and_shows_dataset_path = kagglehub.dataset_download('ashfakyeafi/netflix-movies-and-shows-dataset')
print('Data source import complete.')
Here, the kagglehub library is used to fetch the Netflix Movies and Shows Dataset from Kaggle. This dataset typically includes fields like:

title

director

cast

country

release_year

rating

duration

listed_in (genre)

description

2. 🔧 Installing and Importing Libraries
Essential Python libraries are installed and imported:

python
Copy
Edit
!pip3 install -q numpy pandas matplotlib plotly wordcloud scikit-learn
Then we import the libraries for:

Data manipulation: numpy, pandas

Visualization: matplotlib, plotly, wordcloud

ML & NLP: sklearn, TfidfVectorizer, cosine_similarity

Warnings are suppressed for cleaner outputs

3. 📊 Data Preprocessing
Preprocessing involves cleaning and preparing the data before applying any algorithm.

python
Copy
Edit
df = pd.read_csv('netflix_titles.csv')
df.dropna(inplace=True)
Reading the CSV: The dataset is loaded into a pandas DataFrame.

Handling Nulls: Rows with any missing values are dropped for consistency.

Then comes text preprocessing, crucial for any NLP-based model:

python
Copy
Edit
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text
This function:

Converts all characters to lowercase

Removes punctuation marks (commas, periods, etc.)

It’s applied to fields like title, director, cast, listed_in, and description.

4. 🧠 Feature Engineering — Creating Metadata
Metadata is created to capture a movie's essence using relevant fields:

python
Copy
Edit
df['metadata'] = df['title'] + ' ' + df['director'] + ' ' + df['cast'] + ' ' + df['listed_in'] + ' ' + df['description']
df['metadata'] = df['metadata'].apply(clean_text)
This combines key information into one column per movie. Cleaning this combined string helps in vectorization and similarity computation.

5. 🔡 TF-IDF Vectorization
TF-IDF (Term Frequency-Inverse Document Frequency) helps in converting text into meaningful vectors:

python
Copy
Edit
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['metadata'])
This generates a sparse matrix where each row represents a movie, and each column corresponds to a word’s TF-IDF score. Stopwords like “a”, “and”, “the” are removed.

6. 📏 Cosine Similarity
To measure how similar two movies are:

python
Copy
Edit
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
Cosine similarity calculates the angle between two vectors. If two movies share a lot of similar metadata, their cosine similarity will be close to 1 (very similar).

The resulting matrix is symmetric: cosine_sim[i][j] shows the similarity between movie i and movie j.

7. 🎯 Making Recommendations
To fetch the top recommendations:

python
Copy
Edit
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 similar
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]
Here’s what happens:

The movie's index is found.

Similarities with all other movies are fetched.

Sorted in descending order (most similar first).

The top 10 similar titles are returned (excluding itself).

8. 📁 Storing and Saving the Model
python
Copy
Edit
pickle.dump(tfidf_matrix, open("vector.pkl", "wb"))
This saves the TF-IDF matrix so the model can be reused without recomputing.

9. 🎨 Visualization
Visuals like word clouds and bar charts are created using libraries like matplotlib, plotly, and wordcloud. These help in:

Visualizing common genres

Understanding movie distributions

Spotting trends like most frequent actors/directors

Example:

python
Copy
Edit
wordcloud = WordCloud(width = 800, height = 800, background_color ='black').generate(' '.join(df['listed_in']))
plt.imshow(wordcloud)
plt.axis("off")
10. 🧪 Evaluation (Implicit)
While this code doesn’t use standard ML metrics (like precision/recall), evaluation is often subjective for recommendation systems.

Alternatives include:

User feedback

Click-through rate (CTR)

Manual validation (Does it return logically similar movies?)


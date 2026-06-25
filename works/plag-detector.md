# Plagiarism Detector
 
A minimal plagiarism detector against live web sources, that keeps all processing fully local with no data persistence or third-party transmission.
 
**Status:** Working · **Stack:** Python, scikit-learn, NLTK, BeautifulSoup4, duckduckgo-search
 
There are websites that will check your writing for plagiarism. You paste your text in. They check it. They also now have your unpublished dissertation. This detector extracts keywords from your input, searches DuckDuckGo for each one, scrapes the results, and computes cosine similarity between your text and whatever it finds, using TF-IDF vectorization. The top matches are ranked by similarity score.

Input gets preprocessed before comparison — lowercased, punctuation stripped, stopwords removed for comparison accuracy.
```python
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text
```
The top ten keywords by frequency drive the search.
```python
def extract_keywords(text, top_n=10):
    words = text.split()
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in words if w not in stop_words]
    freq = Counter(filtered)
    return [word for word, _ in freq.most_common(top_n)]
```
Each keyword gets a DuckDuckGo search. Returned URLs get scraped with BeautifulSoup and preprocessed the same way. Then TF-IDF vectors are computed for the input and each page, and cosine similarity scores how close they are.
```python
def check_plagiarism(textA, textB):
    tfidfvec = TfidfVectorizer()
    vectors = tfidfvec.fit_transform([textA, textB])
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix[0][1] * 100
```
 
**Limitations:**
Keyword searches on technical text are limited and will surface technical pages. Content behind logins and paywalls won't be scraped.
 
**Source:** [GitHub](https://github.com/Shriya-Tyagi/PlagiarismChecker)
 

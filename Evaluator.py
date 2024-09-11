import os
import arxiv
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load a pre-trained open-source model from Hugging Face
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embedding(text):
    """Generates the embedding for a given text."""
    return model.encode(text)

def calculate_similarity(embedding1, embedding2):
    """Calculates cosine similarity between two embeddings."""
    return cosine_similarity([embedding1], [embedding2])[0][0]

def rank_papers(paper_embedding, paper_embeddings):
    """Ranks papers by similarity to the reference paper."""
    similarities = [calculate_similarity(paper_embedding, emb) for emb in paper_embeddings]
    rankings = np.argsort(similarities)[::-1]  # Sort by descending similarity
    return rankings, similarities

def load_papers_from_directory(directory_path):
    """Loads paper content from text files in a directory."""
    papers = []
    filenames = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):  # Only consider .txt files
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                papers.append(file.read())
                filenames.append(filename)
    return papers, filenames

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def search_arxiv(keyword, max_results=100):
    """Searches arXiv for papers based on a keyword and returns paper metadata."""
    search = arxiv.Search(
        query=keyword,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    papers = []
    filenames = []
    for result in search.results():
        paper_pdf_url = result.pdf_url
        filename = f"{result.entry_id}.pdf"
        download_pdf(paper_pdf_url, filename)
        paper_text = extract_text_from_pdf(filename)
        papers.append(paper_text)
        filenames.append(f"{result.title}")
    return papers, filenames


def search_and_download_arxiv(query, folder_name, max_results=100):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    search = arxiv.Search(
        query=query,
        max_results=10,  # Adjust this as needed
        sort_by=arxiv.SortCriterion.Relevance
    )

    count = 0
    for result in search.results():
        if count >= max_results:
            break
        title = result.title.replace(' ', '_').replace('/', '_')
        arxiv_id = result.entry_id.split('/')[-1]
        pdf_url = result.pdf_url
        paper_filename = os.path.join(folder_name, f"{arxiv_id}_{title}.pdf")
        
        if os.path.exists(paper_filename):
            print(f"Skipping already downloaded paper: {title}")
            continue
        
        response = requests.get(pdf_url)
        with open(paper_filename, 'wb') as f:
            f.write(response.content)
        
        print(f"Downloaded and saved: {title}")
        count += 1


def load_from_pdf(folder_name):
    papers = []
    filenames = []
    for pdf_file in tqdm(os.listdir(folder_name)):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(folder_name, pdf_file)
            paper_id = pdf_file.split(".pdf")[0]
            print(pdf_path)
            try:
                text = extract_text_from_pdf(pdf_path)
            except Exception as e:
                print(e)
            papers.append(text)
            filenames.append(paper_id)
    return papers, filenames

def download_pdf(url, filename):
    """Downloads a PDF from a given URL."""
    response = requests.get(url)
    with open(filename, 'wb') as file:
        file.write(response.content)


def plot_similarities(filenames, similarities):
    """Plots the similarities of papers."""
    plt.figure(figsize=(10, 6))
    plt.barh(filenames, similarities, color='skyblue')
    plt.xlabel('Similarity Score')
    plt.ylabel('Paper')
    plt.title('Similarity Scores of Papers')
    plt.gca().invert_yaxis()  # To display the highest score at the top
    plt.show()

if __name__ == "__main__":

    queries = [
        'anomaly detection',
        'runtime verification hybrid systems',
        'formal anomaly detection',
        'symbolic verification methods',
        'affine arithmetic decision diagrams',
        'constraint solvers in safety systems',
        'formal methods for anomaly detection',
        'predictive maintenance in cyber-physical systems',
        'hybrid systems verification',
        'digital twins for safety-critical systems',
        'affine form decision trees',
        'runtime verification of hybrid automata',
        'formal verification in cyber-physical systems',
        'machine learning for anomaly detection in embedded systems',
        'linear constraint solving in verification',
        'symbolic execution for runtime verification',
        'formal anomaly detection in embedded systems',
        'affine constraints in decision diagrams',
        'runtime monitoring of hybrid systems',
        'runtime verification with affine arithmetic'
    ]

    folder_name = "all_papers"
    
    # Set the path to the directory containing your text files
    directory_path = "papers/"  # Change this to your actual directory path

    for query in queries:
        print(f"Searching for papers related to: {query} on arXiv")
        search_and_download_arxiv(query, folder_name, max_results=10)

    # Load your paper as text
    your_paper_path = "your_paper.txt"  # Path to your paper
    with open(your_paper_path, 'r', encoding='utf-8') as file:
        your_paper = file.read()

    # Load other papers from the directory
    other_papers, filenames = load_papers_from_directory(directory_path)
    print(other_papers[0])
    # Add arXiv papers based on a keyword search
    #keyword = "anomaly detection"  # Change this to your desired search keyword
    #arxiv_papers, arxiv_filenames = search_arxiv(keyword)
    #other_papers.extend(arxiv_papers)
    #filenames.extend(arxiv_filenames)

    arxiv_papers, arxiv_filenames = load_from_pdf(folder_name)
    other_papers.extend(arxiv_papers)
    filenames.extend(arxiv_filenames)
    
    # Generate embedding for your paper
    your_paper_embedding = generate_embedding(your_paper)

    # Generate embeddings for other papers
    other_paper_embeddings = [generate_embedding(paper) for paper in other_papers]

    # Rank papers by their similarity to your paper
    rankings, similarities = rank_papers(your_paper_embedding, other_paper_embeddings)

    # Display the rankings and similarity scores
    for rank, (index, similarity) in enumerate(zip(rankings, similarities), 1):
        print(f"Rank {rank}: Paper '{filenames[index]}' with similarity {similarity:.4f}")
        print(f"File: {filenames[index]}\n")

    plot_similarities([filenames[i] for i in rankings], [similarities[i] for i in rankings])
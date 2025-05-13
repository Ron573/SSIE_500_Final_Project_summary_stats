import nltk
import numpy as np
from scipy.stats import chisquare
from collections import Counter, defaultdict
import heapq
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd  # For CSV output


import nltk
import numpy as np
from scipy.stats import chisquare
from collections import Counter, defaultdict
import heapq
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd  # For CSV output

# ---------------------------------------------------------------------
# INSERT THE EXTRACTION CODE HERE:
# ---------------------------------------------------------------------

def extract_first_n_characters(input_file, output_file, n=50000):
    """Reads a text file, extracts the first n characters, and saves them to a new file."""
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            text = infile.read()
            truncated_text = text[:n]  # Extract the first n characters

        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.write(truncated_text)
        print(f"Successfully extracted and saved {n} characters from {input_file} to {output_file}")
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}")
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

# Define the original file paths and the new file paths
original_files = {
    "English": "Languages/English.txt",  # Replace with your actual file names
    "Spanish": "Languages/Spanish.txt",
    "Russian": "Languages/Russian.txt",
    "Hindi": "Languages/Hindi.txt",
    "Turkish": "Languages/Turkish.txt",
    "Chinese": "Languages/Chinese.txt"
}

new_files = {
    "English": "Languages/English_first50000.txt",
    "Spanish": "Languages/Spanish_first50000.txt",
    "Russian": "Languages/Russian_text_first50000.txt",
    "Hindi": "Languages/hindi_first50000.txt",
    "Turkish": "Languages/turkish_first50000.txt",
    "Chinese": "Languages/chinese_first50000.txt"
}

# Extract and save the first 50,000 characters for each language
for language, original_file in original_files.items():
    extract_first_n_characters(original_file, new_files[language])

# ---------------------------------------------------------------------
# END OF EXTRACTION CODE
# ---------------------------------------------------------------------

# --- 1. Directory Setup ---
def setup_directories():
    """Creates directories for storing output data and visualizations."""
    directories = ["DATA", "IMAGES"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

# ... rest of your code ...

# --- 1. Directory Setup ---
def setup_directories():
    """Creates directories for storing output data and visualizations."""
    directories = ["DATA", "IMAGES"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

# --- 2. Data Loading and Preprocessing ---
def load_data(sources, char_limit=50000):
    """Loads and concatenates data from multiple sources, limiting to the first char_limit characters."""
    text = {}  # Store text for each language
    for source in sources:
        language = source.split(" ")[1]  # Extract language from filename
        try:
            with open(source, 'r', encoding='utf-8') as f:
                file_content = f.read(char_limit)  # Read only the first char_limit characters
                text[language] = file_content
        except FileNotFoundError:
            print(f"Warning: File not found: {source}")
            text[language] = ""  # Assign empty string if file not found
        except Exception as e:
            print(f"Error reading file {source}: {e}")
            text[language] = ""  # Assign empty string if file not found
    return text

def tokenize(text):
    """Tokenizes the text into words."""
    tokens = nltk.word_tokenize(text)
    return tokens

# --- 3. Entropy Calculation ---
def calculate_entropy(data):
    """Calculates the entropy of a sequence."""
    counts = Counter(data)
    probabilities = [count / len(data) for count in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities)
    return entropy

# --- 4. Huffman Coding Simulation ---
class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data):
    """Builds a Huffman tree from the data."""
    counts = Counter(data)
    heap = [Node(char, freq) for char, freq in counts.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = Node(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap, merged)

    return heap[0] if heap else None

def build_huffman_codes(node, code="", codes={}):
    """Traverses the Huffman tree to build character codes."""
    if node is None:
        return

    if node.char is not None:
        codes[node.char] = code
        return

    build_huffman_codes(node.left, code + "0", codes)
    build_huffman_codes(node.right, code + "1", codes)

def calculate_average_code_length(data, codes):
    """Calculates the average Huffman code length."""
    total_length = sum(len(codes[char]) for char in data)
    return total_length / len(data) if data else 0

def calculate_compression_ratio(data, codes):
    """Calculates the compression ratio using Huffman coding."""
    original_bits = len(data) * 8  # Assuming 8 bits per character
    compressed_bits = sum(len(codes[char]) for char in data)
    return compressed_bits / original_bits if original_bits else 0

# --- 5. Third-Order Markov Model ---
def build_markov_model(data, order=3):
    """Builds a Markov model of the specified order."""
    model = defaultdict(Counter)
    for i in range(len(data) - order):
        history = tuple(data[i:i+order])
        next_token = data[i+order]
        model[history][next_token] += 1

    # Normalize counts to probabilities
    for history, counts in model.items():
        total = sum(counts.values())
        for token, count in counts.items():
            counts[token] = count / total

    return model

def markov_preprocess(data, model, order=3):
    """Preprocesses the data using the Markov model."""
    processed_data = []
    for i in range(len(data) - order):
        history = tuple(data[i:i+order])
        if history in model:
            # Replace token with probability distribution
            processed_data.append(model[history])
        else:
            # Use a uniform distribution if history is not in model
            processed_data.append({token: 1/len(set(data)) for token in set(data)})
    return processed_data

def markov_postprocess(chi2, model, data, order=3):
    """Postprocesses the Chi-squared statistic using the Markov model."""
    # This is a placeholder.  A more sophisticated implementation would
    # adjust the expected frequencies based on the Markov model.
    # For example, you could weight the expected frequencies by the
    # probability of the observed sequence according to the model.
    return chi2  # Return the original Chi-squared statistic for now

# --- 6. KL Divergence Calculation ---
def calculate_kl_divergence(p, q):
    """Calculates KL divergence between two probability distributions."""
    # Add a small constant to avoid division by zero
    p = np.array(list(p.values())) + 1e-9  # Handle dict input
    q = np.array(list(q.values())) + 1e-9  # Handle dict input
    kl_divergence = np.sum(p * np.log2(p / q))
    return kl_divergence

# --- 7. Chi-Squared Test ---
def calculate_chi_squared(data):
    """Calculates the Chi-squared statistic."""
    # The input data is now a list of probability distributions
    # We need to aggregate these distributions to get observed frequencies
    observed_counts = Counter()
    for dist in data:
        for token, prob in dist.items():
            observed_counts[token] += prob

    total_prob = sum(observed_counts.values())
    observed_frequencies = np.array(list(observed_counts.values()))

    # Calculate expected frequencies based on a uniform distribution
    expected_frequency = total_prob / len(observed_counts)
    # Filter out categories with zero expected frequency
    valid_indices = observed_frequencies > 0
    observed_frequencies = observed_frequencies[valid_indices]
    expected_frequency = expected_frequency  # Recalculate if needed

    if len(observed_frequencies) < 2:
        print("Warning: Not enough data for Chi-squared test after filtering.")
        return None

    chi2, p = chisquare(observed_frequencies, f_exp=[expected_frequency] * len(observed_frequencies))
    return chi2

# --- 8. Landauer's Principle ---
def calculate_landauer_cost(entropy, temperature=300):
    """Calculates Landauer's thermodynamic cost in Joules."""
    k = 1.38e-23  # Boltzmann constant
    return k * temperature * math.log(2) * entropy

# --- 9. Higher-Order Markov Tests ---
def calculate_trigram_entropy_rate(data):
    """Calculates the trigram entropy rate."""
    trigrams = nltk.ngrams(data, 3)
    trigram_list = list(trigrams)
    return calculate_entropy(trigram_list)

# --- 10. Visualization ---
def visualize_data(entropy, kl_divergence, landauer_cost, language):
    """Visualizes entropy vs. KL divergence vs. Landauer cost in a 3D scatter plot."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(entropy, kl_divergence, landauer_cost, c=landauer_cost, cmap='viridis', marker='o')
    ax.set_xlabel('Entropy')
    ax.set_ylabel('KL Divergence')
    ax.set_zlabel("Landauer Cost (J)")
    ax.set_title(f'Entropy vs. KL Divergence vs. Landauer Cost - {language}')
    plt.savefig(os.path.join("IMAGES", f"{language}_3D_plot.png"))
    plt.close(fig)

# --- 11. Main Execution ---
if __name__ == "__main__":
    # --- Directory Setup ---
    setup_directories()

    # Replace with your actual file paths
    sources = [
        "Languages/English first50000.txt",
        "Languages/Spanish first50000.txt",
        "Languages/Russian text first50000.txt",
        "Languages/Sri Hanuman Chalisa Hindi.pdf",
        "Languages/turkish book chapter.pdf",
        "Languages/chinese dissertation.txt"
    ]

    # --- Data Loading ---
    language_texts = load_data(sources)

    # --- Analysis and Storage ---
    results = {}
    for language, text in language_texts.items():
        if not text:  # Skip if data loading failed
            print(f"Skipping {language} due to data loading failure.")
            continue

        print(f"Analyzing {language}...")
        tokens = tokenize(text)

        # --- Entropy ---
        entropy = calculate_entropy(tokens)
        print(f"  Entropy: {entropy}")

        # --- Huffman Coding ---
        huffman_tree = build_huffman_tree(tokens)
        if huffman_tree:
            huffman_codes = {}
            build_huffman_codes(huffman_tree, codes=huffman_codes)
            avg_code_length = calculate_average_code_length(tokens, huffman_codes)
            compression_ratio = calculate_compression_ratio(tokens, huffman_codes)
            print(f"  Average Huffman Code Length: {avg_code_length}")
            print(f"  Compression Ratio: {compression_ratio}")
        else:
            print("Could not build Huffman tree (empty data?)")
            avg_code_length = 0
            compression_ratio = 0

        # --- Landauer's Principle ---
        landauer_cost = calculate_landauer_cost(entropy)
        print(f"  Landauer Cost: {landauer_cost} J")

        # --- Markov Preprocessing ---
        markov_model = build_markov_model(tokens)
        preprocessed_data = markov_preprocess(tokens, markov_model)

        # --- KL Divergence ---
        uniform_dist = {token: 1 / len(set(tokens)) for token in set(tokens)}
        kl_divergence = 0
        for dist in preprocessed_data:
            kl_divergence += calculate_kl_divergence(dist, uniform_dist)
        kl_divergence /= len(preprocessed_data)
        print(f"  KL Divergence (vs. Uniform): {kl_divergence}")

        # --- Chi-Squared Test ---
        chi_squared = calculate_chi_squared(preprocessed_data)
        if chi_squared is not None:
            postprocessed_chi2 = markov_postprocess(chi_squared, markov_model, tokens)
            print(f"  Chi-Squared Statistic (Postprocessed): {postprocessed_chi2}")
        else:
            print("  Chi-Squared test could not be performed.")
            postprocessed_chi2 = None

        # --- Higher-Order Markov Test ---
        trigram_entropy_rate = calculate_trigram_entropy_rate(tokens)
        print(f" Trigram Entropy Rate: {trigram_entropy_rate}")

        # --- Visualization ---
        visualize_data(entropy, kl_divergence, landauer_cost, language)

        # --- Store Results ---
        results[language] = {
            "Entropy": entropy,
            "AvgCodeLength": avg_code_length,
            "CompressionRatio": compression_ratio,
            "LandauerCost": landauer_cost,
            "KL Divergence": kl_divergence,
            "ChiSquared": postprocessed_chi2,
            "TrigramEntropyRate": trigram_entropy_rate
        }

    # --- Save Results to CSV ---
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv(os.path.join("DATA", "language_analysis_results.csv"))
    print("Analysis complete. Results saved to DATA/language_analysis_results.csv")

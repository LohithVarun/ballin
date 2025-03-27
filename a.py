# Q1: Regular Expressions Demonstration
import re

text = "The quick brown fox jumps over the lazy dog. The rain in Spain falls mainly on the plain."
pattern = r"\b[Ss]pain\b" # Match 'Spain' or 'spain' as whole words

# Search for the first match
match = re.search(pattern, text)
if match:
    print(f"Q1: First match found: '{match.group()}' at index {match.start()}")
else:
    print("Q1: Pattern not found.")

# Find all matches
matches = re.findall(pattern, text)
print(f"Q1: All matches found: {matches}")

# -----------------------------------------------------

# Q2: Basic Finite State Automaton (Ends with 'ab')
def ends_with_ab(input_string):
    state = 0 # Initial state
    for char in input_string:
        if state == 0 and char == 'a':
            state = 1
        elif state == 1 and char == 'b':
            state = 2
        elif state == 1 and char == 'a':
            state = 1 # Stay in state 1 if another 'a'
        else:
            state = 0 # Reset state otherwise (unless already in state 2)

        # Correction: if we see 'a' after 'ab', restart check for 'ab'
        if state == 2 and char == 'a':
             state = 1
        elif state == 2 and char != 'a':
             state = 0 # Not ending in 'ab' anymore, reset unless next is 'a'

    return state == 2

print(f"Q2: 'cab' ends with 'ab'? {ends_with_ab('cab')}")
print(f"Q2: 'abab' ends with 'ab'? {ends_with_ab('abab')}")
print(f"Q2: 'caba' ends with 'ab'? {ends_with_ab('caba')}")
print(f"Q2: 'abc' ends with 'ab'? {ends_with_ab('abc')}")

# -----------------------------------------------------

# Q3: Morphological Analysis using NLTK (Lemmatization Example)
import nltk
# nltk.download('punkt') # Ensure downloaded
# nltk.download('wordnet') # Ensure downloaded
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()
text = "The cats are running towards the dogs"
words = word_tokenize(text)
lemmatized_words = [lemmatizer.lemmatize(word, pos='v') if word == 'running' else lemmatizer.lemmatize(word) for word in words] # Simple POS hint for 'running'

print(f"Q3: Original words: {words}")
print(f"Q3: Lemmatized words: {lemmatized_words}")

# -----------------------------------------------------

# Q4: Finite-State Machine for Morphological Parsing (Simple Pluralization)
def pluralize(noun):
    if noun.endswith(('s', 'x', 'z', 'ch', 'sh')):
        return noun + 'es'
    elif noun.endswith('y') and noun[-2].lower() not in 'aeiou':
        return noun[:-1] + 'ies'
    else:
        return noun + 's'

nouns = ['cat', 'dog', 'bus', 'box', 'fly', 'toy']
plurals = {noun: pluralize(noun) for noun in nouns}
print(f"Q4: Plurals: {plurals}")

# -----------------------------------------------------

# Q5: Porter Stemmer Algorithm
import nltk
# nltk.download('punkt') # Ensure downloaded
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()
words = ["running", "flies", "happily", "computation", "provision"]
stemmed_words = [stemmer.stem(word) for word in words]
print(f"Q5: Original words: {words}")
print(f"Q5: Stemmed words: {stemmed_words}")

# -----------------------------------------------------

# Q6: Basic N-gram Model (Bigram) for Text Generation
import nltk
import random
nltk.download('punkt') 
from nltk.util import ngrams
from collections import defaultdict
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('brown')
nltk.download('universal_tagset')


text = "The quick brown fox jumps over the lazy dog. The dog barked loudly."
tokens = nltk.word_tokenize(text.lower())
bigrams = list(ngrams(tokens, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))

model = defaultdict(lambda: defaultdict(lambda: 0))
for w1, w2 in bigrams:
    model[w1][w2] += 1

# Normalize counts to probabilities (optional for simple generation)
# for w1 in model:
#     total_count = float(sum(model[w1].values()))
#     for w2 in model[w1]:
#         model[w1][w2] /= total_count

# Generate text
current_word = '<s>'
generated_text = []
while current_word != '</s>' and len(generated_text) < 15: # Limit length
    if current_word not in model or not model[current_word]:
        break # Stop if no continuation found
    # Simple weighted choice based on counts
    next_word_candidates = list(model[current_word].keys())
    next_word_counts = list(model[current_word].values())
    next_word = random.choices(next_word_candidates, weights=next_word_counts, k=1)[0]

    if next_word != '</s>':
        generated_text.append(next_word)
    current_word = next_word

print(f"Q6: Sample generated text: {' '.join(generated_text)}")

# -----------------------------------------------------

# Q7: Part-of-Speech Tagging using NLTK
import nltk
# nltk.download('punkt') # Ensure downloaded
# nltk.download('averaged_perceptron_tagger') # Ensure downloaded

text = "NLTK provides powerful tools for NLP tasks."
tokens = nltk.word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)
print(f"Q7: POS Tags: {pos_tags}")

# -----------------------------------------------------

# Q8: Simple Stochastic POS Tagger (Unigram Tagger)
import nltk
# nltk.download('brown') # Ensure downloaded
# nltk.download('universal_tagset') # Ensure downloaded
from nltk.corpus import brown

# Train on a small portion of Brown corpus
brown_tagged_sents = brown.tagged_sents(categories='news', tagset='universal')[:500] # Use 500 sentences for speed
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents, backoff=nltk.DefaultTagger('NOUN')) # Backoff to NOUN

text = "The quick brown fox jumps over the lazy dog ." # Include period
tokens = nltk.word_tokenize(text)
tags = unigram_tagger.tag(tokens)
print(f"Q8: Stochastic (Unigram) POS Tags: {tags}")

# -----------------------------------------------------

# Q9: Rule-Based POS Tagger using Regular Expressions
import nltk

# Define regex patterns for tags
patterns = [
    (r'.*ing$', 'VERB'),         # Gerunds or present participles
    (r'.*ed$', 'VERB'),          # Past tense verbs
    (r'.*es$', 'NOUN'),          # Plural nouns
    (r'.*\'s$', 'NOUN'),         # Possessive nouns
    (r'.*s$', 'NOUN'),           # Plural nouns
    (r'^-?[0-9]+(\.[0-9]+)?$', 'NUM'), # Numbers
    (r'.*', 'NOUN')              # Default to noun
]
regexp_tagger = nltk.RegexpTagger(patterns)

text = "Gaming consoles costing 500 dollars are popular."
tokens = nltk.word_tokenize(text)
tags = regexp_tagger.tag(tokens)
print(f"Q9: Rule-Based (Regexp) POS Tags: {tags}")

# -----------------------------------------------------

# Q10: Transformation-Based Tagging (Applying One Simple Rule)
import nltk

# Simulate TBL: Start with a basic tagger, apply a transformation rule
text = "They refuse to permit us to obtain the refuse permit."
tokens = nltk.word_tokenize(text)

# 1. Initial tagging (e.g., using a simple default or unigram tagger)
default_tagger = nltk.DefaultTagger('NOUN') # Simple start: tag everything as NOUN
initial_tags = default_tagger.tag(tokens)
print(f"Q10: Initial Tags: {initial_tags}")

# 2. Apply a transformation rule: "Change NN to VB if preceding word is 'to'"
transformed_tags = list(initial_tags) # Make a mutable copy
for i in range(1, len(transformed_tags)):
    word, tag = transformed_tags[i]
    prev_word, prev_tag = transformed_tags[i-1]
    if prev_word.lower() == 'to' and tag == 'NOUN':
         # Apply the rule (simplified: doesn't check context beyond 'to')
        transformed_tags[i] = (word, 'VERB')

print(f"Q10: Transformed Tags: {transformed_tags}")

# -----------------------------------------------------

# Q11: Simple Top-Down Parser (Recursive Descent)
import nltk

grammar = nltk.CFG.fromstring("""
  S -> NP VP
  VP -> V NP | V NP PP
  PP -> P NP
  V -> "saw" | "ate" | "walked"
  NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
  Det -> "a" | "an" | "the"
  N -> "man" | "dog" | "cat" | "telescope" | "park"
  P -> "in" | "on" | "by" | "with"
""")

sentence = "John saw a man in the park".split()
# NOTE: RecursiveDescentParser can be inefficient and might not handle all grammars
# It's primarily for demonstration of the top-down concept.
# Use Chart Parsers (like Earley) for more robust parsing.
try:
    rd_parser = nltk.RecursiveDescentParser(grammar)
    print("Q11: Top-Down Parsing Trees:")
    for tree in rd_parser.parse(sentence):
        print(tree)
        tree.pretty_print() # Nicer visualization
except ValueError as e:
    print(f"Q11: Parsing failed. The grammar or sentence might be unsuitable for simple RD parsing. Error: {e}")
except RecursionError:
     print("Q11: Parsing failed due to potential left-recursion or deep recursion.")


# -----------------------------------------------------

# Q12: Earley Parser Implementation
import nltk

# Using the same grammar as Q11
grammar = nltk.CFG.fromstring("""
  S -> NP VP
  VP -> V NP | V NP PP
  PP -> P NP
  V -> "saw" | "ate" | "walked"
  NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
  Det -> "a" | "an" | "the"
  N -> "man" | "dog" | "cat" | "telescope" | "park"
  P -> "in" | "on" | "by" | "with"
""")

sentence = "Mary saw Bob".split()
earley_parser = nltk.EarleyChartParser(grammar)
parses = list(earley_parser.parse(sentence))

print(f"\nQ12: Number of Earley parses found: {len(parses)}")
if parses:
    print("Q12: First Earley parse tree:")
    parses[0].pretty_print()
else:
    print("Q12: No parse found for the sentence.")

# -----------------------------------------------------

# Q13: Generate Parse Tree using CFG
import nltk

# Using the same grammar and parser as Q12
grammar = nltk.CFG.fromstring("""
  S -> NP VP
  VP -> V NP | V NP PP
  PP -> P NP
  V -> "saw" | "ate"
  NP -> "John" | Det N
  Det -> "a" | "the"
  N -> "cat" | "dog"
  P -> "with"
""")

sentence = "John ate the cat".split()
parser = nltk.EarleyChartParser(grammar)
parses = list(parser.parse(sentence))

print("\nQ13: Generating Parse Tree:")
if parses:
    print("Parse Tree:")
    parses[0].pretty_print() # Display the first parse tree found
    # You can also draw the tree if matplotlib is installed
    # parses[0].draw()
else:
    print("No parse tree found for the sentence.")

# -----------------------------------------------------

# Q14: Check Agreement in Sentences (Feature-Based Grammar)
import nltk

# Feature grammar incorporating number agreement (sg/pl)
feature_grammar = nltk.grammar.FeatureGrammar.fromstring("""
    S -> NP[NUM=?n] VP[NUM=?n]
    NP[NUM=?n] -> Det[NUM=?n] N[NUM=?n]
    NP[NUM=sg] -> 'John' | 'Mary'
    NP[NUM=pl] -> 'they'

    VP[NUM=?n] -> V[NUM=?n] NP
    VP[NUM=sg] -> V[NUM=sg]
    VP[NUM=pl] -> V[NUM=pl]

    Det[NUM=sg] -> 'this' | 'a'
    Det[NUM=pl] -> 'these' | 'all'
    N[NUM=sg] -> 'dog' | 'cat'
    N[NUM=pl] -> 'dogs' | 'cats'
    V[NUM=sg] -> 'sees' | 'walks'
    V[NUM=pl] -> 'see' | 'walk'
""")

# Sentences to check
sentences = [
    "John sees a dog",      # Correct (sg)
    "these cats walk",      # Correct (pl)
    "John walk",            # Incorrect (sg + pl)
    "these dog sees"        # Incorrect (pl + sg)
]

parser = nltk.parse.FeatureChartParser(feature_grammar)

print("\nQ14: Checking Agreement:")
for sent_str in sentences:
    tokens = sent_str.split()
    try:
        trees = list(parser.parse(tokens))
        if trees:
            print(f"'{sent_str}' - AGREES")
            # trees[0].pretty_print() # Optionally print tree
        else:
            print(f"'{sent_str}' - DISAGREES (No valid parse)")
    except ValueError as e:
        print(f"'{sent_str}' - Error during parsing: {e}")


# -----------------------------------------------------

# Q15: Probabilistic Context-Free Grammar (PCFG) Parsing
import nltk

# Define a simple PCFG
pcfg_grammar = nltk.PCFG.fromstring("""
    S -> NP VP [1.0]
    NP -> Det N [0.6] | 'John' [0.4]
    VP -> V NP [0.7] | V [0.3]
    Det -> 'the' [0.8] | 'a' [0.2]
    N -> 'man' [0.5] | 'dog' [0.5]
    V -> 'saw' [0.9] | 'liked' [0.1]
""")

# Viterbi parser finds the most probable parse
viterbi_parser = nltk.ViterbiParser(pcfg_grammar)
sentence = "John saw the man".split()

print("\nQ15: PCFG Parsing:")
parses = list(viterbi_parser.parse(sentence))

if parses:
    print("Most likely parse tree:")
    parses[0].pretty_print()
    print(f"Probability: {parses[0].prob()}")
else:
    print("No parse found for the sentence.")

# -----------------------------------------------------

### Q16: Named Entity Recognition (NER) using SpaCy
##import spacy
##
### Load the small English model
### Make sure to download it first: python -m spacy download en_core_web_sm
##try:
##    nlp_spacy = spacy.load("en_core_web_sm")
##except OSError:
##    print("Q16: SpaCy model 'en_core_web_sm' not found. Download it: python -m spacy download en_core_web_sm")
##    exit()
##
##
##text = "Apple Inc. is planning to open a new store in London next year, says Tim Cook."
##doc = nlp_spacy(text)
##
##print("\nQ16: Named Entity Recognition (SpaCy):")
##if not doc.ents:
##     print("No entities found.")
##for ent in doc.ents:
##    print(f"- Entity: {ent.text}, Label: {ent.label_} ({spacy.explain(ent.label_)})")
##

# -----------------------------------------------------

# Q17: Accessing WordNet
import nltk
# nltk.download('wordnet') # Ensure downloaded
from nltk.corpus import wordnet

word = "bank"
print(f"\nQ17: Exploring WordNet for '{word}':")

synsets = wordnet.synsets(word)
if not synsets:
    print(f"No synsets found for '{word}'.")
else:
    print(f"Synsets found: {len(synsets)}")
    for i, syn in enumerate(synsets):
        print(f"  Synset {i+1}: {syn.name()}")
        print(f"    Definition: {syn.definition()}")
        print(f"    Examples: {syn.examples()}")
        # Show hypernyms (more general concepts) for the first synset
        if i == 0:
             print(f"    Hypernyms: {syn.hypernyms()}")

# -----------------------------------------------------

# Q18: Simple FOPC Parser (First-Order Predicate Calculus)
import nltk
from nltk.sem import logic

# NLTK uses 'exists' for existential quantifier and 'all' for universal
expression_str = "exists x. (dog(x) & walks(x))"
try:
    parsed_expr = logic.Expression.fromstring(expression_str)
    print(f"\nQ18: Parsed FOPC expression for '{expression_str}':")
    print(parsed_expr)
    print(f"Type: {type(parsed_expr)}")

    # Example of checking variables
    # print(f"Free variables: {parsed_expr.free()}") # Should be empty
    # print(f"Variables: {parsed_expr.variables()}") # Should contain x
except Exception as e:
    print(f"\nQ18: Error parsing expression '{expression_str}': {e}")


# -----------------------------------------------------

# Q19: Word Sense Disambiguation (Lesk Algorithm)
import nltk
# nltk.download('punkt') # Ensure downloaded
# nltk.download('wordnet') # Ensure downloaded
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize

sentence = "The bank can guarantee deposits will be safe."
ambiguous_word = "bank"
tokens = word_tokenize(sentence)

# Using NLTK's simple Lesk implementation
sense = lesk(tokens, ambiguous_word, pos='n') # Specify part-of-speech if known (n for noun)

print(f"\nQ19: Word Sense Disambiguation for '{ambiguous_word}' in '{sentence}':")
if sense:
    print(f"Chosen Sense: {sense.name()} - {sense.definition()}")
else:
    print(f"Could not disambiguate '{ambiguous_word}'.")

# Compare with another sentence
sentence2 = "He sat on the bank of the river."
tokens2 = word_tokenize(sentence2)
sense2 = lesk(tokens2, ambiguous_word, pos='n')
if sense2:
    print(f"\nIn '{sentence2}':")
    print(f"Chosen Sense: {sense2.name()} - {sense2.definition()}")

# -----------------------------------------------------

# Q20: Basic Information Retrieval using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "The dog barked.",
    "The brown fox is quick.",
    "Another document about foxes and dogs."
]

vectorizer = TfidfVectorizer()

# Calculate TF-IDF scores
tfidf_matrix = vectorizer.fit_transform(documents)

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

print("\nQ20: TF-IDF Example:")
print(f"Feature Names (Vocabulary): {feature_names}")
print("TF-IDF Matrix (sparse format):")
print(tfidf_matrix)
# To see the dense matrix (might be large for many docs/words)
# print(tfidf_matrix.toarray())

# Example: Show TF-IDF scores for the first document
doc_id = 0
print(f"\nTF-IDF scores for Document {doc_id+1}:")
feature_index = tfidf_matrix[doc_id,:].nonzero()[1]
tfidf_scores = zip([feature_names[i] for i in feature_index], [tfidf_matrix[doc_id, x] for x in feature_index])
for word, score in tfidf_scores:
    print(f"- {word}: {score:.4f}")


# -----------------------------------------------------

### Q21: Syntax-Driven Semantic Analysis (Noun Phrase Extraction)
##import spacy
##
### Using the same SpaCy model as Q16
### nlp_spacy = spacy.load("en_core_web_sm") # Already loaded or load here
##
##text = "The quick brown fox saw the lazy dog with a telescope."
##doc = nlp_spacy(text)
##
##print("\nQ21: Noun Phrase Extraction (SpaCy):")
##if not doc.noun_chunks:
##     print("No noun chunks found.")
##for chunk in doc.noun_chunks:
##    print(f"- Phrase: '{chunk.text}', Root word: '{chunk.root.text}', Root POS: {chunk.root.pos_}")
##    # Basic "meaning" here is just the text and the head word
##    # More complex analysis would involve relations, WordNet lookup, etc.

# -----------------------------------------------------

### Q22: Reference Resolution (Coreference) - Basic SpaCy/NeuralCoref (if installed) or Heuristic
##import spacy
##
### OPTION 1: Using neuralcoref (if installed and compatible)
### Needs: pip install neuralcoref
### Needs a compatible SpaCy model (often medium or large) and may require specific versions.
### Check neuralcoref documentation for current compatibility.
### Example: python -m spacy download en_core_web_lg
##try:
##    import neuralcoref
##    # Load a larger model potentially needed by neuralcoref
##    try:
##        nlp_coref = spacy.load('en_core_web_sm') # Try small first, might need 'en_core_web_lg'
##        # Add neuralcoref to SpaCy's pipeline
##        neuralcoref.add_to_pipe(nlp_coref)
##        text = "My sister has a dog. She loves him."
##        doc_coref = nlp_coref(text)
##
##        print("\nQ22: Reference Resolution (using neuralcoref):")
##        if doc_coref._.has_coref:
##             print(f"Clusters found: {doc_coref._.coref_clusters}")
##        else:
##             print("No coreference clusters found by neuralcoref.")
##
##    except OSError:
##         print("\nQ22: Neuralcoref model load failed (e.g., 'en_core_web_lg' not downloaded or neuralcoref not added).")
##         print("Falling back to simple heuristic.")
##         # Fallback to Option 2 code below if loading fails
##         use_heuristic = True
##    except Exception as e: # Catch other potential neuralcoref errors
##        print(f"\nQ22: Error loading/using neuralcoref: {e}")
##        print("Falling back to simple heuristic.")
##        use_heuristic = True
##
##
##except ImportError:
##    print("\nQ22: 'neuralcoref' library not installed. Install with: pip install neuralcoref")
##    print("Falling back to simple heuristic.")
##    use_heuristic = True
##
### OPTION 2: Simple Heuristic (if neuralcoref is not available or fails)
##if 'use_heuristic' in locals() and use_heuristic:
##    # Very basic heuristic: link pronoun to nearest preceding noun phrase
##    # Doesn't handle gender, number, complex cases, etc.
##    nlp_simple = spacy.load("en_core_web_sm") # Use the small model
##    text = "My sister has a dog. She loves him."
##    doc_simple = nlp_simple(text)
##    print("\nQ22: Reference Resolution (Simple Heuristic):")
##    pronouns = [tok for tok in doc_simple if tok.pos_ == 'PRON']
##    noun_phrases = list(doc_simple.noun_chunks) # Get noun chunks
##
##    for pron in pronouns:
##        resolved = False
##        # Iterate backwards through preceding noun phrases
##        for chunk in reversed([nc for nc in noun_phrases if nc.end <= pron.i]):
##            print(f"Linking '{pron.text}' (at index {pron.i}) to candidate '{chunk.text}' (ends at {chunk.end})")
##            resolved = True # Found the *nearest* preceding NP
##            break # Simple heuristic: take the first one found going backward
##        if not resolved:
##            print(f"Could not resolve '{pron.text}' using simple heuristic.")

# -----------------------------------------------------

# Q23: Evaluate Coherence (Simple Sentence Similarity)
import nltk
# nltk.download('punkt') # Ensure downloaded
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

text = """
Natural language processing enables computers to understand human language.
This field combines computer science and linguistics.
Applications include machine translation and sentiment analysis.
These tools are becoming increasingly important in various industries.
However, challenges remain in handling ambiguity and context.
"""

sentences = nltk.sent_tokenize(text)

print(f"\nQ23: Coherence Evaluation (Sentence Similarity):")
if len(sentences) < 2:
    print("Need at least two sentences to evaluate coherence.")
else:
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)
        similarities = []
        for i in range(len(sentences) - 1):
            sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[i+1])
            similarities.append(sim[0][0])
            print(f"Similarity between sentence {i+1} and {i+2}: {sim[0][0]:.4f}")

        average_similarity = np.mean(similarities)
        print(f"\nAverage adjacent sentence similarity: {average_similarity:.4f}")
        # Very basic interpretation
        if average_similarity > 0.1:
             print("Suggests some topical coherence.")
        else:
             print("Suggests low topical coherence.")
    except ValueError as e:
        print(f"Could not compute TF-IDF, possibly empty sentences? Error: {e}")


# -----------------------------------------------------

# Q24: Recognize Dialog Acts (Simple Keyword/Rule-Based)
def recognize_dialog_act(utterance):
    utterance_lower = utterance.lower()
    if utterance.endswith('?'):
        return "Question"
    elif utterance_lower.startswith(("please", "can you", "could you")):
        return "Request"
    elif utterance_lower.startswith(("yes", "no", "okay", "sure")):
        return "Answer/Acknowledgement"
    elif any(word in utterance_lower for word in ["i think", "i believe", "perhaps", "maybe"]):
         return "Statement (Opinion)"
    elif any(word in utterance_lower for word in ["thank you", "thanks"]):
         return "Thanking"
    elif utterance_lower.startswith(("hi", "hello", "hey")):
         return "Greeting"
    else:
        # Default/fallback
        return "Statement (Inform)"

dialog = [
    "Hello there!",
    "Can you tell me the time?",
    "Yes, it is 3 PM.",
    "Thank you very much.",
    "Maybe I should go now?"
]

print("\nQ24: Dialog Act Recognition (Rule-Based):")
for utt in dialog:
    act = recognize_dialog_act(utt)
    print(f"'{utt}' -> {act}")

# -----------------------------------------------------

### Q25: GPT-3 Text Generation (Requires OpenAI API Key)
##import os
##import openai
##
### Ensure you have set your OPENAI_API_KEY environment variable
### Example: export OPENAI_API_KEY='your-key-here' (Linux/macOS)
### Example: set OPENAI_API_KEY=your-key-here (Windows cmd)
### Example: $env:OPENAI_API_KEY='your-key-here' (Windows PowerShell)
##
### Load the API key from environment variable
##api_key = os.getenv("OPENAI_API_KEY")
##
##print("\nQ25: GPT-3 Text Generation:")
##
##if api_key:
##    openai.api_key = api_key
##    prompt = "Write a short paragraph about the benefits of learning Python for natural language processing."
##    try:
##        # Use the newer OpenAI client structure (v1.0+)
##        client = openai.OpenAI() # Initializes client with api_key from env var
##        response = client.completions.create(
##          model="gpt-3.5-turbo-instruct", # Or another suitable model like "text-davinci-003" if available
##          prompt=prompt,
##          max_tokens=100,  # Limit the length of the generated text
##          temperature=0.7 # Controls randomness (0=deterministic, 1=more random)
##        )
##        generated_text = response.choices[0].text.strip()
##        print("Prompt:", prompt)
##        print("Generated Text:", generated_text)
##
##    except Exception as e:
##        print(f"Error calling OpenAI API: {e}")
##        print("Please check your API key, internet connection, and OpenAI account status.")
##else:
##    print("OPENAI_API_KEY environment variable not set. Skipping GPT-3 generation.")
##    print("Set the variable and rerun the script.")
##
### -----------------------------------------------------
##
### Q26: Machine Translation (Hugging Face Transformers)
##from transformers import pipeline
##
##print("\nQ26: Machine Translation (English to French):")
##
##try:
##    # Load the translation pipeline
##    # This will download the model if run for the first time
##    translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
##
##    english_text = "Hello, how are you today? Natural language processing is fascinating."
##
##    # Perform translation
##    french_translation = translator(english_text, max_length=100)[0]['translation_text'] # max_length to avoid truncation warnings
##
##    print(f"English: {english_text}")
##    print(f"French: {french_translation}")
##
##except Exception as e:
##    print(f"Error initializing or using Hugging Face pipeline: {e}")
##    print("Ensure 'transformers' and 'torch' (or 'tensorflow') are installed and you have internet access.")
##
### -----------------------------------------------------

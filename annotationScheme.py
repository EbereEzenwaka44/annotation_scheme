import numpy as np


class Annotation:
    def __init__(self, vocab_items, annotators, labels):
        self.vocab_items = vocab_items
        self.annotators = annotators
        self.labels = labels

    def annotate(self, n):
        annotations = {}
        for item in self.vocab_items: #words
            annotators_sample = np.random.choice(self.annotators, n)
            labels_sample = np.random.choice(self.labels, n)
            annotations[item] = list(zip(annotators_sample, labels_sample))
        return annotations

    def complexity(self, item_annotations):
        labels = [label for _, label in item_annotations]
        return np.mean(labels)

    def subjectivity(self, item_annotations):
        complexity = self.complexity(item_annotations)
        labels = [label for _, label in item_annotations]
        return np.mean([np.abs(complexity - label) for label in labels])

    def is_complex(self, item_annotations, Tc):
        complexity = self.complexity(item_annotations)
        return complexity > Tc

    def is_subjective(self, item_annotations, Ts):
        subjectivity = self.subjectivity(item_annotations)
        return subjectivity > Ts

V = ['word1', 'word2', 'word3']
H = ['annotator1', 'annotator2', 'annotator3', 'annotator4', 'annotator5']
L = [1, 2, 3, 4, 5]

annotation_scheme = Annotation(V, H, L)
annotations = annotation_scheme.annotate(n=3)

Tc = 3
Ts = 1

for item, item_annotations in annotations.items():
    print(f"Item: {item}")
    print(f"Annotations: {item_annotations}")
    print(f"Complexity: {annotation_scheme.complexity(item_annotations)}")
    print(f"Subjectivity: {annotation_scheme.subjectivity(item_annotations)}")
    print(f"Is complex: {annotation_scheme.is_complex(item_annotations, Tc)}")
    print(f"Is subjective: {annotation_scheme.is_subjective(item_annotations, Ts)}")


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')

# Text to annotate
text = "This is an example sentence to illustrate the annotation process."

# Tokenize and remove stopwords
tokens = word_tokenize(text)
V = [word for word in tokens if word.isalnum() and word not in stopwords.words('english')]

# Annotators and labels
H = ['annotator1', 'annotator2', 'annotator3', 'annotator4', 'annotator5']
L = [1, 2, 3, 4, 5]

# Annotation scheme
annotation_scheme = Annotation(V, H, L)
annotations = annotation_scheme.annotate(n=3)

Tc = 3
Ts = 1

for item, item_annotations in annotations.items():
    print(f"Item: {item}")
    print(f"Annotations: {item_annotations}")
    print(f"Complexity: {annotation_scheme.complexity(item_annotations)}")
    print(f"Subjectivity: {annotation_scheme.subjectivity(item_annotations)}")
    print(f"Is complex: {annotation_scheme.is_complex(item_annotations, Tc)}")
    print(f"Is subjective: {annotation_scheme.is_subjective(item_annotations, Ts)}")

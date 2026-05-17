import unittest
import sys
sys.path.insert(0, '.')
from app import read_article, sentence_similarity, build_similarity_matrix, generate_summary

class TestTextSummarizer(unittest.TestCase):

    def test_read_article(self):
        text = "AI is transforming the world. Machine learning is a subset of AI. Deep learning uses neural networks."
        result = read_article(text)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_sentence_similarity(self):
        sent1 = ["machine", "learning", "is", "good"]
        sent2 = ["machine", "learning", "is", "great"]
        result = sentence_similarity(sent1, sent2)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 1)

    def test_generate_summary(self):
        text = "Artificial intelligence is transforming the world. Machine learning is a subset of AI. Deep learning uses neural networks. AI is used in healthcare finance and education. Data science is growing rapidly."
        result = generate_summary(text, 2)
        self.assertNotEqual(result, "")
        self.assertIsInstance(result, str)

if __name__ == '__main__':
    unittest.main()
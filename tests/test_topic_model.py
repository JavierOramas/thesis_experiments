from pipeline.topic_model.k_means_llm import TopicModel
import unittest

class TestTopicModel(unittest.TestCase):
    
    def setUp(self):
        self.tm = TopicModel()
        sentences = ["This is a sentence about sports number " + str(i) for i in range(1, 26)] + \
                ["This is a sentence about politics number " + str(i) for i in range(1, 26)] + \
                ["This is a sentence about technology number " + str(i) for i in range(1, 26)] + \
                ["This is a sentence about art number " + str(i) for i in range(1, 26)]

        self.topics = self.tm.get_topics(sentences)

    def test_generate_topics(self):    

        self.assertEqual(len(self.topics.items()),4)

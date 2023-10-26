from pipeline.topic_model.k_means_llm import TopicModel


def test_k():
    
    sentences = ["This is a sentence about sports number " + str(i) for i in range(1, 26)] + \
                 ["This is a sentence about politics number " + str(i) for i in range(1, 26)] + \
                 ["This is a sentence about technology number " + str(i) for i in range(1, 26)] + \
                 ["This is a sentence about art number " + str(i) for i in range(1, 26)]

    tm = TopicModel()
    
    topics = tm.get_topics(sentences)

    assert tm.get_topics == 4
    
    for i in topics:
        assert i != None and i != ""

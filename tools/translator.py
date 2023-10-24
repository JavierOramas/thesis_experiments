from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import MarianMTModel, MarianTokenizer

def translate_es_en(spanish_text):
    device = 'cuda'
    # Load the pre-trained model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-es-en"  # Spanish to English translation model
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)


    # Tokenize the Spanish text
    spanish_text_split = [spanish_text[i:i + 300] for i in range(0, len(spanish_text), 300)]
    
    translated_text = []
    for text in spanish_text_split:
        inputs = tokenizer.encode(text, return_tensors="pt")
        translated_ids = model.generate(inputs, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
        translated_text.append(tokenizer.decode(translated_ids[0], skip_special_tokens=True))
    
    return " ".join(translated_text)

def translate_en_es(english_text):
    # Load the pre-trained model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-en-es"  # English to Spanish translation model
    model = MarianMTModel.from_pretrained(model_name).to(device)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    # Tokenize the English text
    inputs = tokenizer.encode(english_text, return_tensors="pt").to(device)

    # Generate translation
    translated_ids = model.generate(inputs, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    
    return translated_text

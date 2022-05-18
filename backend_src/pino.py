import numpy
import torch

import spacy
from spacy.tokens import Token
from spacy import displacy

import nltk
from nltk.corpus import wordnet as wn

from spacy.tokens import Span
from spacytextblob.spacytextblob import SpacyTextBlob

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


import os
CACHE_DIR = os.path.join(os.getcwd(), "data","pretrained_models","huggingface","transformers") 

# for subjectivity
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def summarize(text, model):
    if model == "Pegasus":
        result = pegasus(text)
    elif model == "DistilBart1":
        result = distilbart1(text)
    elif model == "DistilBart2":
        result = distilbart2(text)
    elif model == "Bart1":
        result = bart1(text)
    elif model == "Bart2":
        result = bart2(text)
    result = result.strip()
    return(result)

def render_ner_analysis(text):
    #os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), "data","pretrained_models") # metterlo in una funzione esterna ovviamente
    nltk.data.path.append(os.path.join(os.getcwd(), "data","nltk_data")) # metterlo in una funzione esterna ovviamente
    def is_person_getter(token):
        if token.pos_ == "NOUN" and wn.synsets(token.lemma_):
            return (wn.synset('person.n.01') in wn.synsets(token.lemma_)[0].lowest_common_hypernyms(wn.synset('person.n.01')))
        else:
            return(False)
    Token.set_extension("is_person", getter=is_person_getter, force=True)
    nlp = spacy.load("en_core_web_lg")
    ruler = nlp.add_pipe("entity_ruler", after="ner", config={"overwrite_ents": True})
    patterns = [{"label":"Person","pattern":[{"POS": "NOUN","OP":"?"}, {"POS":"NOUN",'_':{'is_person': True}}]},
                {"label":"Person","pattern":[{"POS": "ADJ","OP":"?"}, {"POS":"NOUN",'_':{'is_person': True}}]}]
    ruler.add_patterns(patterns)
    doc=nlp(text)
    html = displacy.render(doc, style="ent")
    return(html)

def render_sentiment_analysis(text):
    from operator import itemgetter
    import numpy as np
    import seaborn as sns
    def get_dict_col(n_col_to_extreme=1000):
      total_col = 2*n_col_to_extreme
      custom_list_colors = sns.diverging_palette(10, 220, center ="light", s=99, n=total_col+1)
      dict_col = {}
      for n,i in enumerate(np.linspace(-1.0,1.0,total_col+1)):
        col_rgb = tuple(map(lambda x: int(round(x*255)),custom_list_colors[n]))
        dict_col[f'{i:.3f}'] = '#%02x%02x%02x' % (col_rgb)
      dict_col['0.000'] = "#ffffff"
      return(dict_col)

    nlp = spacy.load('en_core_web_lg')
    nlp.add_pipe('spacytextblob')
    #text = 'I had a really horrible day. It was the worst day ever! But every now and then I have a really good day that makes me happy.'
    doc = nlp(text)
    ents = []
    for i in doc.sents:
      ents.append(Span(doc,i.start,i.end,label=f"{i._.blob.polarity:.3f}"))
    doc.ents = ents
    my_palette = get_dict_col()

    polarities = list(map(lambda x: x.label_, doc.ents))
    options = {"colors": dict(zip(polarities, itemgetter(*polarities)(my_palette)))}
    html = displacy.render(doc, style="ent", options=options)
    return(html)
    
def render_subjectivity_analysis(text):
    from operator import itemgetter
    import numpy as np
    import seaborn as sns
    def get_dict_col(n_col_to_extreme=1000):
      total_col = 2*n_col_to_extreme
      custom_list_colors = sns.diverging_palette(10, 220, center ="light", s=99, n=total_col+1)
      dict_col = {}
      for n,i in enumerate(np.linspace(-1.0,1.0,total_col+1)):
        col_rgb = tuple(map(lambda x: int(round(x*255)),custom_list_colors[n]))
        dict_col[f'{i:.3f}'] = '#%02x%02x%02x' % (col_rgb)
      dict_col['0.000'] = "#ffffff"
      return(dict_col)

    #nlp = spacy.load('en_core_web_lg')
    #nlp.add_pipe('spacytextblob')
    #doc = nlp(text)
    #ents = []
    #for i in doc.sents:
    #  ents.append(Span(doc,i.start,i.end,label=f"{i._.blob.polarity:.3f}"))
    #doc.ents = ents
    #my_palette = get_dict_col()

    tokenizer = AutoTokenizer.from_pretrained("spartan97/distilbert-base-uncased-finetuned-objectivity-rotten")
    model = AutoModelForSequenceClassification.from_pretrained("spartan97/distilbert-base-uncased-finetuned-objectivity-rotten")

    doc = nlp(text)
    inputs = doc.sents # dovrebbe essere una lista
    for i in inputs:
        tokenize_input = tokenizer([text], max_length=1024, return_tensors='pt')
        summary_ids = model.generate(tokenize_input['input_ids'], num_beams=5)#, max_length=5, early_stopping=True)
    output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]


    polarities = list(map(lambda x: x.label_, doc.ents))
    options = {"colors": dict(zip(polarities, itemgetter(*polarities)(my_palette)))}
    html = displacy.render(doc, style="ent", options=options)
    return(html)


### models
def pegasus(text):
    src_text = [text]
    model_name = "google/pegasus-xsum"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = PegasusTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    model = PegasusForConditionalGeneration.from_pretrained(model_name, cache_dir=CACHE_DIR).to(device)
    batch = tokenizer(src_text, truncation=True, padding="longest", return_tensors="pt").to(device)
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return(tgt_text[0])

def distilbart1(text):
    model_chosen = "sshleifer/distilbart-xsum-12-1"
    model = BartForConditionalGeneration.from_pretrained(model_chosen, cache_dir=CACHE_DIR)
    tokenizer = BartTokenizer.from_pretrained(model_chosen, cache_dir=CACHE_DIR)

    # Generate Summary
    inputs = tokenizer([text], max_length=1024, return_tensors='pt')
    summary_ids = model.generate(inputs['input_ids'], num_beams=5)#, max_length=5, early_stopping=True)
    output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    return(output[0])

def distilbart2(text):
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6", cache_dir=CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6", cache_dir=CACHE_DIR)
    # Generate Summary
    inputs = tokenizer([text],max_length=1024,return_tensors='pt')#, max_length=1024, return_tensors='pt')
    summary_ids = model.generate(inputs['input_ids'], num_beams=5)#, max_length=5, early_stopping=True)
    output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    return(output[0])

def bart1(text):
    tokenizer = AutoTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum", cache_dir=CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained("philschmid/bart-large-cnn-samsum", cache_dir=CACHE_DIR)

    inputs = tokenizer([text],max_length=1024,return_tensors='pt')#, max_length=1024, return_tensors='pt')
    summary_ids = model.generate(inputs['input_ids'], num_beams=5)#, max_length=5, early_stopping=True)
    output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    return(output[0])

def bart2(text):
    tokenizer = AutoTokenizer.from_pretrained("lidiya/bart-large-xsum-samsum", cache_dir=CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained("lidiya/bart-large-xsum-samsum", cache_dir=CACHE_DIR)

    inputs = tokenizer([text],max_length=1024,return_tensors='pt')#, max_length=1024, return_tensors='pt')
    summary_ids = model.generate(inputs['input_ids'], num_beams=5)#, max_length=5, early_stopping=True)
    output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    return(output[0])
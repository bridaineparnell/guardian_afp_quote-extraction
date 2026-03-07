import os
import pandas as pd
import spacy
from utils.quote_extraction import extract_quotes_and_sentence_speaker
from utils.preprocessing import sentencise_text
from tqdm import tqdm
import re

df = pd.read_csv("main_dataset.csv")

# First clean the text

# def text_repair(text):
#     if not isinstance(text, str): return ""
#
#     stats = {"mojibake_fixes": 0, "quote_normalizations": 0}
#
#     # 1. Targeted Mojibake repair (Priority order matters)
#     # The 'â€' often acts as a prefix for multiple types of marks
#     mojibake_map = {
#         "â€œ": "“",  # Opening double
#         "â€": "”",  # Closing double (standard)
#         "â€\x9d": "”",  # Closing double (variant)
#         "â€": "”",  # Closing double (fallback for your 'wrong' result)
#         "â€™": "'",  # Apostrophe
#         "â€”": "—",  # Em dash
#         "â€“": "–",  # En dash
#         "Â": ""  # Strips the ghost space often found after quotes
#     }
#
#     for bad, good in mojibake_map.items():
#         count = text.count(bad)
#         if count > 0:
#             text = text.replace(bad, good)
#             stats["mojibake_fixes"] += count
#
#     # 2. Smart Quote Normalization
#     # Force double straight quotes to smart
#     text, d_count = re.subn(r'"([^"\n]+?)"', r'“\1”', text)
#
#     # Single Quotes: 'Text' -> “Text” (Excluding contractions like I'm)
#     text, s_count = re.subn(r"(^|\s)'([^'\n]+?)'($|[\s\.,!\?])", r'\1“\2”\3', text)
#
#     stats["quote_normalizations"] = d_count + s_count
#
#     return " ".join(text.split()), stats
#
# report = {"processed_rows": 0, "mojibake": 0, "quotes": 0}
#
# def clean_and_track(text):
#     if not isinstance(text, str) or len(text.strip()) == 0:
#         return text
#
#     cleaned, stats = text_repair(text)
#
#     # Update the report tracker
#     report["processed_rows"] += 1
#     report["mojibake"] += stats.get("mojibake_fixes", 0)
#     report["quotes"] += stats.get("quote_normalizations", 0)
#
#     return cleaned
#
# print("Starting cleanup...")
# df['body_text'] = df['body_text'].apply(clean_and_track)
#
# print("\n" + "="*40)
# print("FINAL CLEANUP REPORT")
# print("="*40)
# print(f"Total Rows Processed:   {report['processed_rows']}")
# print(f"Mojibake Fixed:        {report['mojibake']}")
# print(f"Quote Normalizations:  {report['quotes']}")
# print("="*40)
#
# # Save the cleaned text csv
#
# df.to_csv("clean_main_dataset.csv", index=False)

# Read it back in

df = pd.read_csv("clean_main_dataset.csv")

# Load nlp models

nlp_light = spacy.load("en_core_web_sm")
nlp_trf = spacy.load("en_core_web_trf")

# Load tqdm to follow progress

tqdm.pandas(desc="Extracting Quotes")

# Break the text up so it can be processed at <512 tokens

def chunk_text_by_words(text, chunk_size=400):
    """
    Splits text into chunks of roughly 'chunk_size' words.
    Tries to split at double newlines first to preserve paragraph structure.
    """
    if not text or pd.isna(text):
        return []

    # Split by paragraphs first to avoid cutting a quote in half
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_count = 0

    for para in paragraphs:
        para_count = len(para.split())
        if current_count + para_count <= chunk_size:
            current_chunk.append(para)
            current_count += para_count
        else:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_count = para_count

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks

# Get quotes and speakers

def process_row(text):
    if not text or pd.isna(text):
        return [], [] # Return two empty lists to avoid Series size errors

    article_doc = nlp_trf(text[:1500])  # Scan the first 1500 chars for context
    people = [ent.text for ent in article_doc.ents if ent.label_ == "PERSON"]
    main_character = people[0] if people else "Unknown Source"

    quotes = []
    speakers = []

    text_chunks = chunk_text_by_words(text, chunk_size=400)

    for chunk in text_chunks:
        results, _ = extract_quotes_and_sentence_speaker(chunk, nlp_trf, debug=False)

        for item in results:
            raw_s = item.speaker if hasattr(item, 'speaker') else item[1]
            quote_text = item.quote_text if hasattr(item, 'quote_text') else item[0]

            final_s = raw_s.strip()
            speaker = nlp_trf(final_s)

            valid = ['PERSON', 'ORG', 'NORP', 'FAC', 'GPE']
            is_real = any(ent.label_ in valid for ent in speaker.ents)
            is_pron = speaker.text.lower() in ['he', 'she', 'they', 'it', 'who']

            words = speaker.text.lower().split()
            if len(words) == 1 and words[0] in ['and', 'the', 'a', 'an', 'but', 'or']:
                final_speaker = "Unknown"

            elif not is_real and not is_pron:
                persons = [ent.text for ent in speaker.ents if ent.label_ == "PERSON"]
                if persons:
                    speaker = persons[0]
                else:
                    speaker = f"{speaker} (Unknown)"

            if is_pron:
                speaker = f"{speaker} (likely {main_character})"
                pass

            quotes.append(quote_text)
            speakers.append(speaker)

    # For when running checks
    # status = "✅" if quotes else "❌"
    # print(f"{status} Found {len(quotes)} quotes in this article.")

    combined = list(zip(quotes, speakers))

    unique = list(set(combined))

    if unique:
        quotes, speakers = zip(*unique)
    else:
        quotes, speakers = [], []

    return list(quotes), list(speakers)

# Checks

# df_sample = df.sample(n=10).copy()
# df_sample[['quotes', 'speakers']] = df_sample['body_text'].progress_apply(
#     lambda x: pd.Series(process_row(x))
# )
# pd.set_option('display.max_columns', None)
# print(df_sample[['news_title', 'quotes', 'speakers']])
#
# df_sample.to_csv("df_sample.csv", index=False)

# Run on main

df[['quotes', 'speakers']] = df['body_text'].progress_apply(lambda x: pd.Series(process_row(x)))

# # Save results
df.to_csv("quotes_speakers_main.csv", index=False)
print("Finished! Results saved")
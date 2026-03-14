import os
import sys
import spacy

# Commands to ensure cluster is correctly working
try:
    import cupy
    import cupyx
    print("CuPy check: Success")
except ImportError:
    print("CuPy check: Failed")

try:
    # Check if Cupy can see the device before spacy tries
    device_count = cupy.cuda.runtime.getDeviceCount()
    print(f"Cupy sees {device_count} GPU device(s).")

    if device_count > 0:
        spacy.require_gpu()
        print("Spacy GPU: Active ✅")
    else:
        print("Spacy GPU: No devices found by Cupy ❌")
except Exception as e:
    print(f"Spacy GPU: Failed ❌ ({e})")

# Import packages

import pandas as pd
import coreferee
from utils.quote_extraction import extract_quotes_and_sentence_speaker
from utils.preprocessing import sentencise_text
from tqdm import tqdm
import re

# df = pd.read_csv("main_dataset.csv")
#
# # First clean the text
#
# def text_repair(text):
#     """
#     Standardise the way quotes look across the set
#     and sort out text encoding issues
#     """
#     # Safety code
#     if not isinstance(text, str): return ""
#     # Monitor the progress
#     stats = {"mojibake_fixes": 0, "quote_normalizations": 0}
#
#     # Mojibake repair (Priority order matters)
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
#     # Quote Normalization
#
#     # Pre - Normalization: Standardize complex endings(e.g., ".' or '." to.”)
#     text, w_count = re.subn(r"['\"”]+?\s*?\.", ".”", text)
#     text, ww_count = re.subn(r"\.\s*?['\"”]+", ".”", text)
#     # Force any ' at the end of a sentence to be a smart quote
#     text, l_count = re.subn(r"([\.\?!])'", r"\1”", text)
#
#     # Force double straight quotes to smart
#     text, d_count = re.subn(r'"([^"]+?)"', r'“\1”', text)
#
#     # Single Quotes: 'Text' -> “Text” (Excluding contractions like I'm)
#     text, s_count = re.subn(r"(\s|^)'([\s\S]+?)'(\s|$|[.,!?;])", r'\1“\2”\3', text)
#
#     # Double-Double Quotes: ""Text"" -> "Quote"
#     text, dd_count = re.subn(r'""([^"]+?)""', r'“\1”', text)
#
#     # Any remaining "
#     text, ar_count = re.subn(r'"', '”', text)
#
#     stats["quote_normalizations"] = w_count + ww_count + d_count + s_count + dd_count + ar_count + l_count
#
#     # Rescue broken paragraphs that end with " but don't start with one
#     paragraphs = text.split('\n')
#     healed_paragraphs = []
#     for para in paragraphs:
#         p = para.strip()
#         if not p:
#             healed_paragraphs.append("")
#             continue
#
#         # Add a start quote mark if you find a closing one
#         if re.search(r'[\.\?!][”"\'\’]$', p) and not re.match(r'^[*•\-\s]*?[“"\'\‘]', p):
#             p = "“" + p
#
#         # Add a closing quote mark if you find an opening one
#         if re.match(r'^[*•\-\s]*?[“"\'\‘]', p) and not re.search(r'[”"\'\’]$', p):
#             if p[-1] in ['.', '!', '?']:
#                 p = p + "”"
#             else:
#                 p = p + ".”"
#
#         healed_paragraphs.append(p)
#
#     text = '\n'.join(healed_paragraphs)
#
#     return text.strip(), stats
#
# report = {"processed_rows": 0, "mojibake": 0, "quotes": 0}
#
# def clean_and_track(text):
#       """
#       Function to iterate through and clean the text
#       while tracking the progress
#       """
#
#       # Safety code
#       if not isinstance(text, str) or len(text.strip()) == 0:
#         return text
#
#       # Run the cleanup and gather the stats
#       cleaned, stats = text_repair(text)
#
#       # Update the report tracker
#       report["processed_rows"] += 1
#       report["mojibake"] += stats.get("mojibake_fixes", 0)
#       report["quotes"] += stats.get("quote_normalizations", 0)
#
#       return cleaned
#
# # Run on the df
# print("Starting cleanup...")
# df['body_text'] = df['body_text'].apply(clean_and_track)
#
# # Report results of cleanup
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
# df.to_csv("clean_main_dataset_3.csv", index=False, quoting=1)

# Read it back in

df = pd.read_csv("clean_main_dataset_3.csv")

# Load nlp models (no need to load _lg, coref calls this)

nlp_light = spacy.load('en_core_web_sm')
nlp_trf = spacy.load("en_core_web_trf")
nlp_trf.add_pipe("coreferee")

# Load tqdm to follow progress

tqdm.pandas(desc="Extracting Quotes")

# Get quotes and speakers
def resolve_with_coreferee(doc, raw_speaker_text, quote_text):
    """
    If pronoun, resolve the name,
    if it's already a name, keep it.
    """
    # Safety check
    if not raw_speaker_text or not doc._.coref_chains:
        return raw_speaker_text

    # Find the quote in the chunk
    quote_start_char = doc.text.find(quote_text[:50])

    target_token = None
    min_dist = 9999999

    for token in doc:
        if token.text.lower() == raw_speaker_text.lower():
            # Calculate distance between potential speaker and the quote
            dist = abs(token.idx - quote_start_char)
            if dist < min_dist:
                min_dist = dist
                target_token = token

    if target_token is None:
        return raw_speaker_text

        # Now that we have the nearest potential speaker,
        # Coreferee looks at its internal chain map to find the name.
    resolved = doc._.coref_chains.resolve(target_token)

    if resolved:
        # Returns 'Sam Altman' instead of 'he'
        return " ".join([t.text for t in resolved])

    # Fallback in case 'it' fails to resolve because it's an AI
    if target_token and raw_speaker_text.lower() == 'it':
        ai_keywords = ['chatgpt', 'bot', 'system', 'model', 'ai', 'algorithm', 'robot', 'chatbot', 'gemini']
        # Look at the 10 tokens before the quote
        for i in range(max(0, target_token.i - 10), target_token.i):
            if doc[i].text.lower() in ai_keywords:
                return doc[i].text

    return raw_speaker_text

# Break the text up so it can be processed at <512 tokens, but overlap so we don't miss bits

def overlapping_chunks(text, chunk_size=400, overlap=100):
    """
    Splits text into chunks of `chunk_size` words,
    with `overlap` to capture across chunks.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    # Step size is chunk_size minus overlap
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk_words = words[i: i + chunk_size]
        chunks.append(" ".join(chunk_words))

        # Stop if we've reached the end of the text
        if i + chunk_size >= len(words):
            break

    return chunks

def clean_speaker_name(name_str):
    """
    Cleans and validates the speaker text.
    Returns 'Unknown' if the text grabbed is noise or a placeholder.
    """
    # Safety code
    if not name_str:
        return "Unknown"

    # Remove leading/trailing whitespace and common trailing punctuation
    clean = name_str.strip().strip(',.:; ')

    # If the regex grabbed a whole sentence (more than 5 words)
    # Use spacy to find the actual Person/Org inside that mess
    if len(clean.split()) > 5:
        temp_doc = nlp_light(clean)
        # Grab the first entity found in the messy string
        ents = [e.text for e in temp_doc.ents if e.label_ in ['PERSON', 'ORG','NORP', 'FAC', 'GPE', 'PRODUCT', 'WORK_OF_ART']]
        if ents:
            return ents[0]
        else:
            # If no entity, it's probably just noise (like "it was claimed")
            return "Unknown"

    # Stop-word filter
    if clean.lower() in ['and', 'the', 'that', 'which', 'when']:
        return "Unknown"

    return clean

def process_row(text):
    """
    Go through the df row by row, grabbing the articles
    chunk them, get the quotes and speakers
    throw the speakers to coref to resolve
    return two lists of quotes and speakers
    and a dictionary of attributed quotes:speaker
    """
    # Safety code
    if not text or pd.isna(text):
        return [], [], {}, 0 # Return empty variables to avoid series size errors

    # Create chunks <512 for the transformer, but overlap so we don't miss anything
    chunks = overlapping_chunks(text, chunk_size=400, overlap=100)

    # Start lists for quotes and speakers
    quotes = []
    speakers = []

    # Run the quote extraction
    for chunk in chunks:
        doc = nlp_trf(chunk)

        results, _ = extract_quotes_and_sentence_speaker(chunk, nlp_trf, debug=False)

        # Gather quotes and speakers
        for item in results:
            q = item.quote_text if hasattr(item, 'quote_text') else item[0]
            s = item.speaker if hasattr(item, 'speaker') else item[1]

            # Hand off resolution to Coreferee
            resolved_s = resolve_with_coreferee(doc, s, q)

            # Clean up speakers
            final_s = clean_speaker_name(resolved_s)

            quotes.append(q)
            speakers.append(final_s)

    # Deduplicate quotes across the whole article
    quote_map = {}
    for q, s in zip(quotes, speakers):
        if q not in quote_map:
            quote_map[q] = []
        quote_map[q].append(s)

    final_quotes = []
    final_speakers = []
    matched = {}

    # Find the best speaker

    for q_text, s_list in quote_map.items():
        pronouns = {'he', 'she', 'they', 'it', 'who'}
        candidates = [name for name in s_list if "Unknown" not in name]

        if not candidates:
            best_speaker = "Unknown"
        else:
            best_speaker = sorted(candidates, key=lambda x: (x.lower() not in pronouns, len(x)), reverse=True)[0]

        final_quotes.append(q_text)
        final_speakers.append(best_speaker)
        matched[q_text] = best_speaker

    # Monitor the progress
    status_icon = "✅" if final_quotes else "❌"
    print(f"{status_icon} Quotes: {len(final_quotes)}")

    return final_quotes, final_speakers, matched, len(final_quotes)


# Run on a sample first to iterate on regex and cleaning and to debug

df_sample = df.head(10).copy()
df_sample[['quotes', 'speakers', 'attribution', 'quote_count']] = df_sample['body_text'].progress_apply(
    lambda x: pd.Series(process_row(x))
)
pd.set_option('display.max_columns', None)
print(df_sample[['news_title', 'quotes', 'speakers', 'attribution', 'quote_count']])

df_sample.to_csv("df_sample_2.csv", index=False)
#
# # Run on clean dataset and save results
#
# df[['quotes', 'speakers', 'attribution', 'quote_count']] = df['body_text'].progress_apply(lambda x: pd.Series(process_row(x)))
#
# # # Save results
# df.to_csv("quotes_speakers_coref_2.csv", index=False)
# print("Finished! Results saved")
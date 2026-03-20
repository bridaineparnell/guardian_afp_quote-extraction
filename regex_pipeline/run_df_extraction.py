import os
import sys
import spacy
import re

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

# Read in data

df = pd.read_csv("clean_main_dataset_6.csv")

# Load nlp models (no need to load _lg, coref calls this)

nlp_light = spacy.load('en_core_web_sm')
nlp_trf = spacy.load("en_core_web_trf")
nlp_trf.add_pipe("coreferee")

# Load tqdm to follow progress

tqdm.pandas(desc="Extracting Quotes")

# Create a map/dictionary of all entities in a text

AI_KEYWORDS = {'chatgpt', 'bot', 'system', 'model', 'ai', 'algorithm', 'robot', 'chatbot', 'gemini', 'app'}
IDENTITY_LABELS = ['PERSON', 'ORG', 'NORP']
FORBIDDEN_LABELS = {'GPE', 'LOC', 'FAC'}
# Use the verb list to find active speakers
with open('utils/quote_verb_list.txt', 'r') as f:
    # Use lemmas (root words) for the best matching results
    ATTRIBUTION_VERBS = {line.strip().lower() for line in f if line.strip()}
# Adverbs and particles that often get trapped in speaker extraction
ATTRIBUTION_NOISE = {'later', 'then', 'also', 'now', 'here', 'however', 'finally', 'since', 'originally'}
REGISTRY_NOISE = {'principle', 'likely', 'actually', 'indeed', 'thought', 'course', 't', 's'}

def entity_registry(doc, registry):
    """
    Find entities in the text, including AIs
    """
    # Use the ent label from Spacy
    for ent in doc.ents:
        if ent.label_ in IDENTITY_LABELS or ent.label_ in FORBIDDEN_LABELS:
            full_name = ent.text.strip()

            # GUARD: Ignore single letters or specific noise words tagged as entities
            if len(full_name) < 2 or full_name.lower() in REGISTRY_NOISE:
                continue

            # Look for appositives using universal dependencies from Spacy
            # First, look before the name
            prefix = []
            for i in range(ent.start - 1, -1, -1):
                if i < ent.start - 6: break # Max 5 words back
                word = doc[i].text
                if doc[i].pos_ in ['NOUN', 'ADJ', 'PROPN'] and word.lower() not in full_name.lower():
                    prefix.insert(0, word)
                else:
                    break

            # The 'child' is based on dependency trees
            appos = ""
            for child in ent.root.children:
                if child.dep_ == "appos":
                    appos = "".join([t.text_with_ws for t in child.subtree]).strip()

            full_identity = " ".join(prefix + [full_name]).strip()

            if appos:
                full_identity = f"{full_identity} ({appos})"

            # Only save if it's better and keep its label
            existing = registry.get(full_name, {}).get('text', "")
            if len(full_identity) >= len(existing):
                registry[full_name] = {'text': full_identity, 'label': ent.label_}

            # Make sure that the logic doesn't just end if it finds a surname, it looks for more
            if ent.label_ == "PERSON" and " " in full_name:
                surname = full_name.split()[-1]
                if len(full_identity) >= len(registry.get(surname, {}).get('text', "")):
                    registry[surname] = {'text': full_identity, 'label': 'PERSON'}

    # Now check if it's an AI that's 'speaking'
    for token in doc:
        if token.text.lower() in AI_KEYWORDS and token.ent_type_ == "":
            name = token.text
            identity_text = f"{name} (AI {name if name.lower() != 'system' else 'model'})"

            # Use .get().get() to safely check the length of existing registry entries
            existing_entry_text = registry.get(name, {}).get('text', "")

            if len(identity_text) >= len(existing_entry_text):
                # Save as a dictionary to match the PERSON/ORG entries
                registry[name] = {
                    'text': identity_text,
                    'label': 'AI'  # This label tells the memory track where to put it
                }

def registry_resolve(speaker_text, registry):
    """
    Run the speakers through the registry to get the best version of their identity
    """
    # Safety check
    if not speaker_text or speaker_text == "Unknown":
        return "Unknown"

    # Clean the text
    cleaned = speaker_text.strip().lower()

    # Look in the registry for the person or entity. If they're in the registry,
    # return the best version of identity from the registry.
    for key, val in registry.items():
        if cleaned == key.lower() or cleaned == f"the {key.lower()}":
            return val['text']

    # Substring match
    sorted_keys = sorted(registry.keys(), key=len, reverse=True)
    for key in sorted_keys:
        k_low = key.lower()
        # Check if the cleaned name is actually INSIDE the registry key
        if cleaned in k_low and len(cleaned) > 3:
            if registry[key].get('label') in FORBIDDEN_LABELS and cleaned not in k_low:
                continue
            return registry[key]['text']

    return speaker_text

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
    quote_start_char = doc.text.find(quote_text[:30])
    target_token = None
    min_dist = 9999999

    clean_raw = raw_speaker_text.lower().strip()

    for token in doc:
        if token.text.lower() == clean_raw or token.lemma_.lower() == clean_raw:
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
    search_terms = [raw_speaker_text.lower()]
    if raw_speaker_text.lower() in ['it', 'the bot', 'the model', 'the system']:
        search_terms.extend(AI_KEYWORDS)

    # Look at the tokens surrounding the speaker for an AI keyword
    start_search = max(0, target_token.i - 10)
    end_search = min(len(doc), target_token.i + 5)

    for i in range(start_search, end_search):
        if doc[i].text.lower() in AI_KEYWORDS:
            return doc[i].text  # Return 'ChatGPT' or 'Gemini'

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

    # Strip trailing punctuation and whitespace
    clean = re.sub(r"(\'s|\(.*\))", "", name_str).strip(' ,.:;”"\'').strip()

    # GUARD: If it's 2 characters and doesn't start with a capital, it's not a speaker
    if len(clean) < 2: return "Unknown"
    if len(clean) == 2 and not clean[0].isupper(): return "Unknown"

    # Check for messy/long strings (more than 5 words)
    if len(clean.split()) > 5:
        temp_doc = nlp_light(clean)

        # Check for entities
        ents = [e.text for e in temp_doc.ents if e.label_ in IDENTITY_LABELS]
        if ents:
            return ents[0]

        # Check for AIs
        for token in temp_doc:
            if token.text.lower() in AI_KEYWORDS:
                return token.text

        return "Unknown"

    # Discard if the speaker starts with a conjunction or preposition
    # journalists often use "But he said..." or "To which he added..."
    lower_words = clean.lower().split()
    if lower_words[0] in {'but', 'and', 'to', 'for', 'with', 'when', 'is', 'of'}:
        # If it's a long phrase starting with 'but', try to find the entity inside
        if len(lower_words) > 1:
            clean = " ".join(clean.split()[1:]) # Remove the first word
        else:
            return "Unknown"

    # Clean off verbs in case they were grabbed
    words = clean.split()
    cleaned_words = [
        w for w in words
        if w.lower().strip() not in ATTRIBUTION_VERBS
           and w.lower().strip() not in ATTRIBUTION_NOISE
    ]

    # Remove duplicates in titles
    seen = set()
    deduped_words = []
    for w in cleaned_words:
        if w.lower() not in seen:
            deduped_words.append(w)
            seen.add(w.lower())

    final_name = " ".join(deduped_words).strip()

    return final_name

def process_row(row):
    """
    Go through the df row by row, grabbing the articles and rids,
    chunk the text, get the quotes and speakers
    throw the speakers to coref to resolve
    return two lists of quotes and speakers
    and a dictionary of attributed quotes:speaker
    """
    text = row['body_text']
    rid = row.get('rid', 'unknown')
    pronouns = {'he', 'she', 'they', 'it', 'who'}

    # Safety code
    if not text or pd.isna(text):
        return [], [], {}, 0 # Return empty variables to avoid series size errors

    # Keep track of the last known speaker
    last_person = ["Unknown", -999]
    last_org = ["Unknown", -999]
    last_ai = ["Unknown", -999]

    DECAY_THRESHOLD = 400

    # Initialise the registry
    id_registry = {}
    # Store quotes, speakers
    temp_quotes = []
    temp_speakers = []
    current_token_offset = 0

    # Create chunks <512 for the transformer, but overlap so we don't miss anything
    chunks = overlapping_chunks(text, chunk_size=400, overlap=100)

    # Run the quote extraction
    for chunk in chunks:
        doc = nlp_trf(chunk)

        # Update the registry with any entities found
        entity_registry(doc, id_registry)

        # Update memory with the first entity found in this chunk
        # This is probably the 'current' subject
        for token in doc:
            g_idx = current_token_offset + token.i

            if token.dep_ == "nsubj":
                label = token.ent_type_
                full_id = id_registry.get(token.text, {}).get('text', token.text)

                # Track Persons
                if label == 'PERSON' and label not in FORBIDDEN_LABELS:
                    last_person = [full_id, g_idx]
                # Track Organizations
                elif token.head.lemma_.lower() in ATTRIBUTION_VERBS:
                    if label in {'ORG', 'NORP'} and label not in FORBIDDEN_LABELS:
                        last_org = [full_id, g_idx]
                    # Track AI
                    elif token.text.lower() in AI_KEYWORDS:
                        last_ai = [full_id, g_idx]

        # Extract quotes
        results, _ = extract_quotes_and_sentence_speaker(chunk, nlp_trf, debug=False)

        # Gather quotes and speakers
        for item in results:
            q = item.quote_text if hasattr(item, 'quote_text') else item[0]

            # Only keep if the quote has 3 or more words
            words_in_quote = q.strip('“”" ').split()
            if len(words_in_quote) < 2:
                continue  # Skip this quote

            # Find the approximate global index of the quote
            # We use the start of the quote within the chunk
            quote_chunk_idx = chunk.find(q[:20])
            # Convert character index to rough token index (char_idx / 4)
            global_quote_idx = current_token_offset + (quote_chunk_idx // 4)

            s = item.speaker if hasattr(item, 'speaker') else item[1]
            # Clean up speakers
            clean_s = clean_speaker_name(s)

            # Hand off resolution to Coreferee for specific categories
            low_s = clean_s.lower()

            # Resolve Pronouns to their specific category
            resolved_s = "Unknown"

            if low_s in {'he', 'she'}:
                # Try Coreferee first
                resolved_s = resolve_with_coreferee(doc, clean_s, q)
                # Fallback only to a PERSON
                if resolved_s.lower() in {'he', 'she'}:
                    if global_quote_idx - last_person[1] < DECAY_THRESHOLD:
                        resolved_s = last_person[0]

            elif low_s in {'it', 'the system', 'the bot', 'the app'}:
                resolved_s = resolve_with_coreferee(doc, clean_s, q)
                # Fallback to AI first, then ORG
                if resolved_s.lower() in {'it', 'the system', 'the bot', 'the app'}:
                    if global_quote_idx - last_ai[1] < DECAY_THRESHOLD:
                        resolved_s = last_ai[0]
                    elif global_quote_idx - last_org[1] < DECAY_THRESHOLD:
                        resolved_s = last_org[0]

            elif low_s in {'they', 'the company', 'the team'}:
                if global_quote_idx - last_org[1] < DECAY_THRESHOLD:
                    resolved_s = last_org[0]
                elif global_quote_idx - last_person[1] < DECAY_THRESHOLD:
                    resolved_s = last_person[0]

            else:
                resolved_s = clean_s

            # Look in registry for best match
            final_s = registry_resolve(resolved_s, id_registry)

            temp_quotes.append(q)
            temp_speakers.append(final_s)

            # Update memory to this speaker for the next quote
            if final_s != "Unknown":
                # Find the label for final_s to know which track to update
                entry = next((v for k, v in id_registry.items() if v['text'] == final_s), None)
                if entry:
                    if entry['label'] == 'PERSON':
                        last_person = [final_s, global_quote_idx]
                    elif entry['label'] in {'ORG', 'NORP'}:
                        last_org = [final_s, global_quote_idx]
                elif low_s in AI_KEYWORDS:
                    last_ai = [final_s, global_quote_idx]

        current_token_offset += (len(doc) - 100)

    # Deduplicate quotes across the whole article
    quote_map = {}
    for q, s in zip(temp_quotes, temp_speakers):
        if q not in quote_map:
            quote_map[q] = []
        quote_map[q].append(s)

    final_quotes = []
    final_speakers = []
    matched = {}

    # Find the best speaker

    for q_text, s_list in quote_map.items():
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
    print(f"{status_icon} RID {rid} | Quotes: {len(final_quotes)}")

    return final_quotes, final_speakers, matched, len(final_quotes)


# Run on a sample first to iterate on regex and cleaning and to debug

df_sample = df.iloc[11:15].copy()
df_sample[['quotes', 'speakers', 'attribution', 'quote_count']] = df_sample.progress_apply(
    lambda row: pd.Series(process_row(row)), axis=1
)

def unique_speakers(speakers):
    if isinstance(speakers, list):
        return list(set(speakers))
    return []

df_sample['unique_speakers'] = df_sample['speakers'].apply(unique_speakers)

pd.set_option('display.max_columns', None)
print(df_sample[['news_title', 'quotes', 'speakers', 'attribution', 'quote_count', 'unique_speakers']])

df_sample.to_csv("df_sample_6.csv", index=False)

#
# # Run on clean dataset and save results
#
# df[['quotes', 'speakers', 'attribution', 'quote_count']] = df['body_text'].progress_apply(lambda x: pd.Series(process_row(x)))
#
# all_speakers_ever = set([s for sublist in df['speakers'] for s in sublist])
#
# print(f"Total unique speakers identified in corpus: {len(all_speakers_ever)}")

# # # Save results
# df.to_csv("quotes_speakers_coref_2.csv", index=False)
# print("Finished! Results saved")
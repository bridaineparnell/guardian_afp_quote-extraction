# Import packages

import spacy
import pandas as pd
import re
nlp = spacy.load('en_core_web_trf')
nlp_light = spacy.load('en_core_web_md')
nlp.add_pipe('coreferee')
spacy.prefer_gpu()

## Commands to ensure cluster is correctly working

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

# Watch the progress
from tqdm.auto import tqdm
tqdm.pandas()

IDENTITY_LABELS = ['PERSON', 'ORG', 'NORP']
PRONOUNS = {'he', 'she', 'it', 'they', 'his', 'her', 'him', 'their', 'them', 'who', 'its'}
AI_KEYWORDS = {'chatgpt', 'bot', 'system', 'model', 'ai', 'algorithm', 'robot', 'chatbot', 'gemini', 'app'}
with open('utils/quote_verb_list.txt', 'r') as f:
    # Use lemmas (root words) for the best matching results
    ATTRIBUTION_VERBS = {line.strip().lower() for line in f if line.strip()}

## Break the text up
# so it can be processed at <512 tokens, but overlap so we don't miss bits

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

## Find all speakers in each article

def get_speakers(text):

    chunks = overlapping_chunks(text, chunk_size=400, overlap=100)
    s_registry = {}
    chain_counter = 0

    for chunk in chunks:
        doc = nlp(chunk)

        # Process coreference chains
        for chain in doc._.coref_chains:
            # Initialise everything we need
            names = []
            is_active = False
            ent_labels = []

            # Look at the mentions in a chain
            for mention in chain:
                for idx in mention.token_indexes:
                    token = doc[idx]

                    if token.pos_ == 'PROPN' and len(token.text) > 2:
                        names.append(token.text)

                    # Collect all NER labels in this chain
                    if token.ent_type_:
                        ent_labels.append(token.ent_type_)
                    elif token.text.lower() in AI_KEYWORDS:
                        ent_labels.append('AI')

                    # Obvious speaker if it's a noun subject that has one of the speaking verbs attached
                    if token.dep_ == "nsubj" and token.head.lemma_.lower() in ATTRIBUTION_VERBS:
                        is_active = True

            # If not a proper noun, ignore
            if not names:
                continue

            # Get the best name
            best_name = max(names, key=len)

            # Determine the majority label (e.g., if one mention is ORG, the whole chain is ORG)
            if ent_labels:
                # Pick the most frequent label that isn't empty
                ent_type = max(set(ent_labels), key=ent_labels.count)
            else:
                ent_type = "Unknown"

            # Only say 'Active' if it's a valid entity type
            status = "Passive"
            if is_active and (ent_type in IDENTITY_LABELS or ent_type == "AI"):
                status = "Active"

            s_registry[chain_counter] = {
                "name": best_name,
                "type": ent_type,
                "status": status,
                "mention_count": len(chain)
            }
            chain_counter += 1

    return s_registry

# Run on dataset
df = pd.read_csv("clean_main_dataset_6.csv")

speakers = df['body_text'].progress_apply(get_speakers)

rows = []

for article_idx, registry in speakers.items():
    title = df.loc[article_idx, 'news_title']
    rid = df.loc[article_idx, 'rid']

    for chain_id, info in registry.items():
        if info['status'] == 'Active':
            rows.append({"rid": rid,
                         "news_title": title,
                         "name": info['name'],
                         "type": info['type'],
                         "mentions": info['mention_count']
                         })

df_registry = pd.DataFrame(rows)

df_registry.to_csv('speaker_registry.csv', index=False)

## Get context for speakers

# First consolidate speakers

df_temp = df_registry.groupby(['rid', 'name'], as_index=False).agg({
    'mentions': 'sum',
    'type': 'first',
})

df_uniq = df_temp.groupby('rid').apply(
    lambda x: x[['name', 'type', 'mentions']].to_dict('records')
).reset_index().rename(columns={0: 'speakers'})

df_master = df.merge(df_uniq, on='rid', how='left')

df_master.to_csv('speaker_registry_uniq.csv', index=False)

# Now find their full context

def resolve_identity(row):
    # Safety check
    if not isinstance(row['speakers'], list):
        return []

    # Run spacy
    text = row['body_text']
    doc = nlp_light(text)

    # Unpack dictionary
    for speaker_dict in row['speakers']:
        name = speaker_dict['name']
        found_acronym = False

        # Find first mention of the word
        match = re.search(rf'\b{re.escape(name)}\b', text, re.IGNORECASE)
        if not match:
            speaker_dict['full_identity'] = name
            continue

        # Find token index so we can look ahead and behind
        start_char = match.start()
        target_token = None
        for token in doc:
            if token.idx >= start_char:
                target_token = token
                break
        if not target_token:
            speaker_dict['full_identity'] = name
            continue

        # Check if this name is an acronym inside parentheses
        # We look 1 token back for '(' and get the first letter as an anchor
        if target_token.i > 1 and doc[target_token.i - 1].text == "(" and name.isupper():
            first_letter = name[0].upper()
            # Search back up to 20 tokens for the first word starting with 'M'
            for i in range(target_token.i - 2, max(-1, target_token.i - 20), -1):
                if doc[i].text[0].upper() == first_letter:
                    long_name = doc[i:target_token.i - 1].text.strip()
                    # Clean up any leading 'and' or 'the' before returning
                    speaker_dict['full_identity'] = re.sub(r'^(and|the|of|for|in|&)\s+', '', long_name,
                                                           flags=re.IGNORECASE)
                    found_acronym = True
                    break
            if found_acronym: continue

        # Get anchor points to expand around the word
        s, e = target_token.i, target_token.i + 1

        # Expand left
        while s > 0:
            prev = doc[s - 1]
            # Keep going if Capitalized OR if it's a Noun/Adj title component or a connector
            if prev.text[0].isupper() or prev.pos_ in ['NOUN', 'ADJ'] or prev.text.lower() in ['of', 'the', 'and', '&']:
                if prev.text in ['.', '!', '?', ';']: break
                s -= 1
            else:
                break

        # Expand right
        while e < len(doc):
            nxt = doc[e]
            if nxt.text[0].isupper():
                e += 1
            elif nxt.text.lower() in ['and', 'of', '&'] and e < len(doc) - 1 and doc[e + 1].text[0].isupper():
                e += 1
            else:
                break

        # If preceded by a comma, look for titles + possessives
        is_comma = False
        if s > 1 and doc[s - 1].text == ",":
            title_start = s - 1
            while title_start > 0:
                p = doc[title_start - 1]
                if p.text[0].isupper() or p.pos_ in ['NOUN', 'ADJ', 'PART'] or p.text == ",":
                    if p.text in ['.', ';', '!', '?']: break
                    title_start -= 1
                else:
                    break

            if title_start < s - 1:
                full_id = doc[title_start:e].text.strip()
                # Clean up leading noise
                full_id = re.sub(r'^([Tt]he|[Aa]nd)\s+', '', full_id)
                speaker_dict['full_identity'] = full_id.strip(', ').replace(" 's", "'s")
                is_comma = True

        if is_comma: continue

        # Capture descriptions
        description = ""
        if e < len(doc) and doc[e].text == ",":
            desc_tokens = []
            for j in range(e + 1, min(len(doc), e + 15)):
                if doc[j].text in [".", ";", ":"] or (doc[j].text == "," and j > e + 5): break
                desc_tokens.append(doc[j].text_with_ws)
            desc_text = "".join(desc_tokens).strip()
            if desc_text.lower().startswith(('whose', 'who', 'a ', 'an ', 'the ')):
                description = desc_text

        # Final identity
        identity = doc[s:e].text.strip()
        identity = re.sub(r'^([Tt]he|[Aa]nd)\s+', '', identity)
        speaker_dict['full_identity'] = f"{identity} ({description})" if description else identity

    return row['speakers']

tqdm.pandas(desc="Resolving Identities")

# Run

# If you're reading a df back in, you need ast
# import ast
# df_master['speakers'] = df_master['speakers'].apply(
#     lambda x: ast.literal_eval(x) if isinstance(x, str) else x
# )

df_master['speakers'] = df_master.progress_apply(resolve_identity, axis=1)

def sort_speakers(speaker_list):
    if isinstance(speaker_list, list):
        return sorted(speaker_list, key=lambda x: x['mentions'], reverse=True)
    return speaker_list

df_master['speakers'] = df_master['speakers'].apply(sort_speakers)

df_master.to_csv("identity_registry.csv", index=False)




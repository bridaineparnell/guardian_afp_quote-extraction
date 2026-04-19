import pandas as pd
import ast
import spacy
import networkx as nx
from tqdm import tqdm

nlp_light = spacy.load('en_core_web_md')
AI_KEYWORDS = {'chatgpt', 'bot', 'system', 'model', 'ai', 'algorithm', 'robot', 'chatbot', 'gemini', 'app'}
IDENTITY_LABELS = ['PERSON', 'ORG', 'NORP']
STOP_ANCHORS = {'the', 'and', 'but', 'mr', 'ms', 'mrs', 'dr', 't', 's', 'royal', 'expert', 'new', 'prime', 'minister'}

df_gs = pd.read_csv('identity_registry.csv')

df_gs.rename(columns={'speakers': 'unique_speakers'}, inplace=True)

# For sampling
# df_gs = df_gs1.iloc[0:5].copy()

# print(df_gs)

df_qe = pd.read_csv('quotes_speakers_coref_3.csv')

# For sampling
# df_qe = df_qe1.iloc[0:5].copy()
#
# print(df_qe)

# Load tqdm to follow progress

tqdm.pandas(desc="Combining speakers")

## Combine the speakers using networkx and spacy

def combine_speakers(row, threshold=87):
    """
    Look at the unique speakers found with both speaker extraction methods
    Create a network of nodes that are similar using proper nouns and AI keywords as anchors
    If they are not in the 'ground truth' dictionary made by the get_speakers function,
    add them as a new dictionary, with mentions set to None
    """
    items_qe = row['unique_speakers_qe'] # This is a list
    items_gs = row['unique_speakers_gs'] # This is a list of dictionaries

    # When we load from CSV, pd sees everything as strings, this makes them lists and dictionaries again
    if isinstance(items_qe, str): items_qe = ast.literal_eval(items_qe)
    if isinstance(items_gs, str): items_gs = ast.literal_eval(items_gs)

    # # If it's a list, but the contents are still strings, evaluate the contents
    # if isinstance(items_gs, list) and len(items_gs) > 0:
    #     if isinstance(items_gs[0], str) and '{' in items_gs[0]:
    #         try:
    #             items_gs = [ast.literal_eval(i) if isinstance(i, str) else i for i in items_gs]
    #         except Exception as e:
    #             print(f"Failed to fix internal strings: {e}")

    # Make a memory bank of the metadata held in the df_gs dictionaries,
    # with the names as the 'label'
    gs_data = {d['name']: d for d in (items_gs or [])}

    # Make a master list and get rid of 'Unknowns'
    gs_names = list(gs_data.keys())
    raw_qe = [s for s in (items_qe or []) if s and str(s).lower() != 'unknown']

    combined = list(set(gs_names + raw_qe))

    # Identify anchor nouns for speakers
    anchors = {}
    # Start a list of valid speakers
    valid = []

    for s in combined:
        s_norm = str(s).strip()
        if len(s_norm) < 2:
            continue

        doc = nlp_light(s_norm)

        # Check if they're propn or AI using spacy, drop them if they've been misidentified
        cores = {t.text.lower() for t in doc
                 if (t.pos_ == "PROPN" or t.text.lower() in AI_KEYWORDS)
                 and t.text.lower() not in STOP_ANCHORS}

        if cores:
            anchors[s_norm] = cores
            valid.append(s_norm)

    # Start the graph using the networkx package
    G = nx.Graph()
    G.add_nodes_from(valid)

    # To build the graph, compare each item against the rest
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            s1, s2 = valid[i], valid[j]
            # If they share any Proper Nouns or AI Keywords, they are the same person
            if anchors[s1].intersection(anchors[s2]):
                G.add_edge(s1, s2)

    # Update df_gs with any new speakers from df_qe
    final_speakers =[]
    # Make lists of names in clusters
    for cluster in nx.connected_components(G):
        nodes = list(cluster)

        # Check if the cluster has a name that's already in df_gs,
        # if so, keep df_gs details
        gs_matches = [n for n in nodes if n in gs_data]

        if gs_matches:
            primary = sorted(gs_matches, key=lambda x: gs_data[x].get('mentions', 0), reverse=True)[0]
            orig = gs_data[primary]

            name = orig['name']
            s_type = orig['type']
            mentions = orig['mentions']
            # Keep df_gs full identity if it exists
            full_id = orig.get('full_identity', name)
            # Create aliases for additional info from df_qe
            aliases = [n for n in nodes if n != name and n != full_id]
        else:
            name = min(nodes, key=len)
            mentions = None
            full_id = max(nodes, key=len)

            temp_doc = nlp_light(full_id)
            s_type = 'PERSON' # default
            if temp_doc.ents:
                s_type = temp_doc.ents[0].label_ if temp_doc.ents[0].label_ in IDENTITY_LABELS else 'PERSON'

            aliases = [n for n in nodes if n != name and n != full_id]

        final_speakers.append({
            'name': name,
            'type': s_type,
            'mentions': mentions,
            'full_identity': full_id,
            'aliases': list(set(aliases))
        })

    # Sort: GS speakers first (by mentions), then new QE speakers
    final_speakers.sort(key=lambda x: (x['mentions'] is not None, x['mentions'] or 0), reverse=True)

    return pd.Series([final_speakers])

# Merge the two dataframes, keeping only the separate columns of unique speakers

df_merge = pd.merge(
    df_gs,
    df_qe[['rid', 'unique_speakers']],
    on='rid',
    suffixes=('_gs', '_qe')
)

# Apply the function to combine speakers

df_merge[['combined_speakers']] = df_merge.progress_apply(
    lambda row: combine_speakers(row, threshold=87), axis=1
)

# Save the results

df_merge.to_csv('speaker_merge.csv', index=False)

# Add attribution column for quotes

df_sm = pd.read_csv("speaker_merge.csv")

df_qc = pd.read_csv("quotes_speakers_coref_3.csv")

column = df_qc["attribution"]

df_merge = pd.concat([df_sm, column], axis=1)

df_merge.to_csv("speaker_merge_attribution.csv", index=False)
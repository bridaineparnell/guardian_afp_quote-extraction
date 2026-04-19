import ast
import wikipediaapi
import pandas as pd
import re
import time
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm
import requests

wiki = wikipediaapi.Wikipedia('en')
wiki._session.headers.update({
    'User-Agent': 'MySpeakerEnricher/1.0 (contact@example.com)'
})
sparql = SPARQLWrapper("http://dbpedia.org/sparql")
df = pd.read_csv('speaker_merge.csv', converters={'combined_speakers': ast.literal_eval})

# TEST
test_page = wiki.page("Joseph Weizenbaum")
print(f"Connection Test - Page Exists: {test_page.exists()}")
if test_page.exists():
    print(f"Summary: {test_page.summary[:100]}...")

# Restore the list of dictionaries (flattened by the CSV)
#df['combined_speakers'] = df['combined_speakers'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# For testing a sample
df_s = df.iloc[0:3].copy()

# Back-up function to id occupation from keyword search
def classify_industry(text):
    text_low = text.lower()

    # Define weight-based categories
    categories = {
        'AI & Tech': ['artificial intelligence', 'openai', 'software engineer', 'entrepreneur', 'silicon valley',
                      'technology'],
        'Academia': ['professor', 'university', 'researcher', 'scientist', 'ph.d', 'academic'],
        'Politics': ['prime minister', 'politician', 'member of parliament', 'senator', 'government', 'statesman'],
        'Royal Family': ['member of the british royal family', 'prince of', 'duke of', 'duchess of', 'monarch'],
        'Journalism & Media': ['journalist', 'author', 'writer', 'correspondent', 'biographer', 'columnist']
    }

    for industry, keywords in categories.items():
        if any(kw in text_low for kw in keywords):
            return industry

    return "Unknown"


def get_wiki_info(s_dict):
    """
    1. Finds the best Wikipedia match using prioritized, cleaned queries.
    2. Uses the resulting URL to query DBpedia for structured data.
    """
    # --- STEP 1: WIKIPEDIA CLEAN SEARCH ---
    name_only = str(s_dict.get('name', '')).strip()
    full_id = str(s_dict.get('full_identity', '')).strip()

    # Prioritize: 1. Name inside ID, 2. The Name field, 3. The Full ID string
    queries = []
    if full_id:
        clean_match = re.findall(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)', full_id)
        if clean_match:
            queries.append(clean_match[-1])

    if name_only: queries.append(name_only)
    if full_id: queries.append(full_id)

    wiki_para, wiki_url = "No biography found.", None
    seen = set()

    for q in queries:
        # Strip 'the' and possessives
        q_clean = re.sub(r"^[Tt]he\s+", "", q)
        q_clean = re.sub(r"['’]s\b", "", q_clean).strip()

        if q_clean in seen or len(q_clean) < 3: continue
        seen.add(q_clean)

        page = wiki.page(q_clean)
        try:
            if page.exists():
                paragraphs = page.summary.split('\n')
                wiki_para = paragraphs[0] if paragraphs else page.summary
                wiki_url = page.fullurl
                break  # Found a match, stop searching
        except Exception:
            continue

    # --- STEP 2: DBPEDIA SPARQL QUERY ---
    occupation, industry = "Unknown", "Other"

    if wiki_url:
        # Extract the entity ID from the URL (e.g., Sam_Altman)
        entity_id = wiki_url.split('/')[-1]
        resource_uri = f"http://dbpedia.org/resource/{entity_id}"

        # SPARQL Query for Types, Fields, and Occupations
        query = f"""
        SELECT DISTINCT ?type ?field ?occ WHERE {{
            <{resource_uri}> rdf:type ?type .
            OPTIONAL {{ <{resource_uri}> dbo:field ?field . }}
            OPTIONAL {{ <{resource_uri}> dbo:occupation ?occ . }}
            FILTER (strstarts(str(?type), "http://dbpedia.org/ontology/"))
        }}
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)

        try:
            results = sparql.query().convert()
            bindings = results["results"]["bindings"]

            # Aggregate all unique terms
            tags = set()
            for b in bindings:
                tags.add(b['type']['value'].split('/')[-1])
                if 'field' in b: tags.add(b['field']['value'].split('/')[-1])
                if 'occ' in b: tags.add(b['occ']['value'].split('/')[-1])

            # Remove generic boilerplate tags
            junk = {'Person', 'Agent', 'Thing', 'Species', 'Eukaryote', 'Animal', 'Work'}
            clean_tags = [t for t in tags if t not in junk]

            if clean_tags:
                occupation = ", ".join(clean_tags[:3])  # Limit to top 3
                # Synthesize industry using your keyword mapper on the para + tags
                industry = classify_industry(wiki_para + " " + occupation)
        except Exception as e:
            # If SPARQL fails, we still have the Wiki description
            print(f"SPARQL Error for {entity_id}: {e}")

    return {
        'description': wiki_para,
        'occupation': occupation,
        'industry': industry,
        'wiki_url': wiki_url
    }

# def get_wiki_info(speaker_dict):
#     """
#     Takes a speaker dict, finds Wiki para, then uses that to get DBpedia tags.
#     """
#     # Try searching the full_identity first, then the name
#     full_id = str(speaker_dict.get('full_identity', '')).strip()
#     name_only = str(speaker_dict.get('name', '')).strip()
#     entity_type = speaker_dict.get('type', '')
#
#     queries = []
#
#     if entity_type == "ORG" or "AI":
#         queries.append(name_only)
#     else:
#         clean_match = re.findall(r'([A-Z][a-z]+(?:\s[A-Z][.\s]?[A-Z][a-z]+)+)', full_id)
#         if clean_match:
#             queries.append(clean_match[-1])
#         queries.append(name_only)
#         queries.append(full_id)
#
#     wiki_para, wiki_url = "No bio found.", None
#     seen = set()
#
#     for q in queries:
#         # Clean the query
#         q_clean = re.sub(r"^[Tt]he\s+", "", q)
#         q_clean = re.sub(r"['’]s\b", "", q_clean).strip()
#
#         if q_clean in seen or len(q_clean) < 3:
#             continue
#         seen.add(q_clean)
#
#         page = wiki.page(q_clean)
#         try:
#             if page.exists():
#                 # We take the first para
#                 paragraphs = page.summary.split('\n')
#                 wiki_para = paragraphs[0] if paragraphs else page.summary
#                 wiki_url = page.fullurl
#                 break
#         except: continue
#
#     occupation, industry = "Unknown", "Unknown"
#     if wiki_url:
#         # Extract DBpedia identifier
#         entity_id = wiki_url.split('/')[-1]
#         resource_uri = f"https://dbpedia.org/resource/{entity_id}"
#
#         query = f"""
#             SELECT DISTINCT ?val WHERE {{
#                 {{ <{resource_uri}> rdf:type ?type . BIND(?type AS ?val) }}
#                 UNION
#                 {{ <{resource_uri}> dbo:occupation ?occ . BIND(?occ AS ?val) }}
#                 UNION
#                 {{ <{resource_uri}> dbo:field ?field . BIND(?field AS ?val) }}
#                 UNION
#                 {{ <{resource_uri}> dbo:industry ?ind . BIND(?ind AS ?val) }}
#                 UNION
#                 {{ <{resource_uri}> dbo:purpose ?purp . BIND(?purp AS ?val) }}
#                 UNION
#                 {{ <{resource_uri}> <http://purl.org/linguistics/gold/hypernym> ?hyp . BIND(?hyp AS ?val) }}
#                 FILTER (strstarts(str(?val), "http://dbpedia.org/ontology/"))
#             }}
#             """
#         sparql.setQuery(query)
#         sparql.setReturnFormat(JSON)
#
#         try:
#             results = sparql.query().convert()
#             bindings = results["results"]["bindings"]
#
#             # Collect all unique ontology terms
#             tags = set()
#             for b in bindings:
#                 tag = b['type']['value'].split('/')[-1]
#                 if tag not in ['Person', 'Agent', 'Thing', 'Species', 'Eukaryote', 'Animal', 'Work', 'Organisation']:
#                     tags.add(tag)
#
#             if tags:
#                 occupation = ", ".join(tags[:3])  # Take top 3 for brevity
#                 # Use keyword logic as a fallback for 'Industry' mapping
#                 industry = classify_industry(wiki_para + " " + occupation)
#         except Exception as e:
#             print(f"SPARQL Error for {entity_id}: {e}")
#
#     return {
#         'description': wiki_para,
#         'occupation': occupation,
#         'industry': industry,
#         'wiki_url': wiki_url
#     }

# Function to run over the dataset
def add_wiki_info(df):
    unique_entities = {}
    for row_list in df['combined_speakers']:
        for s in row_list:
            if s['full_identity'] not in unique_entities:
                unique_entities[s['full_identity']] = s

    print(f"Getting info for {len(unique_entities)} unique speakers...")

    for name, s_dict in tqdm(unique_entities.items()):
        data = get_wiki_info(s_dict)
        s_dict.update(data)
        # For the APIs
        time.sleep(0.2)

    return df

wiki_df = add_wiki_info(df_s)
wiki_df.to_csv('wiki_enriched_s.csv', index=False)
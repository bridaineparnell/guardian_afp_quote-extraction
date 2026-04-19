import pandas as pd
import ast
from thefuzz import fuzz
import re

AI_KEYWORDS = {'chatgpt', 'bot', 'system', 'model', 'ai', 'algorithm', 'robot', 'chatbot', 'gemini', 'app'}

df_as = pd.read_csv("annotation_sample.csv")

df = pd.read_csv("speaker_merge_attribution.csv")

df_sm = df.loc[df['rid'].isin([17, 27, 37, 47, 57, 67, 77, 87, 97, 107])]
df_sm['combined_speakers'] = df_sm['combined_speakers'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

#print(df_sm)

# Clean for identity matching

def clean_identity_logic(text):
    if not isinstance(text, str) or text.lower() == 'nan':
        return ""

    words = re.findall(r'\w+', text)
    filtered_words = []

    for word in words:
        # If the word is an AI keyword, rename it
        if word.lower() in AI_KEYWORDS:
            filtered_words.append("AI_ENTITY")
        elif word[0].isupper():
            filtered_words.append(word.upper())

    return " ".join(filtered_words)

def clean_list(entity_list):
    # Apply logic to each item and deduplicate
    cleaned = [clean_identity_logic(e) for e in entity_list]
    return list(set([e for e in cleaned if e]))  # Remove empty strings

# Group the separate sources into a list for each RID

df_as_grouped = df_as.groupby('rid').agg({
    'source': lambda x: list(x.dropna()),
    'descriptors': lambda x: list(x.dropna())
}).reset_index()

# Clean the manual annotation
df_as_grouped['clean_source'] = df_as_grouped['source'].apply(clean_list)
df_as_grouped['clean_desc'] = df_as_grouped['descriptors'].apply(clean_list)

# Extract the source from the computational dictionary
df_sm['comp_source'] = df_sm['combined_speakers'].apply(
    lambda x: list(set([d.get('name') for d in x])) if isinstance(x, list) else []
)

# Clean
df_sm['clean_comp'] = df_sm['comp_source'].apply(clean_list)

# Merge for comparison
comparison = pd.merge(
    df_as_grouped[['rid', 'clean_source', 'clean_desc']],
    df_sm[['rid', 'clean_comp']],
    on='rid',
    how='outer'
)

# Fill NaNs with empty lists
for col in ['clean_source', 'clean_desc', 'clean_comp']:
    comparison[col] = comparison[col].apply(lambda x: x if isinstance(x, list) else [])

# Compare and score
def calculate_weighted_metrics(row):
    h_sources = row['clean_source']
    h_descs = row['clean_desc']
    c_sources = row['clean_comp']

    # Case 1: Both are empty (True Negative) - correctly identified no speakers
    if not h_sources and not c_sources:
        return 1.0, 1.0

    # Case 2: Human found sources, but code found nothing (Total Miss)
    if not c_sources and h_sources:
        return 0.0, 0.0

    # Case 3: Code found something, but human found nothing (Total Hallucination)
    if c_sources and not h_sources:
        return 0.0, 0.0

    tp_score = 0
    threshold = 80 # Quite high

    for extracted in c_sources:
        # Step 1: Check against names (1.0 points)
        name_scores = [fuzz.partial_ratio(str(extracted), str(h)) for h in h_sources]
        if max(name_scores + [0]) >= threshold:
            tp_score += 1.0
            continue

            # Step 2: Check against descriptors (0.7 points)
        desc_scores = [fuzz.partial_ratio(str(extracted), str(d)) for d in h_descs]
        if max(desc_scores + [0]) >= threshold:
            tp_score += 0.7

    precision = tp_score / len(c_sources)
    recall = tp_score / len(h_sources)

    # Cap precision and recall at 1.0 in case partial points overlap weirdly
    return min(precision, 1.0), min(recall, 1.0)


# Apply to your grouped dataframe
comparison[['precision', 'recall']] = comparison.apply(
    lambda r: pd.Series(calculate_weighted_metrics(r)), axis=1
)

pd.set_option('display.max_columns', None)
#print(comparison)

comparison.to_csv("comparison_sample.csv", index=False)

# Aggregate the scores
final_precision = comparison['precision'].mean()
final_recall = comparison['recall'].mean()

# Calculate the F1 Score (Harmonic mean of Precision and Recall)
if (final_precision + final_recall) > 0:
    final_f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall)
else:
    final_f1 = 0

print(f"--- Model Performance Report ---")
print(f"Weighted Precision: {final_precision:.2%}")
print(f"Weighted Recall:    {final_recall:.2%}")
print(f"Weighted F1 Score:  {final_f1:.2%}")


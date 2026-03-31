# Import packages
import pandas as pd
import re

df = pd.read_csv("main_dataset.csv")

# First clean the text

def text_repair(text):
    """
    Standardise the way quotes look across the set
    and sort out text encoding issues
    """
    # Safety code
    if not isinstance(text, str): return ""
    # Monitor the progress
    stats = {"mojibake_fixes": 0, "quote_normalisations": 0}

    # Mojibake repair (Priority order matters)
    # The 'â€' often acts as a prefix for multiple types of marks
    mojibake_map = {
        "â€œ": '"', "â€": '"', "â€\x9d": '"', "â€": '"',
        "â€™": "'", "â€”": "—", "â€“": "–", "Â": ""
    }

    for bad, good in mojibake_map.items():
        count = text.count(bad)
        if count > 0:
            text = text.replace(bad, good)
            stats["mojibake_fixes"] += count

    # Quote Normalization

    # Force all variations to straight quotes
    text, sq_count = re.subn(r'[“”„‟«»]', '"', text)
    text, s_count = re.subn(r"[‘’'‛′]", "'", text)
    stats["quote_normalisations"] = sq_count + s_count

    # Single to double quotes
    text, so_count = re.subn(r"(^|[\s:;—\(\[])'([a-zA-Z0-9])", r'\1"\2', text)  # Opening
    text, sc_count = re.subn(r"([^s])'([\s\.,!?;]|$)", r'\1"\2', text) # Closing
    stats["quote_normalisations"] = so_count + sc_count

    # Collapse whitespaces
    text, ws_count = re.subn(r'[ \t]+', ' ', text)
    stats["quote_normalisations"] += ws_count

    return text.strip(), stats

report = {"processed_rows": 0, "mojibake": 0, "quotes": 0}

def clean_and_track(text):
      """
      Function to iterate through and clean the text
      while tracking the progress
      """

      # Safety code
      if not isinstance(text, str) or len(text.strip()) == 0:
        return text

      # Run the cleanup and gather the stats
      cleaned, stats = text_repair(text)

      # Update the report tracker
      report["processed_rows"] += 1
      report["mojibake"] += stats.get("mojibake_fixes", 0)
      report["quotes"] += stats.get("quote_normalisations", 0)

      return cleaned

# check

# text = """
#  " It could perhaps in time create a song that is, on the surface, indistinguishable from an original,
#  but it will always be a replication, a kind of burlesque,"he wrote in his The Red Hand Files newsletter."
#  Songs arise out of suffering, by which I mean they are predicated upon the complex,
#  internal human struggle of creation and, well, as far as I know, algorithms don't feel.
#  Data doesn't suffer. "The apocalypse is well on its way. This song sucks."
#  """
#
# cleaned = clean_and_track(text)
#
# print(cleaned)

# Run on the df
print("Starting cleanup...")
df['body_text'] = df['body_text'].apply(clean_and_track)

# Report results of cleanup
print("\n" + "="*40)
print("FINAL CLEANUP REPORT")
print("="*40)
print(f"Total Rows Processed:   {report['processed_rows']}")
print(f"Mojibake Fixed:        {report['mojibake']}")
print(f"Quote Normalisations:  {report['quotes']}")
print("="*40)

# Save the cleaned text csv

df.to_csv("clean_main_dataset_6.csv", index=False, quoting=1)
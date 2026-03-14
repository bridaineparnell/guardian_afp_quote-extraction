import logging
import re

from utils.classes import Quote
from utils.preprocessing import sentencise_text, get_quote_indices, uniq
from utils.functions_spacy3 import get_complete_ents_list


########################################################
## Regex definitions and quote verb list
########################################################

with open('utils/quote_verb_list.txt', 'r') as f:
    quote_verbs = [(line.strip()) for line in f]

quote_verb_boolean_list = [quote + "|" for quote in quote_verbs]
quote_verb_boolean_list = [quote + "|" for quote in quote_verbs if quote[-1:] in ['d', 'g', 's']]
quote_verb_boolean_string = ''.join(quote_verb_boolean_list)
quote_verb_boolean_string = quote_verb_boolean_string[:-1]

# All of this regex changed from the original after iterations
any_quote = r'“[^”]+?”'

# 1. Someone Said: [Quote] [Speaker] [Verb]
re_quote_someone_said = r'({quote})\s*[,]?\s*((?:\w+\s+){{1,5}})({cue_verbs})'.format(
    quote=any_quote,
    cue_verbs=quote_verb_boolean_string)

# 2. Said Someone: [Quote] [Verb] [Speaker]
re_quote_said_someone = r'({quote})\s+({cue_verbs})\s+((?:\w+\s+){{1,5}})'.format(
    quote=any_quote,
    cue_verbs=quote_verb_boolean_string)

# 3. Someone Told Someone: [Speaker] [Verb] [Quote]
re_quote_someone_told_someone = r'({quote})\s+([^\.!?“"‘\']+?)\s+({cue_verbs})\s+([^\.!?“"‘\']+?)'.format(
    quote=any_quote,
    cue_verbs=quote_verb_boolean_string)

# 4. Colon Style: [Speaker] [Verb] : [Quote]
re_quote_someone_said_colon = r'([^“"‘\'\n\.!?]+?)\s+({cue_verbs})(?:\s+\w+){{0,5}}\s*:\s*({quote})'.format(
    quote=any_quote,
    cue_verbs=quote_verb_boolean_string)

# 5. Adding Colon: [Speaker] [Verb] adding : [Quote]
re_quote_someone_said_adding_colon = r'([^“"‘\'\n\.!?]+?)\s+({cue_verbs})\s+adding\s*:\s*({quote})'.format(
    quote=any_quote,
    cue_verbs=quote_verb_boolean_string)

# These patterns added
# 6. Floating Name : [Speaker] : [Quote]
re_speaker_colon_quote = r'([A-Z][a-z]+(?:\s[A-Z][a-z]+){{0,2}})\s*:\s*({quote})'.format(
    quote=any_quote)

# 7. Transcript Style: [Speaker]: [Text] (No quotation marks required)
# This looks for a Capitalized Name at the start of a line/sentence followed by a colon
re_transcript_style = r'(?:^|\n)([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2})\s*:\s*([^“"‘\'\n\.!?]+)'


# Define the helper variables to keep the rest of the script happy
between_quotes = any_quote
between_quotes_sentence_start = r'^' + any_quote
between_quotes_ends_with_comma = r'(?:“[^”\n]+?,”|"[^"\n]+?,"|‘[^’\n]+?,\’|\'[^\'\n]+?,\')'

# # This was the original regex before alterations and added pattern
# re_quote_someone_said = \
#     r'(“[^“\n]+?[,?!]”) ([^\.!?]+?)[\n ]({cue_verbs})([^\.!?]*?)[\.,][\n ]{{0,2}}(“[\w\W]+?”){{0,1}}'.format(
#     cue_verbs= quote_verb_boolean_string)
# re_quote_said_someone = '(“[^“\n]+?[,?!]”)[\n ]({cue_verbs}) ([^\.!?]+?)[\.,](\s{{0,2}}“[^”]+?”){{0,1}}'.format(
#     cue_verbs=quote_verb_boolean_string)
# re_quote_someone_told_someone = \
#     '(“[^“\n]+?[,?!]”)[\n ]([^\.!?]*?) ({cue_verbs}) ([^\.!?]*?)[\.,][\n ]{{0,2}}(“[\w\W]+?”){{0,1}}'.format(
#     cue_verbs=quote_verb_boolean_string)
# re_quote_someone_said_colon = \
#     '([^“\n]+?) ({cue_verbs})( \w*?){{0,5}}: (“[\w\W]+?”){{1,1}}'.format(
#     cue_verbs=quote_verb_boolean_string)
# re_quote_someone_said_adding_colon = \
#     '([\w\W]+?) (“[\w\W]+?”)([-–\’\s,\w]*?) (adding)( \w*?){0,5}: (“[\w\W]+?”){1,1}'
#
# between_quotes = '“[^“”,]+?”'
# between_quotes_sentence_start = '$“[^“”]+?”'
# between_quotes_ends_with_comma = '“[^“”]+?,”'

QUOTE_TYPES = {
    'someone_said': 1,
    'said_someone': 2,
    'someone_told_someone': 3,
    'someone_said_colon': 4,
    'someone_said_adding_colon': 5,
    'speaker_colon_quote': 6, # This added
    'transcript_style': 7 # This added
}
# Update these inside utils/quote_extraction.py
# These patterns updated
QUOTE_TYPES_PATTERNS = {
    1: {'quote_text': 0, 'speaker': 1}, # re_quote_someone_said
    2: {'quote_text': 0, 'speaker': 2}, # re_quote_said_someone
    3: {'quote_text': 0, 'speaker': 1}, # re_quote_someone_told_someone
    4: {'quote_text': 2, 'speaker': 0}, # re_quote_someone_said_colon
    5: {'quote_text': 2, 'speaker': 0},  # re_quote_someone_said_adding_colon
    6: {'quote_text': 1, 'speaker': 0}, # This added
    7: {'quote_text': 1, 'speaker': 0} # This added
}
# These were the original patterns
# QUOTE_TYPES_PATTERNS = {
#     1: {'quote_text': 0, 'speaker': 1, 'quote_text_optional_second_part': -1},
#     2: {'quote_text': 0, 'speaker': 2, 'quote_text_optional_second_part': 3},
#     3: {'quote_text': 0, 'speaker': 1, 'quote_text_optional_second_part': 4},
#     4: {'quote_text': 3, 'speaker': 0},
#     5: {'quote_text': 0, 'quote_text_optional_second_part': 1, 'additional_cue': 3, 'quote_text_optional_third_part': 4}
# }

########################################################
## Function definitions
########################################################

def parse_sentence_quotes(sents, nlp_model, debug=False):
    """ Takes a list of sentences of the article and parses out quotes.
        Uses spacy's dependency parser:
        1) It replaces everything between quotes with a dummy phrase (to simplify the
        structure of the sentence that spacy needs to parse).
        2) It then checks each token. If a quote verb appears as the verb of a sentence, it finds the nsubj of the
           sentence, collects the subtree of the nsubj and stores them as the speaker.

        TO BE IMPROVED:
        For sentences that have two quotes within them, this process becomes quite difficult - current approach
        is to split the sentence in two at the end of the first quote but this won't work for all sentences.

        :param sents: the pre-processed text of an article split up into sentences
        :param nlp: spacy model

        returns: a list of sentence_parse_quotes:
                [quote_text, speaker (if possible), quote_verb, sent_index, start_index, end_index]
                start_index and end_index are for the quote text and relative to the sentence.
        """
    assert type(sents) == list

    sentence_parse_quotes = []
    for sent_index in range(len(sents)):
        sent = sents[sent_index]

        sentence_quote_indices = get_quote_indices(sent)
        if debug:
            logging.debug(sentence_quote_indices)
        if len(sentence_quote_indices) == 0:
            pass

        else:
            if sent[0] == '“':
                if sent.find(',”'):
                    modified_sent = re.sub(between_quotes_sentence_start, '“Dummy phrase,”', sent)
            else:
                if sent.find(',”'):
                    modified_sent = re.sub(between_quotes_ends_with_comma, '“dummy phrase,”', sent)
                else:
                    modified_sent = re.sub(between_quotes, '“dummy phrase”', sent)

            m_doc = nlp_model(modified_sent)
            logging.debug(sent)
            logging.debug(modified_sent)
            m_sentence_quote_indices = get_quote_indices(modified_sent)

        if len(sentence_quote_indices) == 1:
            for start_index, end_index in sentence_quote_indices:
                m_start_index, m_end_index = m_sentence_quote_indices[0]

                quote_text = sent[start_index:end_index + 1]

                for tok in m_doc:
                    if ((tok.head.idx < m_start_index or tok.head.idx > m_end_index) and
                            (tok.idx < m_start_index or tok.idx > m_end_index) and
                            tok.dep_ == 'nsubj' and tok.head.pos_ == 'VERB' and tok.head.text in quote_verbs
                    ):
                        subtree = [t for t in tok.subtree]
                        idxes = [t.idx for t in subtree]
                        speaker = modified_sent[idxes[0]:idxes[-1] + len(subtree[-1])]
                        speaker = speaker.replace('“', '').replace('”', '').replace('dummy phrase', '').replace(
                            'Dummy phrase', '').strip()
                        logging.debug(sent)
                        logging.debug(modified_sent)
                        logging.debug(speaker)

                        if speaker in ('He', 'She'):
                            speaker = speaker.lower()
                        quote_verb = tok.head.text
                        sentence_parse_quotes.append(
                            [quote_text, speaker, quote_verb, sent_index, start_index, end_index])
                        break

                try:
                    if quote_text != sentence_parse_quotes[-1][0] and quote_text[0] != '“' and quote_text[-1] != '”':
                        for tok in m_doc:
                            if (tok.head.pos_ == 'VERB' and tok.head.text in quote_verbs
                            ):
                                quote_verb = tok.head.text
                                sentence_parse_quotes.append(
                                    [quote_text, '', quote_verb, sent_index, start_index, end_index])
                                break

                except IndexError:
                    pass

        # deal with sentences with two quotes in by splitting the sentence in two
        elif len(sentence_quote_indices) == 2:

            first_quote_indices = sentence_quote_indices[0]
            second_quote_indices = sentence_quote_indices[1]
            end_of_first_quote = first_quote_indices[1] + 1

            m_sentence_quote_indices = get_quote_indices(modified_sent)
            if debug:
                logging.debug(m_sentence_quote_indices)
                logging.debug('sent: ', sent)
                logging.debug('modified_sent: ', modified_sent)
            if len(m_sentence_quote_indices) < 2:
                if debug:
                    logging.debug("Skipping sentence: missing matching quotation marks.")
                continue
            m_first_quote_indices = m_sentence_quote_indices[0]
            m_second_quote_indices = m_sentence_quote_indices[1]
            m_end_of_first_quote = m_first_quote_indices[1] + 1

            quote_and_index_list = []
            for start_index, end_index in sentence_quote_indices:
                quote_text = sent[start_index:end_index + 1]
                quote_and_index_list.append([quote_text, start_index, end_index])

            for quote_text, start_index, end_index in quote_and_index_list:
                for tok in m_doc:
                    if start_index == first_quote_indices[0]:
                        m_start_index, m_end_index = m_first_quote_indices

                        if ((tok.head.idx < m_start_index or tok.head.idx > m_end_index) and
                                (tok.idx < m_start_index or tok.idx > m_end_index) and
                                tok.dep_ == 'nsubj' and tok.head.pos_ == 'VERB' and tok.head.text in quote_verbs and
                                tok.idx < m_end_of_first_quote
                        ):
                            subtree = [t for t in tok.subtree]
                            idxes = [t.idx for t in subtree]
                            speaker = modified_sent[idxes[0]:idxes[-1] + len(subtree[-1])]
                            speaker = speaker.replace('“', '').replace('”', '').replace('dummy phrase', '').replace(
                                'Dummy phrase', '').strip()

                            if speaker in ('He', 'She'):
                                speaker = speaker.lower()
                            quote_verb = tok.head.text
                            sentence_parse_quotes.append(
                                [quote_text, speaker, quote_verb, sent_index, start_index, end_index])
                            break
                    elif start_index == second_quote_indices[0]:
                        m_start_index, m_end_index = m_second_quote_indices
                        #                         logging.debug(m_start_index, m_end_index, tok, tok.idx, tok.dep_, 'HEAD:', tok.head, tok.head.idx, tok.head.pos_)
                        if ((tok.head.idx < m_start_index or tok.head.idx > m_end_index) and
                                (tok.idx < m_start_index or tok.idx > m_end_index) and
                                tok.dep_ == 'nsubj' and tok.head.pos_ == 'VERB' and tok.head.text in quote_verbs and
                                tok.idx >= m_end_of_first_quote
                        ):
                            subtree = [t for t in tok.subtree]
                            idxes = [t.idx for t in subtree]
                            speaker = modified_sent[idxes[0]:idxes[-1] + len(subtree[-1])]
                            speaker = speaker.replace('“', '').replace('”', '').strip()

                            if speaker in ('He', 'She'):
                                speaker = speaker.lower()
                            quote_verb = tok.head.text
                            sentence_parse_quotes.append(
                                [quote_text, speaker, quote_verb, sent_index, start_index, end_index])
                            break

                try:
                    if quote_text != sentence_parse_quotes[-1][0] and quote_text[0] != '“' and quote_text[-1] != '”':
                        for tok in m_doc:
                            if (tok.head.pos_ == 'VERB' and tok.head.text in quote_verbs
                            ):
                                quote_verb = tok.head.text
                                sentence_parse_quotes.append(
                                    [quote_text, '', quote_verb, sent_index, start_index, end_index])
                                break
                except IndexError:
                    pass

    return sentence_parse_quotes


def parse_quote(list_, quote_pattern):
    if quote_pattern is None:
        raise ValueError(f"Incorrect quote pattern provided for '{list_}'")
    return Quote(**dict((k, list_[quote_pattern[k]]) for k in quote_pattern))


def parse_regex_matches(matches, quote_type):
    results = []
    for match in matches:
        # clean_match = filter(None, match)
        quote = parse_quote(match, QUOTE_TYPES_PATTERNS.get(quote_type))
        quote.QUOTE_TYPE = quote_type
        results.append(quote)
    return results


def extract_quotes_sentence_regex(pattern, text):
    """ Extract matching groups and sentences from `text` based on regex `pattern` provided.
        Returns: list(tuple), list(str) – matched groups and sentences
    """
    groups = []
    sentences = []
    for match in re.finditer(pattern, text):
        groups.append(match.groups())
        sentences.append(text[match.start():match.end()])
    return groups, sentences

# This code altered to adapt to new patterns
def extract_quotes_and_sentence_speaker(text, nlp_model, debug=False):
    """ Takes the pre-processed text of an article and returns a dictionary of attributed quotes,
        unattributed_quotes and quote marks only (everything else between quotes)

        Uses: 
        1) The regular expressions defined above to capture well-defined quotes
        2) The parse_sentence_quotes function to capture quote fragments
        3) Finds orphan quotes - those that are whole paragraphs in quote marks - and attributes them to the named
           entity in the previous sentence (if available)
        
        For the regular expression quotes, if the sentence is also parsed well, it replaces the speaker from the 
        regex quote with that from the sentence parsing because it includes less noise.

        :param sents: the pre-processed text of an article split up into sentences
        :param nlp: spacy model
        
        returns: a dictionary of quotes:
                {'attributed_quotes': those that can be given a speaker
                  ,'unattributed_quotes': those that can't be given a speaker
                  ,'just_quote_marks': Everything else. This will include quote fragments that can't be parsed 
                                       correctly, so wrongly missed quotes are usually in here.
                  }
        """

    sentences = sentencise_text(text)
    if len(sentences) == 0:
        logging.warning(f"Cannot sentencise '{sentences}'")
        quotes_dict = {'attributed_quotes': [],
                       'unattributed_quotes': [],
                       'just_quote_marks': []
                       }
        return quotes_dict

    all_regex_quotes = {}
    all_regex_sentences = {}

    # New code
    patterns = [
        ('someone_said', re_quote_someone_said),
        ('said_someone', re_quote_said_someone),
        ('someone_told_someone', re_quote_someone_told_someone),
        ('someone_said_colon', re_quote_someone_said_colon),
        ('someone_said_adding_colon', re_quote_someone_said_adding_colon),
        ('speaker_colon_quote', re_speaker_colon_quote),
        ('transcript_style', re_transcript_style)
    ]

    for qt_name, pattern in patterns:
        all_regex_quotes[qt_name], all_regex_sentences[qt_name] = extract_quotes_sentence_regex(pattern, text)

    # Old code here
    # all_regex_quotes['someone_said'], all_regex_sentences['someone_said'] = extract_quotes_sentence_regex(re_quote_someone_said, text)
    # all_regex_quotes['said_someone'], all_regex_sentences['said_someone'] = extract_quotes_sentence_regex(re_quote_said_someone, text)
    # all_regex_quotes['someone_told_someone'],  all_regex_sentences['someone_told_someone'] = extract_quotes_sentence_regex(re_quote_someone_told_someone, text)
    # all_regex_quotes['someone_said_colon'],  all_regex_sentences['someone_said_colon'] = extract_quotes_sentence_regex(re_quote_someone_said_colon, text)
    article_quote_indices = get_quote_indices(text)
    article_quote_texts = [text[quote_pair[0]:quote_pair[1] + 1] for quote_pair in article_quote_indices]

    if debug:
        logging.debug('someone_said:', all_regex_quotes['someone_said'])
        logging.debug('said_someones:', all_regex_quotes['said_someone'])
        logging.debug('someone_told_someones:', all_regex_quotes['someone_told_someone'])
        logging.debug('someone_said_colons:', all_regex_quotes['someone_said_colon'])

    # Parse the sentence out using spacy dependency and attribute using that
    sentence_parse_quotes = parse_sentence_quotes(sentences, nlp_model)

    # Orphan quotes: quotes that are entire paragraphs that follow on from a non-quote sentence
    # This code updated
    orphan_quotes = []
    for quote in article_quote_texts:
        for sent_index, sent in enumerate(sentences):
            if quote.strip() == sent.strip():
                # Look at previous sentence for a potential speaker
                if sent_index > 0:
                    previous_sent = sentences[sent_index - 1]
                    sent_ents = get_complete_ents_list(previous_sent, nlp_model)

                    # Try to find a Verb + Subject link
                    doc = nlp_model(previous_sent)
                    found = False
                    for tok in doc:
                        if (tok.dep_ == 'nsubj' and tok.head.pos_ == 'VERB' and tok.head.text in quote_verbs):
                            subtree = [t for t in tok.subtree]
                            idxes = [t.idx for t in subtree]
                            # Extract the raw speaker (He/She/Name)
                            speaker = previous_sent[idxes[0] - doc[0].idx: idxes[-1] - doc[0].idx + len(subtree[-1])]
                            orphan_quotes.append([quote, speaker.strip(), tok.head.text, sent_index, sent_ents])
                            found = True
                            break

                    if not found:
                        # Fallback: Just provide the quote and the entities from the prev sentence
                        orphan_quotes.append([quote, '', '', sent_index, sent_ents])

    regex_quotes = []
    for qt_name, matches in all_regex_quotes.items():
        if matches:
            # Uses the mapping dict we defined earlier to find which group is the speaker
            type_id = QUOTE_TYPES.get(qt_name)
            regex_quotes.extend(parse_regex_matches(matches, type_id))

    # Old code here
    # orphan_quotes = []
    # for quote in article_quote_texts:
    #
    #     for sent_index in range(len(sentences)):
    #         sent = sentences[sent_index]
    #         if quote == sent:
    #             previous_sent = sentences[sent_index - 1]
    #             sent_ents = get_complete_ents_list(previous_sent, nlp_model)
    #             if '“' not in previous_sent and '”' not in previous_sent:
    #                 doc = nlp_model(previous_sent)
    #                 found = False
    #                 for tok in doc:
    #                     if (tok.dep_ == 'nsubj' and tok.head.pos_ == 'VERB' and tok.head.text in quote_verbs):
    #                         subtree = [t for t in tok.subtree]
    #                         idxes = [t.idx for t in subtree]
    #                         speaker = previous_sent[idxes[0]:idxes[-1] + len(subtree[-1])]
    #                         if speaker in ('He', 'She'):
    #                             speaker = speaker.lower()
    #                         quote_verb = tok.head.text
    #                         orphan_quotes.append([quote, speaker, quote_verb, sent_index, sent_ents])
    #                         found = True
    #                         break
    #                 if found == False:
    #                     orphan_quotes.append([quote, '', '', sent_index, sent_ents])


    # # Parse regex quotes
    # regex_quotes = []
    # for qt_name, matches in all_regex_quotes.items():
    #     regex_quotes.extend(parse_regex_matches(matches, QUOTE_TYPES.get(qt_name)))

    if debug:
        logging.debug('sentence_parse_quotes:')
        logging.debug(sentence_parse_quotes)

        logging.debug('regex_quotes:')
        logging.debug(regex_quotes)

    # extra_adding_regex_quotes = []
    # for sentence in sentences:
    #     if len(get_quote_indices(sentence)) > 1:
    #         extra_adding_regex_quote = re.findall(re_quote_someone_said_adding_colon, sentence)
    #         if len(extra_adding_regex_quote) > 0: extra_adding_regex_quotes.extend(parse_regex_matches(extra_adding_regex_quote, QUOTE_TYPES['someone_said_adding_colon']))
    #
    # logging.debug('extra_adding_regex_quotes:')
    # logging.debug(extra_adding_regex_quotes)

    logging.debug('final regex_quotes:')
    logging.debug(regex_quotes)

    # New code here - Final Cleanup and Deduplication
    combined_quotes = regex_quotes + orphan_quotes + sentence_parse_quotes

    # Flatten sentences for the return
    regex_sentences = [s for sublist in all_regex_sentences.values() for s in sublist]

    return combined_quotes, list(set(regex_sentences))

    # Old code - Return quotes and sentences after removing duplicates
    regex_sentences = [_ for sublist in all_regex_sentences.values() for _ in sublist]
    return list(set(regex_quotes)) + list(set(extra_adding_regex_quotes)) + orphan_quotes, list(set(regex_sentences))

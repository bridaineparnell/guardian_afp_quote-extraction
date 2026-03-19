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

# Rescue broken paragraphs that end with " but don't start with one
    paragraphs = text.split('\n')
    healed_paragraphs = []
    for para in paragraphs:
        p = para.strip()
        if not p:
            healed_paragraphs.append("")
            continue

        # Add a start quote mark if you find a closing one
        if re.search(r'[\.!\?]"$', p) and not re.match(r'^[*•\-\s]*"', p):
            p = '"' + p

        # Add a closing quote mark if you find an opening one
        if re.match(r'^[*•\-\s]*"', p) and not re.search(r'"$', p):
            if p[-1] in ['.', '!', '?']:
                p = p + '"'
            else:
                p = p + '."'

        healed_paragraphs.append(p)

    text = '\n'.join(healed_paragraphs)
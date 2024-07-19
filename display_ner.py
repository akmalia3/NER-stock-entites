def annotated_text(*args):
    """This function is a placeholder to mimic the output format. It does not perform any real operation."""
    return args

# Function to convert tokens to annotated text and add spaces
def convert_to_annotated_text(tokens):
    annotated_parts = []
    for token, label in tokens:
        if label == 'O':
            annotated_parts.append(token + ' ')
        else:
            annotated_parts.append((token, label))
            annotated_parts.append(' ')

    # Remove the last space to avoid an extra trailing space
    if annotated_parts and annotated_parts[-1] == ' ':
        annotated_parts.pop()

    return annotated_text(*annotated_parts)

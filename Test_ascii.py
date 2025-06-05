def list_non_ascii_verbose(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            for col, char in enumerate(line):
                if ord(char) > 127:
                    start = max(0, col - 15)
                    end = min(len(line), col + 15)
                    context = line[start:end].replace('\n', '')
                    print(f"Ligne {lineno}, position {col}: caractère non-ASCII → '{char}' (U+{ord(char):04X})")
                    print(f"  Contexte: ...{context}...")
                    print()

list_non_ascii_verbose("muVINSv1_withTv9.py")

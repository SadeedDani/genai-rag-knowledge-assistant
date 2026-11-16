import re

# STEP 1: Load the document
# Read the entire company policy file into one string.

with open("docs/company_policy.txt", "r") as file:
    text = file.read()
# STEP 2: Split the document into sections
# Capture numbered headings so we can keep them in each chunk.

matches = re.split(
    r"^(\d+\.)",
    text,
    flags=re.MULTILINE
)

section_chunks = []

for i in range(1, len(matches), 2):
    chunk = matches[i] + matches[i + 1]
    section_chunks.append(chunk)


if __name__ == "__main__":
    print("Total chunks:", len(section_chunks))


# re.split() splits the document around numbered headings.
# (\d+\.) captures headings like 1. and 2. so they aren't lost.
# range(1, len(matches), 2) visits each heading and its following text.
# section_chunks.append(chunk) saves each complete section.

if __name__ == "__main__":
    print("Total chunks:", len(section_chunks))
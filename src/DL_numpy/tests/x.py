with open("src/data/raw/train.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

with open("src/data/raw/train.txt", "w", encoding="utf-8") as f:
    for line in lines:
        line = line.strip()
        if line:
            if not line.endswith("</s>"):
                line += " </s>"
            f.write(line + "\n")
filename = "test.trace"
target_word = "WR"
count = 0

with open(filename, 'r', encoding='utf-8') as f:
    for line in f:
        count += line.count(target_word)

print(f"'{target_word}' appears {count} times in {filename}")
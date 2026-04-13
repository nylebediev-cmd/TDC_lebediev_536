import random
import collections
import math
import string
import matplotlib.pyplot as plt

SURNAME = "Lebediev"
GROUP_NUMBER = "536"
STUDENT_ID = 8
N_SEQ = 100

def save_to_results(text):
    with open("results_sequence.txt", "a", encoding="utf-8") as f:
        f.write(text + "\n")


open("results_sequence.txt", "w").close()


# 1. Original sequence 1
list1 = ['1'] * STUDENT_ID
list0 = ['0'] * (N_SEQ - STUDENT_ID)
res1 = list1 + list0
random.shuffle(res1)
seq1 = "".join(res1)

# 2. Original sequence 2
list_surname = list(SURNAME)
list0_2 = ['0'] * (N_SEQ - len(list_surname))
seq2 = "".join(list_surname + list0_2)

# 3. Original sequence 3
res3 = list_surname + ['0'] * (N_SEQ - len(list_surname))
random.shuffle(res3)
seq3 = "".join(res3)

# 4. Original sequence 4
letters_4 = list(SURNAME) + list(GROUP_NUMBER)
n_repeats = N_SEQ // len(letters_4)
remainder = N_SEQ % len(letters_4)
seq4 = "".join(map(str, (letters_4 * n_repeats + letters_4[:remainder])))

# 5. Original sequence 5
alphabet_5 = list(SURNAME[:2]) + list(GROUP_NUMBER)
seq5_list = []
for char in alphabet_5:
    seq5_list.extend([char] * 20)
random.shuffle(seq5_list)
seq5 = "".join(seq5_list)

# 6. Original sequence 6
letters_6 = list(SURNAME[:2])
digits_6 = list(GROUP_NUMBER)
n_letters = int(0.7 * N_SEQ)
n_digits = int(0.3 * N_SEQ)
res6 = [random.choice(letters_6) for _ in range(n_letters)] + \
       [random.choice(digits_6) for _ in range(n_digits)]
random.shuffle(res6)
seq6 = "".join(res6)

# 7. Original sequence 7
elements_7 = string.ascii_lowercase + string.digits
seq7 = "".join([random.choice(elements_7) for _ in range(N_SEQ)])

# 8. Original sequence 8
seq8 = '1' * N_SEQ

all_sequences = [seq1, seq2, seq3, seq4, seq5, seq6, seq7, seq8]

with open("sequence.txt", "w", encoding="utf-8") as f:
    for s in all_sequences:
        f.write(s + "\n")

results_for_table = []

for i, sequence in enumerate(all_sequences, 1):
    alphabet = set(sequence)
    alphabet_size = len(alphabet)
    seq_size_bytes = len(sequence)

    counts = collections.Counter(sequence)
    probability = {symbol: count / N_SEQ for symbol, count in counts.items()}

    mean_prob = sum(probability.values()) / len(probability)
    is_equal = all(abs(p - mean_prob) < 0.05 * mean_prob for p in probability.values())
    uniformity = "рівна" if is_equal else "нерівна"

    entropy = -sum(p * math.log2(p) for p in probability.values())
    if alphabet_size > 1:
        source_excess = 1 - (entropy / math.log2(alphabet_size))
    else:
        source_excess = 1.0

    prob_str = ', '.join([f"{s}={p:.4f}" for s, p in probability.items()])

    report = (f"Тестова послідовність №{i}\n"
              f"Sequence: {sequence}\n"
              f"Alphabet size: {alphabet_size}\n"
              f"Size: {seq_size_bytes} bytes\n"
              f"Probabilities: {prob_str}\n"
              f"Mean Probability: {mean_prob:.4f}\n"
              f"Type: {uniformity}\n"
              f"Entropy: {entropy:.4f}\n"
              f"Redundancy: {source_excess:.4f}\n"
              + "-" * 50)
    save_to_results(report)

    results_for_table.append([alphabet_size, round(entropy, 2), round(source_excess, 2), uniformity])

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

headers = ['Розмір алфавіту', 'Ентропія', 'Надмірність', 'Ймовірність']
rows = [f'Послідовність {i}' for i in range(1, 9)]

table = ax.table(cellText=results_for_table,
                 colLabels=headers,
                 rowLabels=rows,
                 loc='center',
                 cellLoc='center')

table.set_fontsize(12)
table.scale(1, 2)
plt.title("Характеристики сформованих послідовностей", pad=20)

fig.savefig("Характеристики сформованих послідовностей.png", bbox_inches='tight', dpi=300)
print("Розрахунки завершено. Файли створено.")
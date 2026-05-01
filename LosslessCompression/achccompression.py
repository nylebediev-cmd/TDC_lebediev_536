import math
import collections
import matplotlib.pyplot as plt


def float_bin(point, size_cod):
    binary_code = ""

    for _ in range(size_cod):
        point *= 2

        if point > 1:
            binary_code += "1"
            point -= int(point)

        elif point == 1:
            binary_code += "1"
            break

        else:
            binary_code += "0"

    return binary_code


def encode_ac(uniq_chars, probabilitys, alphabet_size, sequence):
    alphabet = list(uniq_chars)
    probability = [probabilitys[c] for c in alphabet]

    unity = []
    probability_range = 0.0

    for i in range(alphabet_size):
        l = probability_range
        probability_range += probability[i]
        u = probability_range
        unity.append([alphabet[i], l, u])

    for i in range(len(sequence) - 1):
        for j in range(len(unity)):
            if sequence[i] == unity[j][0]:
                low = unity[j][1]
                high = unity[j][2]
                diff = high - low

                for k in range(len(unity)):
                    unity[k][1] = low
                    unity[k][2] = probability[k] * diff + low
                    low = unity[k][2]
                break

    low = 0
    high = 0

    for i in range(len(unity)):
        if unity[i][0] == sequence[-1]:
            low = unity[i][1]
            high = unity[i][2]

    point = (low + high) / 2

    if high - low == 0:
        size_cod = 1
        bin_code = "0"
    else:
        size_cod = math.ceil(math.log2(1 / (high - low)) + 1)
        bin_code = float_bin(point, size_cod)

    return [point, alphabet_size, alphabet, probability], bin_code


def decode_ac(encoded_data, length_seq):
    point, alphabet_size, alphabet, probability = encoded_data

    unity = []
    probability_range = 0.0

    for i in range(alphabet_size):
        l = probability_range
        probability_range += probability[i]
        u = probability_range
        unity.append([alphabet[i], l, u])

    decoded = ""

    for _ in range(length_seq):
        for j in range(len(unity)):
            if unity[j][1] < point < unity[j][2]:
                decoded += unity[j][0]

                low = unity[j][1]
                high = unity[j][2]
                diff = high - low

                for k in range(len(unity)):
                    unity[k][1] = low
                    unity[k][2] = probability[k] * diff + low
                    low = unity[k][2]
                break

    return decoded


def encode_ch(uniq_chars, probabilitys, sequence):
    alphabet = list(uniq_chars)
    probability = [probabilitys[c] for c in alphabet]

    final = [[alphabet[i], probability[i]] for i in range(len(alphabet))]
    final.sort(key=lambda x: x[1])

    tree = []

    if len(set(probability)) == 1:
        symbol_code = []
        for i in range(len(alphabet)):
            symbol_code.append([alphabet[i], "1" * i + "0"])

        encode = "".join([symbol_code[alphabet.index(c)][1] for c in sequence])
        return [encode, symbol_code], encode

    while len(final) > 1:
        left = final.pop(0)
        right = final.pop(0)

        tot = left[1] + right[1]
        tree.append([left[0], right[0]])

        final.append([left[0] + right[0], tot])
        final.sort(key=lambda x: x[1])

    tree.reverse()
    alphabet.sort()

    symbol_code = []

    for i in range(len(alphabet)):
        code = ""

        for j in range(len(tree)):
            if alphabet[i] in tree[j][0]:
                code += "0"
                if alphabet[i] == tree[j][0]:
                    break
            else:
                code += "1"
                if alphabet[i] == tree[j][1]:
                    break

        symbol_code.append([alphabet[i], code])

    encode = ""
    for c in sequence:
        encode += [sc[1] for sc in symbol_code if sc[0] == c][0]

    return [encode, symbol_code], encode


def decode_ch(encoded_data):
    encode = list(encoded_data[0])
    symbol_code = encoded_data[1]

    sequence = ""
    count = 0
    flag = 0

    for i in range(len(encode)):
        for j in range(len(symbol_code)):
            if encode[i] == symbol_code[j][1]:
                sequence += symbol_code[j][0]
                flag = 1

        if flag == 1:
            flag = 0
        else:
            count += 1
            if count == len(encode):
                break
            else:
                encode.insert(i + 1, encode[i] + encode[i + 1])
                encode.pop(i + 2)

    return sequence


def main():
    with open("sequence.txt", "r") as file:
        original_sequences = [line.strip() for line in file.readlines() if line.strip()]

    results = []
    open("results_AC_CH.txt", "w").close()

    for idx, sequence in enumerate(original_sequences):

        sequence = sequence[:10]

        sequence_length = len(sequence)
        unique_chars = set(sequence)
        alphabet_size = len(unique_chars)

        counts = collections.Counter(sequence)

        N_sequence = 100
        probability = {c: counts[c] / N_sequence for c in counts}

        entropy = -sum(p * math.log2(p) for p in probability.values())

        enc_ac_data, enc_ac = encode_ac(unique_chars, probability, alphabet_size, sequence)
        dec_ac = decode_ac(enc_ac_data, sequence_length)
        bps_ac = len(enc_ac) / sequence_length

        enc_ch_data, enc_ch = encode_ch(unique_chars, probability, sequence)
        dec_ch = decode_ch(enc_ch_data)
        bps_ch = len(enc_ch) / sequence_length

        with open("results_AC_CH.txt", "a", encoding="utf-8") as f:
            f.write(f"\nSequence {idx+1}: {sequence}\n")
            f.write(f"Entropy: {entropy:.4f}\n")

            f.write(f"AC encoded: {enc_ac}\n")
            f.write(f"AC decoded: {dec_ac}\n")
            f.write(f"BPS AC: {bps_ac:.4f}\n")

            f.write(f"CH encoded: {enc_ch}\n")
            f.write(f"CH decoded: {dec_ch}\n")
            f.write(f"BPS CH: {bps_ch:.4f}\n")

        results.append([round(entropy, 2), bps_ac, bps_ch])

    N = len(results)
    fig, ax = plt.subplots(figsize=(14/1.54, N/1.54))

    ax.axis('off')

    table = ax.table(
        cellText=results,
        colLabels=["Entropy", "bps AC", "bps CH"],
        rowLabels=[f"Seq {i+1}" for i in range(N)],
        loc="center",
        cellLoc="center"
    )

    table.set_fontsize(14)
    table.scale(0.8, 2)

    fig.savefig("Результати стиснення методами AC та CH.png")


if __name__ == "__main__":
    main()
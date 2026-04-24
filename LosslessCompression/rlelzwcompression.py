import ast
import math
import collections
import matplotlib.pyplot as plt


def encode_rle(sequence):
    if not sequence:
        return "", []
    count = 1
    result = []
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i - 1]:
            count += 1
        else:
            result.append((sequence[i - 1], count))
            count = 1
    result.append((sequence[-1], count))
    encoded_str = "".join([f"{item[1]}{item[0]}" for item in result])
    return encoded_str, result


def decode_rle(encoded_data):
    result = []
    for char, count in encoded_data:
        result.append(char * count)
    return "".join(result)


def encode_lzw(data, output_file):
    dictionary = {chr(i): i for i in range(65536)}
    current = ""
    result = []
    total_bits = 0
    with open(output_file, "a", encoding="utf-8") as f:
        for c in data:
            new_str = current + c
            if new_str in dictionary:
                current = new_str
            else:
                code = dictionary[current]
                bits = 16 if code < 65536 else math.ceil(math.log2(len(dictionary)))
                result.append(code)
                total_bits += bits
                f.write(f"Code: {code}, Element: {current}, Bits: {bits}\n")
                dictionary[new_str] = len(dictionary)
                current = c
        if current:
            code = dictionary[current]
            bits = 16 if code < 65536 else math.ceil(math.log2(len(dictionary)))
            result.append(code)
            total_bits += bits
            f.write(f"Code: {code}, Element: {current}, Bits: {bits}\n")
    return result, total_bits


def decode_lzw(sequence):
    dictionary = {i: chr(i) for i in range(65536)}
    result = ""
    previous = None
    for code in sequence:
        if code in dictionary:
            current = dictionary[code]
        else:
            current = previous + previous[0]
        result += current
        if previous is not None:
            dictionary[len(dictionary)] = previous + current[0]
        previous = current
    return result


def main():
    try:
        with open("sequence.txt", "r", encoding="utf-8") as file:
            content = file.read().strip()

            if not (content.startswith('[') or content.startswith('(')):
                lines = content.splitlines()
                formatted_content = str([line.strip() for line in lines])
            else:
                formatted_content = content

            original_sequences = ast.literal_eval(formatted_content)
            original_sequences = [str(sequence).strip("[]'\" ") for sequence in original_sequences]

    except Exception as e:
        print(f"Error reading file: {e}")
        return

    results_for_table = []
    output_filename = "results_rle_lzw.txt"
    open(output_filename, "w").close()

    for idx, seq in enumerate(original_sequences):
        N_seq = len(seq)
        counts = collections.Counter(seq)
        probability = {symbol: count / N_seq for symbol, count in counts.items()}
        entropy = -sum(p * math.log2(p) for p in probability.values() if p > 0)
        input_size_bits = N_seq * 16

        with open(output_filename, "a", encoding="utf-8") as f:
            f.write(f"\n--- Sequence {idx + 1} ---\nOriginal: {seq}\n")
            f.write(f"Entropy: {entropy:.4f}\n")
            f.write(f"Input size: {input_size_bits} bits\n")

        encoded_rle_str, encoded_rle_list = encode_rle(seq)
        rle_size_bits = len(encoded_rle_str) * 16
        cr_rle = round(input_size_bits / rle_size_bits, 2) if rle_size_bits > 0 else 0
        display_cr_rle = cr_rle if cr_rle >= 1 else "-"

        decoded_rle_res = decode_rle(encoded_rle_list)

        lzw_codes, lzw_size_bits = encode_lzw(seq, output_filename)
        cr_lzw = round(input_size_bits / lzw_size_bits, 2)
        decoded_lzw_res = decode_lzw(lzw_codes)

        with open(output_filename, "a", encoding="utf-8") as f:
            f.write(f"RLE Result: {encoded_rle_str}\n")
            f.write(f"RLE CR: {display_cr_rle}\n")
            f.write(f"LZW Codes: {lzw_codes}\n")
            f.write(f"LZW Size: {lzw_size_bits} bits\n")
            f.write(f"LZW CR: {cr_lzw}\n")

        results_for_table.append([round(entropy, 2), display_cr_rle, cr_lzw])

    fig, ax = plt.subplots(figsize=(10, len(original_sequences) * 0.8))
    headers = ['Ентропія', 'КС RLE', 'КС LZW']
    rows = [f'Послідовність {i + 1}' for i in range(len(original_sequences))]
    ax.axis('off')
    table = ax.table(cellText=results_for_table, colLabels=headers, rowLabels=rows, loc='center', cellLoc='center')
    table.set_fontsize(12)
    table.scale(1, 2)
    fig.savefig("Результати стиснення методами RLE та LZW.png")


if __name__ == "__main__":
    main()
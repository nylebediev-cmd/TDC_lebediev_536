import os
import math
import numpy as np
from scipy import fftpack
from PIL import Image
from huffman import HuffmanTree


def dct_2d(image):
    return fftpack.dct(fftpack.dct(image.T, norm='ortho').T, norm='ortho')


def idct_2d(image):
    return fftpack.idct(fftpack.idct(image.T, norm='ortho').T, norm='ortho')


def load_quantization_table(component, variant=1):
    if component == 'lum':
        if variant == 1:
            return np.array([
                [2, 2, 2, 2, 3, 4, 5, 6], [2, 2, 2, 2, 3, 4, 5, 6],
                [2, 2, 2, 2, 4, 5, 7, 9], [2, 2, 2, 4, 5, 7, 9, 12],
                [3, 3, 4, 5, 8, 10, 12, 12], [4, 4, 5, 7, 10, 12, 12, 12],
                [5, 5, 7, 9, 12, 12, 12, 12], [6, 6, 9, 12, 12, 12, 12, 12]])
        else:
            return np.array([
                [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 48, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]])
    elif component == 'chrom':
        if variant == 1:
            return np.array([
                [3, 3, 5, 9, 13, 15, 15, 15], [3, 4, 6, 11, 14, 12, 12, 12],
                [5, 6, 9, 14, 12, 12, 12, 12], [9, 11, 14, 12, 12, 12, 12, 12],
                [13, 14, 12, 12, 12, 12, 12, 12], [15, 12, 12, 12, 12, 12, 12, 12],
                [15, 12, 12, 12, 12, 12, 12, 12], [15, 12, 12, 12, 12, 12, 12, 12]])
        else:
            return np.array([
                [17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99]])


def quantize(block, component, variant=1):
    q = load_quantization_table(component, variant)
    return (block / q).round().astype(np.int32)


def zigzag_points(rows, cols):
    UP, DOWN, RIGHT, LEFT, UP_RIGHT, DOWN_LEFT = range(6)

    def move(direction, point):
        return {
            UP: lambda p: (p[0] - 1, p[1]), DOWN: lambda p: (p[0] + 1, p[1]),
            LEFT: lambda p: (p[0], p[1] - 1), RIGHT: lambda p: (p[0], p[1] + 1),
            UP_RIGHT: lambda p: move(UP, move(RIGHT, p)), DOWN_LEFT: lambda p: move(DOWN, move(LEFT, p))
        }[direction](point)

    def inbounds(p):
        return 0 <= p[0] < rows and 0 <= p[1] < cols

    point = (0, 0)
    move_up = True
    for _ in range(rows * cols):
        yield point
        if move_up:
            if inbounds(move(UP_RIGHT, point)):
                point = move(UP_RIGHT, point)
            else:
                move_up = False
                if inbounds(move(RIGHT, point)):
                    point = move(RIGHT, point)
                else:
                    point = move(DOWN, point)
        else:
            if inbounds(move(DOWN_LEFT, point)):
                point = move(DOWN_LEFT, point)
            else:
                move_up = True
                if inbounds(move(DOWN, point)):
                    point = move(DOWN, point)
                else:
                    point = move(RIGHT, point)


def block_to_zigzag(block):
    return np.array([block[point] for point in zigzag_points(*block.shape)])


def bits_required(n):
    n = abs(n)
    result = 0
    while n > 0:
        n >>= 1
        result += 1
    return result


def binstr_flip(binstr):
    return ''.join(map(lambda c: '0' if c == '1' else '1', binstr))


def uint_to_binstr(number, size):
    return bin(number)[2:].zfill(size)[-size:]


def int_to_binstr(n):
    if n == 0: return ''
    binstr = bin(abs(n))[2:]
    return binstr if n > 0 else binstr_flip(binstr)


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def run_length_encode(arr):
    last_nonzero = -1
    for i, elem in enumerate(arr):
        if elem != 0: last_nonzero = i
    symbols, values = [], []
    run_length = 0
    for i, elem in enumerate(arr):
        if i > last_nonzero:
            symbols.append((0, 0));
            values.append(int_to_binstr(0))
            break
        elif elem == 0 and run_length < 15:
            run_length += 1
        else:
            size = bits_required(elem)
            symbols.append((run_length, size))
            values.append(int_to_binstr(elem))
            run_length = 0
    return symbols, values


def write_to_file(filepath, dc, ac, blocks_count, tables):
    with open(filepath, 'w', encoding='utf-8') as f:
        for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:
            f.write(uint_to_binstr(len(tables[table_name]), 16))
            for key, value in tables[table_name].items():
                if table_name in {'dc_y', 'dc_c'}:
                    f.write(uint_to_binstr(key, 4))
                    f.write(uint_to_binstr(len(value), 4))
                    f.write(value)
                else:
                    f.write(uint_to_binstr(key[0], 4))
                    f.write(uint_to_binstr(key[1], 4))
                    f.write(uint_to_binstr(len(value), 8))
                    f.write(value)
        f.write(uint_to_binstr(blocks_count, 32))
        for b in range(blocks_count):
            for c in range(3):
                category = bits_required(dc[b, c])
                symbols, values = run_length_encode(ac[b, :, c])
                dc_table = tables['dc_y'] if c == 0 else tables['dc_c']
                ac_table = tables['ac_y'] if c == 0 else tables['ac_c']
                f.write(dc_table[category])
                f.write(int_to_binstr(dc[b, c]))
                for i in range(len(symbols)):
                    f.write(ac_table[tuple(symbols[i])])
                    f.write(values[i])



def encode():
    input_images = ["strong_texture.bmp", "medium_texture.bmp", "weak_texture.bmp"]
    results_dir = "Results"

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Створення файлу результатів
    with open("results_jpeg.txt", "w", encoding="utf-8") as res_file:
        res_file.write("=== Результати тестування стиснення JPEG ===\n\n")

    for file_name in input_images:
        if not os.path.exists(file_name):
            print(f"Помилка: Файл {file_name} не знайдено.")
            continue

        print(f"Обробка файлу: {file_name}")
        image = Image.open(file_name)
        w, h = image.size
        size = min(w, h, 1024)
        size -= (size % 8)
        image = image.crop((0, 0, size, size))

        ycbcr = image.convert('YCbCr')
        npmat = np.array(ycbcr, dtype=np.uint8)
        rows, cols = npmat.shape[0], npmat.shape[1]
        blocks_count = (rows // 8) * (cols // 8)

        for variant in [1, 2]:
            output_asf = os.path.join(results_dir, f"{file_name.split('.')[0]}_v{variant}.asf")
            output_jpg = os.path.join(results_dir, f"{file_name.split('.')[0]}_v{variant}_decoded.jpg")

            dc = np.empty((blocks_count, 3), dtype=np.int32)
            ac = np.empty((blocks_count, 63, 3), dtype=np.int32)

            block_index = 0
            for i in range(0, rows, 8):
                for j in range(0, cols, 8):
                    for k in range(3):
                        block = npmat[i:i + 8, j:j + 8, k].astype(float) - 128
                        dct_matrix = dct_2d(block)
                        quant_matrix = quantize(dct_matrix, 'lum' if k == 0 else 'chrom', variant)
                        zigzag = block_to_zigzag(quant_matrix)
                        dc[block_index, k] = zigzag[0]
                        ac[block_index, :, k] = zigzag[1:]
                    block_index += 1

            H_DC_Y = HuffmanTree(np.vectorize(bits_required)(dc[:, 0]))
            H_DC_C = HuffmanTree(np.vectorize(bits_required)(dc[:, 1:].flat))
            H_AC_Y = HuffmanTree(flatten(run_length_encode(ac[i, :, 0])[0] for i in range(blocks_count)))
            H_AC_C = HuffmanTree(
                flatten(run_length_encode(ac[i, :, j])[0] for i in range(blocks_count) for j in [1, 2]))

            tables = {
                'dc_y': H_DC_Y.value_to_bitstring_table(),
                'ac_y': H_AC_Y.value_to_bitstring_table(),
                'dc_c': H_DC_C.value_to_bitstring_table(),
                'ac_c': H_AC_C.value_to_bitstring_table()
            }

            write_to_file(output_asf, dc, ac, blocks_count, tables)
            image.save(output_jpg, "JPEG")


            original_size = os.path.getsize(file_name)
            compressed_size = os.path.getsize(output_asf)

            with open("results_jpeg.txt", "a", encoding="utf-8") as res_file:
                res_file.write(f"Файл: {file_name} | Таблиця: {variant}\n")
                res_file.write(f"Розмір вихідного файла: {original_size} байт\n")
                res_file.write(f"Розмір стисненого файла (.asf): {compressed_size} байт\n")
                res_file.write(f"Коефіцієнт стиснення: {round(original_size / compressed_size, 2)}\n")
                res_file.write("-" * 40 + "\n")


if __name__ == "__main__":
    encode()
    print("Програма завершена. Результати збережено в папці Results.")
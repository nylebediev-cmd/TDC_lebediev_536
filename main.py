import soundfile as sf
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import resample, convolve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skimage.restoration import cycle_spin

# Автоматичне визначення шляху до папки, де лежить цей скрипт
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOUNDS_DIR = os.path.join(BASE_DIR, "Sounds")

# Константи з повними шляхами
NAME_ORIGINAL_WAV = os.path.join(SOUNDS_DIR, "Sound_44100[Hz]_2[byte].wav")
NAME_RESAMPLED_WAV = os.path.join(SOUNDS_DIR, "Sound_4000[Hz]_2[byte].wav")
SAMPLE_RATE = 44100


def to_scientific_pretty(x, precision=2):
    """Перетворення числа у формат з математичним степенем"""
    superscripts = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    mantissa, exponent = f"{x:.{precision}e}".split('e')
    mantissa = mantissa.rstrip('0').rstrip('.')
    return f"{mantissa} · 10{str(int(exponent)).translate(superscripts)}"


def gaussian_kernel(size, sigma):
    x = np.linspace(-(size // 2), size // 2, size)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def wavelet_denoiser(signal, level=5, mode='hard', wavelet='db4'):
    signal = np.asarray(signal)
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(signal.size))
    denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode=mode) for c in coeffs[1:]]
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
    return denoised_signal[:len(signal)]


if __name__ == "__main__":
    # Перевірка наявності головного файлу
    if not os.path.exists(NAME_ORIGINAL_WAV):
        print(f"Помилка! Не вдалося знайти файл за шляхом: {NAME_ORIGINAL_WAV}")
        print(f"Перевірте, чи файл Sound_44100[Hz]_2[byte].wav точно лежить у {SOUNDS_DIR}")
    else:
        results = []
        row_labels = []
        headers = ['MSE', 'MAE', 'RMSE', 'R2', 'D']

        # 1. Зчитування оригінального сигналу
        data_original, fs_orig = sf.read(NAME_ORIGINAL_WAV)

        # 2. Пошук усіх wav файлів у папці Sounds
        wav_files = glob.glob(os.path.join(SOUNDS_DIR, "*.wav"))

        for sounds in wav_files:
            sounds = sounds.replace("\\", "/")

            # Пропустити оригінальний файл
            if os.path.normpath(sounds) == os.path.normpath(NAME_ORIGINAL_WAV):
                continue

            # Обробка ресемплованого файлу
            if os.path.normpath(sounds) == os.path.normpath(NAME_RESAMPLED_WAV):
                row_labels.append('Ресемпл 4 кГц')
                data, fs = sf.read(sounds)
                data = resample(data, len(data_original))

            # Обробка всіх інших фільтрованих сигналів
            else:
                type_filter = os.path.basename(sounds)
                type_filter = type_filter.replace('Filtered_', '').replace('.wav', '').replace('_', ' ')

                if '4000[Hz] 2[byte]' in type_filter:
                    type_filter = 'Лінійний фільтр 4 кГц'

                row_labels.append(type_filter)
                data, fs = sf.read(sounds)

            # Розрахунок метрик
            mse = mean_squared_error(data_original, data)
            mae = mean_absolute_error(data_original, data)
            rmse = np.sqrt(mse)
            r2 = r2_score(data_original, data)
            D = np.var(data_original - data)

            results.append([
                to_scientific_pretty(mse),
                to_scientific_pretty(mae),
                to_scientific_pretty(rmse),
                round(r2, 2),
                to_scientific_pretty(D)
            ])

        # 3. Побудова таблиці
        if results:
            n_rows = len(row_labels)
            n_cols = len(headers)
            fig, ax = plt.subplots(figsize=(n_cols * 2.5, n_rows * 0.5 + 1))
            ax.axis('off')

            table = ax.table(
                cellText=results,
                rowLabels=row_labels,
                colLabels=headers,
                loc='center',
                cellLoc='center',
                bbox=[0.1, 0, 0.9, 1]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)

            # Зберігаємо результат у ту саму папку Sounds
            save_path = os.path.join(SOUNDS_DIR, "Table_Metrics_Results.png")
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"Розрахунок завершено! Таблицю збережено: {save_path}")
            plt.show()
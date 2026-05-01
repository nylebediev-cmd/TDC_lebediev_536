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


def filtration_efficiency():
    # 1. Зчитування початкового сигналу та ініціалізація
    data, fs_original = sf.read(NAME_ORIGINAL_WAV)
    signal_power = np.mean(data ** 2)
    max_shifts = 5

    mse_wt_mean = []
    mse_wt_list = []
    mse_g_mean = []
    mse_g_list = []
    snr_values = []

    # Списки для додаткових метрик (MAE, RMSE, R2, D)
    mae_wt_mean, rmse_wt_mean, r2_wt_mean, d_wt_mean = [], [], [], []
    mae_g_mean, rmse_g_mean, r2_g_mean, d_g_mean = [], [], [], []

    print("Розрахунок завадостійкості... Зачекайте завершення.")

    # 2. Основні цикли
    for SNR_dB in np.arange(-10, 21, 0.5):
        mse_wt = []
        mse_g = []
        t_mae_w, t_rmse_w, t_r2_w, t_d_w = [], [], [], []
        t_mae_g, t_rmse_g, t_r2_g, t_d_g = [], [], [], []

        for i in range(0, 10):
            # Генерація шуму
            noise_power = signal_power / (10 ** (SNR_dB / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), size=data.shape)
            noisy_signal = data + noise

            # Фільтрація
            sig_filtered_wavelet = cycle_spin(noisy_signal, func=wavelet_denoiser,
                                              max_shifts=max_shifts, shift_steps=5, num_workers=1)

            kernel = gaussian_kernel(size=11, sigma=2)
            sig_filtered_gaussian = convolve(noisy_signal, kernel, mode='same')

            # MSE
            m_w = mean_squared_error(data, sig_filtered_wavelet)
            m_g = mean_squared_error(data, sig_filtered_gaussian)
            mse_wt.append(m_w)
            mse_g.append(m_g)

            # Інші метрики
            t_mae_w.append(mean_absolute_error(data, sig_filtered_wavelet))
            t_rmse_w.append(np.sqrt(m_w))
            t_r2_w.append(r2_score(data, sig_filtered_wavelet))
            t_d_w.append(np.var(data - sig_filtered_wavelet))

            t_mae_g.append(mean_absolute_error(data, sig_filtered_gaussian))
            t_rmse_g.append(np.sqrt(m_g))
            t_r2_g.append(r2_score(data, sig_filtered_gaussian))
            t_d_g.append(np.var(data - sig_filtered_gaussian))

        # Збереження результатів
        mse_wt_mean.append(np.mean(mse_wt))
        mse_wt_list.append(list(mse_wt))
        mse_g_mean.append(np.mean(mse_g))
        mse_g_list.append(list(mse_g))
        snr_values.append(SNR_dB)

        mae_wt_mean.append(np.mean(t_mae_w))
        rmse_wt_mean.append(np.mean(t_rmse_w))
        r2_wt_mean.append(np.mean(t_r2_w))
        d_wt_mean.append(np.mean(t_d_w))

        mae_g_mean.append(np.mean(t_mae_g))
        rmse_g_mean.append(np.mean(t_rmse_g))
        r2_g_mean.append(np.mean(t_r2_g))
        d_g_mean.append(np.mean(t_d_g))

    # 3. Підготовка даних для Scatter
    snr_scatter_wt, mse_scatter_wt = [], []
    snr_scatter_g, mse_scatter_g = [], []
    for snr, mse_list in zip(snr_values, mse_wt_list):
        snr_scatter_wt.extend([snr] * len(mse_list))
        mse_scatter_wt.extend(mse_list)
    for snr, mse_list in zip(snr_values, mse_g_list):
        snr_scatter_g.extend([snr] * len(mse_list))
        mse_scatter_g.extend(mse_list)

    # 4. Побудова графіків MSE (Лінійний та Логарифмічний)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax in axes:
        ax.scatter(snr_scatter_wt, mse_scatter_wt, color='red', alpha=0.05, label="Окремі значення MSE")
        ax.plot(snr_values, mse_wt_mean, color='red', linewidth=2, label="Середнє MSE WT")
        ax.scatter(snr_scatter_g, mse_scatter_g, color='green', alpha=0.05, label="Окремі значення MSE")
        ax.plot(snr_values, mse_g_mean, color='green', linewidth=2, label="Середнє MSE GF")
        ax.set_xlabel("SNR (дБ)")
        ax.set_ylabel("MSE")
        ax.grid(True)
        ax.legend()

    axes[0].set_xticks(np.arange(-10, 21, 2))
    axes[0].set_title("Лінійний масштаб")
    axes[1].set_xticks(np.arange(-10, 21, 1))
    axes[1].set_yscale('log')
    axes[1].set_title("Логарифмічний масштаб")

    plt.tight_layout()
    plt.savefig("MSE_vs_SNR.png", dpi=600)
    plt.show()

    # 5. Побудова додаткових метрик
    fig_m, axes_m = plt.subplots(2, 2, figsize=(12, 10))
    metrics_data = [
        (mae_wt_mean, mae_g_mean, "MAE"),
        (rmse_wt_mean, rmse_g_mean, "RMSE"),
        (r2_wt_mean, r2_g_mean, "R2 Score"),
        (d_wt_mean, d_g_mean, "Dispersion (D)")
    ]

    for ax, (wt_d, g_d, title) in zip(axes_m.flatten(), metrics_data):
        ax.plot(snr_values, wt_d, color='red', label="Wavelet")
        ax.plot(snr_values, g_d, color='green', label="Gaussian")
        ax.set_title(title)
        ax.set_xlabel("SNR (дБ)")
        ax.grid(True)
        ax.legend()
        if title != "R2 Score": ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig("Metrics_Comparison.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    # filtration_efficiency()


    """
   

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

            save_path = os.path.join(SOUNDS_DIR, "Table_Metrics_Results.png")
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            print(f"Розрахунок завершено! Таблицю збережено: {save_path}")
            plt.show()
    """

    # Виклик нової функції для Практичної роботи №6
    filtration_efficiency()
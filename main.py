import soundfile as sf
from math import gcd
from scipy.signal import resample_poly, butter, sosfiltfilt, convolve
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import (
    denoise_wavelet, denoise_invariant, denoise_tv_chambolle,
    denoise_bilateral, cycle_spin
)
import pywt

# Шляхи до аудіо файлів та константи
NAME_ORIGINAL_WAV = f"./Sounds/Sound_44100[Hz]_2[byte].wav"
SAMPLE_RATE = 44100


def gaussian_kernel(size, sigma):
    """Створення ядра згортки для фільтра Гаусса"""
    x = np.linspace(-(size // 2), size // 2, size)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def wavelet_denoiser(signal, level=5, mode='hard', wavelet='db4'):
    """
    Очищення сигналу від шуму за допомогою DWT (з параметрами за замовчуванням).
    """
    # Переконуємось, що сигнал має потрібний тип для pywt
    signal = np.asarray(signal)
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # Обчислення порогу за універсальною формулою
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(signal.size))

    denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode=mode) for c in coeffs[1:]]
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)

    # Вирівнюємо довжину (важливо для cycle_spin)
    return denoised_signal[:len(signal)]


def wavelet_shifted_filter():
    # 1. Зчитування сигналу
    data, fs_original = sf.read(NAME_ORIGINAL_WAV)
    time = (np.arange(len(data)) / fs_original) * 1000  # Час у мс для графіків

    # --- Блок Wavelet Shifted Filtering ---
    max_shifts = [0, 1, 3, 5]
    signals = []

    for n, s in enumerate(max_shifts):
        # Використання Cycle Spinning для інваріантності до зсуву
        sig_filtered = cycle_spin(
            data,
            func=wavelet_denoiser,
            max_shifts=s,
            shift_steps=1  # Крок зсуву
        )
        # Збереження результатів (всього 4 файли)
        sf.write(f"./Sounds/Filtered_Shifted_Wavelet_{n}.wav", sig_filtered, SAMPLE_RATE)
        signals.append(sig_filtered)

    # --- Блок Gaussian Filter ---
    kernel = gaussian_kernel(size=11, sigma=2)
    filtered_signal_gauss = convolve(data, kernel, mode='same')
    sf.write(f"./Sounds/Filtered_Gaussian_Filter.wav", filtered_signal_gauss, SAMPLE_RATE)

    # --- Візуалізація результатів: Вейвлет ---
    plt.figure(figsize=(12, 6))
    plt.plot(time, data, label=f"Оригінал (fs={SAMPLE_RATE} Гц)", alpha=0.5)
    plt.plot(time, signals[0], label=f"Wavelet Shifted: no shift")
    plt.plot(time, signals[1], label=f"Wavelet Shifted: 1x2")  # Умовно позначимо зсуви як у завданні
    plt.plot(time, signals[2], label=f"Wavelet Shifted: 1x4")
    plt.plot(time, signals[3], label=f"Wavelet Shifted: 1x6")
    plt.title("Порівняння сигналів у часовій області, вейвлет-фільтр, модифікований")
    plt.xlabel("Час (мс)")
    plt.ylabel("Амплітуда")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./Sounds/Figure_Wavelet_Modified.png")
    plt.show()

    # --- Візуалізація результатів: Гаусс ---
    plt.figure(figsize=(12, 6))
    plt.plot(time, data, label=f"Оригінал (fs={SAMPLE_RATE} Гц)", alpha=0.5)
    plt.plot(time, filtered_signal_gauss, label=f"Gaussian Filter", color='red')
    plt.title("Порівняння сигналів у часовій області, фільтр Гаусса")
    plt.xlabel("Час (мс)")
    plt.ylabel("Амплітуда")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./Sounds/Figure_Gaussian_Filter.png")
    plt.show()


if __name__ == "__main__":
    wavelet_shifted_filter()
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet, denoise_invariant, denoise_tv_chambolle, denoise_bilateral
import pywt

# Шляхи до аудіо файлів
NAME_ORIGINAL_WAV = f"./Sounds/Sound_44100[Hz]_2[byte].wav"
SAMPLE_RATE = 44100
DTYPE = np.int16

# --- Функції фільтрації ---

def wavelet_denoiser(signal, level, mode, wavelet):
    """
    Denoise a 1D signal using Discrete Wavelet Transform (DWT) thresholding.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(signal.size))
    denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, threshold, mode=mode) for c in coeffs[1:]]
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
    return denoised_signal[:len(signal)]

def invarince_denoiser(image, **kwargs):
    """
    J-Invariance denoising using wavelet with sigma=0.5.
    """
    return denoise_wavelet(image, sigma=0.5, wavelet='db4', mode='soft')

# --- Основна функція для фільтрації ---
def sound_filter():
    # Зчитування аудіо
    data, fs_original = sf.read(NAME_ORIGINAL_WAV)
    time = np.arange(len(data)) / fs_original

    # Перетворення в 2D для деяких фільтрів
    data_2d = data.reshape(1, -1)

    # --- Виконання фільтрації ---
    invariance = denoise_invariant(data_2d, denoise_function=invarince_denoiser).flatten()
    total_variation = denoise_tv_chambolle(data_2d, weight=0.1, channel_axis=None).flatten()
    bilateral = denoise_bilateral(data_2d, sigma_color=0.05, sigma_spatial=15, channel_axis=None).flatten()
    wavelet = wavelet_denoiser(data, level=5, mode='soft', wavelet='db4')

    # --- Збереження фільтрованих сигналів ---
    sf.write("./Sounds/Filtered_Invariance.wav", invariance, SAMPLE_RATE)
    sf.write("./Sounds/Filtered_Total_Variation.wav", total_variation, SAMPLE_RATE)
    sf.write("./Sounds/Filtered_Bilateral.wav", bilateral, SAMPLE_RATE)
    sf.write("./Sounds/Filtered_Wavelet.wav", wavelet, SAMPLE_RATE)

    # --- Візуалізація результатів ---
    filters = {
        "J-Invariance": invariance,
        "Total Variation (TV)": total_variation,
        "Bilateral": bilateral,
        "Wavelet (db4)": wavelet
    }

    for name, filtered_signal in filters.items():
        plt.figure(figsize=(10, 6))
        plt.plot(time, data, 'b-', label='Original Signal')
        plt.plot(time, filtered_signal, 'g-', linewidth=2, label=name)
        plt.title(f"Фільтрація сигналу: {name}")
        plt.xlabel("Час (с)")
        plt.ylabel("Амплітуда")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.savefig(f"./Sounds/Figure_{name.replace(' ', '_')}.png")


# --- Виклик основної функції ---
if __name__ == "__main__":
    sound_filter()
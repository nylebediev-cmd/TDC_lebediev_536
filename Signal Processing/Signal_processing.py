import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftshift, fftfreq

# =========================
#  (варіант 8)
# =========================
n = 500
Fs = 1000
F_max = 17
F_filter = 24

random_signal = np.random.normal(0, 10, n)
t = np.arange(n) / Fs

w = F_max / (Fs / 2)
sos = signal.butter(3, w, 'low', output='sos')

filtered_signal = signal.sosfiltfilt(sos, random_signal)

spectrum = np.abs(fftshift(fft(filtered_signal)))
freqs = fftshift(fftfreq(n, 1/Fs))

discrete_signals = []
discrete_spectrums = []
restored_signals = []
variances = []
snr_values = []

for Dt in [2, 4, 8, 16]:

    discrete_signal = np.zeros(n)

    for i in range(0, round(n / Dt)):
        discrete_signal[i * Dt] = filtered_signal[i * Dt]

    discrete_signals.append(discrete_signal)

    discrete_spectrums.append(np.abs(fftshift(fft(discrete_signal))))

    sos2 = signal.butter(3, F_filter / (Fs / 2), 'low', output='sos')
    restored = signal.sosfiltfilt(sos2, discrete_signal)

    restored_signals.append(restored)

    E1 = restored - filtered_signal

    variances.append(np.var(E1))
    snr_values.append(np.var(filtered_signal) / np.var(E1))


def plot_2x2(x, y, title, xlabel, ylabel):

    fig, ax = plt.subplots(2, 2, figsize=(21/2.54, 14/2.54))

    k = 0
    for i in range(2):
        for j in range(2):
            ax[i][j].plot(x, y[k], linewidth=1)
            ax[i][j].set_title(f"Крок Dt = {[2,4,8,16][k]}")
            ax[i][j].grid()
            k += 1

    fig.supxlabel(xlabel, fontsize=14)
    fig.supylabel(ylabel, fontsize=14)
    fig.suptitle(title, fontsize=14)

    fig.savefig('./figures/' + title + '.png', dpi=600)
    plt.show()


# =========================
# ГРАФІКИ
# =========================

plot_2x2(t, discrete_signals,
         "Дискретизовані сигнали",
         "Час (с)",
         "Амплітуда")

plot_2x2(freqs, discrete_spectrums,
         "Спектри дискретизованих сигналів",
         "Частота (Гц)",
         "Амплітуда")

plot_2x2(t, restored_signals,
         "Відновлені сигнали",
         "Час (с)",
         "Амплітуда")

plt.figure(figsize=(21/2.54, 14/2.54))
plt.plot([2,4,8,16], variances, linewidth=1)
plt.title("Залежність дисперсії від кроку дискретизації")
plt.xlabel("Крок дискретизації Dt")
plt.ylabel("Дисперсія похибки")
plt.grid()
plt.savefig('./figures/variance.png', dpi=600)
plt.show()

plt.figure(figsize=(21/2.54, 14/2.54))
plt.plot([2,4,8,16], snr_values, linewidth=1)
plt.title("Залежність співвідношення сигнал/шум від кроку дискретизації")
plt.xlabel("Крок дискретизації Dt")
plt.ylabel("SNR")
plt.grid()
plt.savefig('./figures/snr.png', dpi=600)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fft
from scipy.fft import fft, fftshift, fftfreq

# =========================
#  (варіант 8)
# =========================
n = 500
Fs = 1000
F_max = 17

random_signal = np.random.normal(0, 10, n)
t = np.arange(n) / Fs

w = F_max / (Fs / 2)
sos = signal.butter(3, w, 'low', output='sos')
filtered_signal = signal.sosfiltfilt(sos, random_signal)


def plot_graph(x, y, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
    ax.plot(x, y, linewidth=1)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=14)
    plt.grid()
    fig.savefig('./figures/' + title + '.png', dpi=600)
    plt.show()


def plot_2x2(x, y, title, xlabel, ylabel):
    fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
    k = 0
    for i in range(2):
        for j in range(2):
            ax[i][j].plot(x, y[k], linewidth=1)
            ax[i][j].set_title(f"Рівні M = {[4, 16, 64, 256][k]}")
            ax[i][j].grid()
            k += 1
    fig.supxlabel(xlabel, fontsize=14)
    fig.supylabel(ylabel, fontsize=14)
    fig.suptitle(title, fontsize=14)
    fig.savefig('./figures/' + title + '.png', dpi=600)
    plt.show()


quantized_signals = []
variances = []
snr_values = []

for M in [4, 16, 64, 256]:
    bits = []

    delta = (np.max(filtered_signal) - np.min(filtered_signal)) / (M - 1)

    quantize_signal = delta * np.round(filtered_signal / delta)
    quantized_signals.append(quantize_signal)

    quantize_levels = np.arange(np.min(quantize_signal), np.max(quantize_signal) + 1, delta)

    quantize_bit = np.arange(0, M)
    quantize_bit = [format(b, '0' + str(int(np.log(M) / np.log(2))) + 'b') for b in quantize_bit]

    quantize_table = np.c_[quantize_levels[:M], quantize_bit[:M]]

    fig, ax = plt.subplots(figsize=(14 / 2.54, M / 2.54))
    table = ax.table(cellText=quantize_table, colLabels=['Значення сигналу', 'Кодова послідовність'], loc='center')
    table.set_fontsize(14)
    table.scale(1, 2)
    ax.axis('off')
    fig.savefig('./figures/Таблиця квантування для М ' + str(M) + ' рівнів.png', dpi=600)
    plt.show()

    for signal_value in quantize_signal:
        for index, value in enumerate(quantize_levels[:M]):
            if np.round(np.abs(signal_value - value), 0) == 0:
                bits.append(quantize_bit[index])
                break

    bits = [int(item) for item in list(''.join(bits))]

    fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
    x = np.arange(0, len(bits))
    ax.step(x, bits, linewidth=0.1)
    ax.set_xlabel('Відліки', fontsize=14)
    ax.set_ylabel('Біти', fontsize=14)
    plt.title('Кодова послідовність М=' + str(M), fontsize=14)
    fig.savefig('./figures/Кодова послідовність М ' + str(M) + '.png', dpi=600)
    plt.show()

    E1 = filtered_signal - quantize_signal
    variances.append(np.var(E1))
    snr_values.append(np.var(filtered_signal) / np.var(E1))

plot_2x2(t, quantized_signals,
         "Цифрові сигнали з різними рівнями квантування",
         "Час (с)",
         "Амплітуда")

plt.figure(figsize=(21 / 2.54, 14 / 2.54))
plt.plot([4, 16, 64, 256], variances, linewidth=1)
plt.title("Залежність дисперсії від кількості рівнів квантування")
plt.xlabel("Кількість рівнів квантування M")
plt.ylabel("Дисперсія")
plt.grid()
plt.savefig('./figures/Залежність дисперсії.png', dpi=600)
plt.show()

plt.figure(figsize=(21 / 2.54, 14 / 2.54))
plt.plot([4, 16, 64, 256], snr_values, linewidth=1)
plt.title("Залежність співвідношення сигнал-шум")
plt.xlabel("Кількість рівнів квантування M")
plt.ylabel("SNR")
plt.grid()
plt.savefig('./figures/Залежність SNR.png', dpi=600)
plt.show()
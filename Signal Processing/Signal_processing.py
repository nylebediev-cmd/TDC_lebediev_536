import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fft

# -------------------------
#  (вариант 8)
# -------------------------
n = 500
Fs = 1000
F_max = 17


random_signal = np.random.normal(0, 10, n)


t = np.arange(n) / Fs


w = F_max / (Fs / 2)
sos = signal.butter(3, w, 'low', output='sos')


filtered_signal = signal.sosfiltfilt(sos, random_signal)


def plot_graph(x, y, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))
    ax.plot(x, y, linewidth=1)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=14)
    plt.grid()
    fig.savefig('./figures/' + title + '.png', dpi=600)
    plt.show()


plot_graph(t, filtered_signal,
           "Сигнал після фільтрації",
           "Час, с",
           "Амплітуда")


spectrum = fft.fft(filtered_signal)
spectrum = np.abs(fft.fftshift(spectrum))

freqs = fft.fftfreq(n, 1/Fs)
freqs = fft.fftshift(freqs)


plot_graph(freqs, spectrum,
           "Спектр сигналу",
           "Частота, Гц",
           "Амплітуда")
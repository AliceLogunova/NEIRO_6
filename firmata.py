import argparse
import csv
import time

import matplotlib.pyplot as plt
import numpy as np
from pyfirmata import Arduino, util  # type: ignore # get_pin('a:0:i'), read()->0..1, enable_reporting()
from scipy import signal

BANDS = [
    ("delta", 0.5, 4),
    ("theta", 4, 8),
    ("alpha", 8, 13),
    ("beta", 13, 30),
    ("gamma", 30, 45),
]  # при низкой частоте дискретизации gamma может быть недостоверна


def bandpower(freqs, psd, fmin, fmax):
    idx = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(idx):
        return 0.0
    return np.trapz(psd[idx], freqs[idx])


def main():
    p = argparse.ArgumentParser(description="Firmata analog reader + live plot + band powers")
    p.add_argument("--port", required=True, help="COM-порт, например COM3 или /dev/ttyACM0")
    p.add_argument("--pin", type=int, default=0, help="Аналоговый вход (0 = A0)")
    p.add_argument("--duration", type=float, default=0, help="Секунд (0 = бесконечно)")
    p.add_argument("--target-hz", type=float, default=100.0, help="Целевая частота опроса, Гц (регуляция цикла)")
    p.add_argument("--window", type=float, default=2.0, help="Окно БПФ/Уэлча, сек")
    p.add_argument("--csv", default="", help="Путь для сохранения CSV (опционально)")
    args = p.parse_args()

    print("Подключение к плате…")
    # StandardFirmata по умолчанию работает на 57600 бод; pyFirmata сам откроет порт
    board = Arduino(args.port)

    it = util.Iterator(board)
    it.start()  # поток чтения
    ain = board.get_pin(f"a:{args.pin}:i")  # 'a:номер:i' — аналоговый вход
    ain.enable_reporting()  # включить репортинг с платы

    # Прогрев, пока не появятся данные (pyFirmata отдаёт None, пока нет свежего значения)
    t0 = time.time()
    while ain.read() is None and time.time() - t0 < 3.0:
        time.sleep(0.01)

    ts, ys = [], []
    plt.ion()
    fig, (ax_ts, ax_bp) = plt.subplots(2, 1, figsize=(10, 6))
    (line_ts,) = ax_ts.plot([], [])
    ax_ts.set_title("Поток с A0 (0..1023)")
    ax_ts.set_xlabel("Время, с")
    ax_ts.set_ylabel("ADC")

    bars = ax_bp.bar([b for b, _, _ in BANDS], [0] * len(BANDS))
    ax_bp.set_title("Мощность по полосам (метод Уэлча)")
    ax_bp.set_ylabel("Мощность (отн.)")

    t_start = time.time()
    last_redraw = 0.0
    try:
        while True:
            v = ain.read()
            if v is not None:
                ts.append(time.time() - t_start)
                ys.append(v * 1023.0)  # pyFirmata нормирует 0..1 --> вернёмся к сырой ADC
            # простая регуляция целевой частоты цикла
            if args.target_hz > 0:
                time.sleep(max(0.0, 1.0 / args.target_hz - 0.0005))

            # обновление графиков ~10 раз/с
            now = time.time()
            if now - last_redraw > 0.1 and len(ts) > 2:
                # временной ряд (последние 10 с)
                tmax = ts[-1]
                tmin = max(0.0, tmax - 10.0)
                ax_ts.set_xlim(tmin, max(10.0, tmax))
                ax_ts.set_ylim(0, 1023)
                line_ts.set_data(ts, ys)

                # оценка частоты дискретизации по временным меткам
                if ts[-1] - ts[0] > 0:
                    fs = (len(ts) - 1) / (ts[-1] - ts[0])
                else:
                    fs = np.nan
                ax_ts.set_xlabel(f"Время, с   (оценка Fs ≈ {fs:.1f} Гц)")

                # спектральные полосы по последнему окну
                w_start = tmax - args.window
                idx0 = np.searchsorted(ts, max(0.0, w_start))
                yw = np.array(ys[idx0:], dtype=float)
                tw = np.array(ts[idx0:], dtype=float)

                if len(yw) > 16 and tw[-1] - tw[0] > 0:
                    fsw = (len(yw) - 1) / (tw[-1] - tw[0])
                    nperseg = min(256, len(yw))
                    f, Pxx = signal.welch(yw, fs=fsw, nperseg=nperseg)
                    for j, (_, f1, f2) in enumerate(BANDS):
                        bars[j].set_height(bandpower(f, Pxx, f1, f2))
                    ax_bp.relim()
                    ax_bp.autoscale_view()

                fig.canvas.draw()
                fig.canvas.flush_events()
                last_redraw = now

            if args.duration and ts and ts[-1] >= args.duration:
                break

    except KeyboardInterrupt:
        pass
    finally:
        if args.csv:
            with open(args.csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t_sec", "adc_0_1023"])
                w.writerows(zip(ts, ys))
            print(f"Сохранено: {args.csv}")
        if ts:
            fs = (len(ts) - 1) / (ts[-1] - ts[0])
            print(f"Собрано {len(ts)} отсчётов; оценка частоты дискретизации ≈ {fs:.1f} Гц")
        board.exit()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()

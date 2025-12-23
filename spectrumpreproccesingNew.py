import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# -----------------------------
# Globals
# -----------------------------
sample_x = sample_y = None
buffer_files = []

avg_buffer = None
avg_buffer_df = None

after_buffer = None
after_sg = None
after_gauss = None
after_median = None

plot_color = "#1f77b4"

# -----------------------------
# Utility functions
# -----------------------------
def read_csv(path):
    df = pd.read_csv(path)
    return df.iloc[:, 0].values.astype(float), df.iloc[:, 1].values.astype(float)


def normalize(y, mode):
    if y is None:
        return None
    if mode == "none":
        return y
    if mode == "minmax":
        return (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-12)
    if mode == "zscore":
        return (y - np.mean(y)) / (np.std(y) + 1e-12)
    return y


def wavelength_mask(x):
    wl_min = float(range_min.get())
    wl_max = float(range_max.get())
    return (x >= wl_min) & (x <= wl_max)

# -----------------------------
# Plot
# -----------------------------
def update_plot():
    fig.clear()
    axs = fig.subplots(2, 3)

    norm = norm_view.get()
    district = district_var.get()

    plots = [
        (sample_x, sample_y, "Original Sample"),
        (sample_x, avg_buffer, "Average Buffer"),
        (sample_x, after_buffer, "After Buffer Subtraction"),
        (sample_x, normalize(after_sg, norm), "Savitzky–Golay"),
        (sample_x, normalize(after_gauss, norm), "Gaussian"),
        (sample_x, normalize(after_median, norm), "Median"),
    ]

    for ax, (x, y, title) in zip(axs.flat, plots):
        if x is not None and y is not None:
            ax.plot(x, y, color=plot_color)
            ax.set_title(f"{title} ({district})")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Intensity")

    fig.tight_layout()
    canvas.draw()

# -----------------------------
# Load functions
# -----------------------------
def load_sample():
    global sample_x, sample_y
    path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not path:
        return
    sample_x, sample_y = read_csv(path)
    reset_processing()
    update_plot()


def load_buffers():
    global buffer_files
    buffer_files = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
    if buffer_files:
        messagebox.showinfo("Buffers Loaded", f"{len(buffer_files)} buffer files loaded")

# -----------------------------
# Processing
# -----------------------------
def reset_processing():
    global avg_buffer, avg_buffer_df
    global after_buffer, after_sg, after_gauss, after_median
    avg_buffer = avg_buffer_df = None
    after_buffer = after_sg = after_gauss = after_median = None


def compute_average_buffer():
    global avg_buffer, avg_buffer_df

    buffer_stack = []
    for path in buffer_files:
        bx, by = read_csv(path)
        buffer_stack.append(np.interp(sample_x, bx, by))

    avg_buffer = np.mean(buffer_stack, axis=0)
    avg_buffer_df = pd.DataFrame({
        "wavelength": sample_x,
        "avg_buffer_intensity": avg_buffer
    })


def apply_buffer_subtraction():
    global after_buffer
    if sample_x is None or not buffer_files:
        messagebox.showerror("Error", "Load sample and buffer files first")
        return

    compute_average_buffer()
    after_buffer = sample_y - avg_buffer
    apply_filters()


def apply_filters():
    global after_sg, after_gauss, after_median

    win = sg_window.get()
    if win % 2 == 0:
        win += 1

    after_sg = savgol_filter(after_buffer, win, sg_poly.get())
    after_gauss = gaussian_filter1d(after_buffer, gauss_sigma.get())

    k = median_kernel.get()
    if k % 2 == 0:
        k += 1
    after_median = medfilt(after_buffer, k)

    update_plot()

# -----------------------------
# Save
# -----------------------------
def save_csv(stage):
    y = {
        "After Buffer": after_buffer,
        "Savitzky–Golay": after_sg,
        "Gaussian": after_gauss,
        "Median": after_median,
    }.get(stage)

    if y is None:
        messagebox.showerror("Error", "Nothing to save")
        return

    mask = wavelength_mask(sample_x)
    district = district_var.get()

    df = pd.DataFrame({
        "wavelength": sample_x[mask],
        "intensity": normalize(y, norm_view.get())[mask],
        "district": district
    })

    path = filedialog.asksaveasfilename(defaultextension=".csv")
    if path:
        df.to_csv(path, index=False)
        messagebox.showinfo("Saved", "CSV saved successfully")

# -----------------------------
# Color Picker
# -----------------------------
def pick_color():
    global plot_color
    color = colorchooser.askcolor(title="Select Plot Color")[1]
    if color:
        plot_color = color
        update_plot()

# -----------------------------
# GUI
# -----------------------------
root = tk.Tk()
root.title("Fluorescence Spectra – Multi-Buffer Averaging")
root.geometry("1550x800")

control = tk.Frame(root, width=350)
control.pack(side="left", fill="y", padx=10)

tk.Button(control, text="Load SAMPLE", width=30, command=load_sample).pack(pady=3)
tk.Button(control, text="Load MULTIPLE BUFFERS", width=30, command=load_buffers).pack(pady=3)
tk.Button(control, text="Apply Avg Buffer Subtraction", width=30,
          command=apply_buffer_subtraction).pack(pady=8)

# District selection
tk.Label(control, text="Sample District").pack()
district_var = tk.StringVar(value="Nuwara Eliya")
tk.OptionMenu(control, district_var,
              "Nuwara Eliya", "Matale", "Ruhuna", "Dimbula", "Uva",
              command=lambda _: update_plot()).pack()

tk.Button(control, text="Pick Plot Color", width=30, command=pick_color).pack(pady=6)

tk.Label(control, text="Savitzky–Golay").pack()
sg_window = tk.Scale(control, from_=5, to=25, resolution=2, orient="horizontal")
sg_window.set(9)
sg_window.pack()
sg_poly = tk.Scale(control, from_=1, to=5, orient="horizontal")
sg_poly.set(2)
sg_poly.pack()

tk.Label(control, text="Gaussian σ").pack()
gauss_sigma = tk.Scale(control, from_=0.5, to=5.0, resolution=0.1, orient="horizontal")
gauss_sigma.set(1.0)
gauss_sigma.pack()

tk.Label(control, text="Median Kernel").pack()
median_kernel = tk.Scale(control, from_=3, to=21, resolution=2, orient="horizontal")
median_kernel.set(7)
median_kernel.pack()

tk.Button(control, text="Re-Apply Filters", width=30, command=apply_filters).pack(pady=6)

tk.Label(control, text="Visualization Normalization").pack(pady=6)
norm_view = tk.StringVar(value="none")
tk.Radiobutton(control, text="None", variable=norm_view, value="none", command=update_plot).pack()
tk.Radiobutton(control, text="Z-score", variable=norm_view, value="zscore", command=update_plot).pack()
tk.Radiobutton(control, text="Min–Max", variable=norm_view, value="minmax", command=update_plot).pack()

tk.Label(control, text="CSV Export λ range (nm)").pack(pady=6)
range_min = tk.Entry(control, width=10)
range_min.insert(0, "1000")
range_min.pack()
range_max = tk.Entry(control, width=10)
range_max.insert(0, "3000")
range_max.pack()

stage_var = tk.StringVar(value="Savitzky–Golay")
tk.OptionMenu(control, stage_var,
              "After Buffer", "Savitzky–Golay", "Gaussian", "Median").pack(pady=4)

tk.Button(control, text="Save CSV", width=30,
          command=lambda: save_csv(stage_var.get())).pack(pady=6)

plot_frame = tk.Frame(root)
plot_frame.pack(side="right", fill="both", expand=True)

fig = Figure(figsize=(9, 6), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack(fill="both", expand=True)

update_plot()
root.mainloop()

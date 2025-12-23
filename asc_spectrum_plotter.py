import tkinter as tk
from tkinter import filedialog, ttk, colorchooser
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

class CSVAveragePlotter:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV/ASC Average Spectrum Plotter with Error Bands")
        self.root.geometry("1400x900")

        self.csv_data = []
        self.filenames = []
        self.groups = {}
        self.selected_color = '#1f77b4'

        # ================= MAIN LAYOUT =================
        main = ttk.Frame(root)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, width=320)
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=6, pady=6)
        left.pack_propagate(False)

        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ================= CONTROLS =================
        ctrl = ttk.LabelFrame(left, text="Controls", padding=8)
        ctrl.pack(fill=tk.X)

        ttk.Button(ctrl, text="Upload CSV / ASC", command=self.upload_files).pack(fill=tk.X, pady=2)
        ttk.Button(ctrl, text="Clear All", command=self.clear_all).pack(fill=tk.X, pady=2)

        # ================= FILE LIST =================
        ttk.Label(left, text="Loaded Files").pack(anchor="w", pady=(6, 2))
        self.file_listbox = tk.Listbox(left, selectmode=tk.MULTIPLE, height=10)
        self.file_listbox.pack(fill=tk.BOTH, expand=True)

        # ================= GROUPING =================
        grp = ttk.LabelFrame(left, text="Grouping", padding=8)
        grp.pack(fill=tk.X, pady=6)

        self.group_entry = ttk.Entry(grp)
        self.group_entry.pack(fill=tk.X, pady=2)

        self.color_btn = tk.Button(grp, text="Pick Color", bg=self.selected_color,
                                   command=self.pick_color)
        self.color_btn.pack(fill=tk.X)

        ttk.Button(grp, text="Create / Update Group", command=self.assign_group).pack(fill=tk.X, pady=4)
        self.groups_listbox = tk.Listbox(grp, height=6)
        self.groups_listbox.pack(fill=tk.X)

        # ================= NORMALIZATION (ADDED) =================
        norm = ttk.LabelFrame(left, text="Normalization", padding=8)
        norm.pack(fill=tk.X, pady=6)

        self.norm_type = tk.StringVar(value="None")
        ttk.Combobox(
            norm,
            textvariable=self.norm_type,
            values=["None", "Global Max (All Groups)"],
            state="readonly"
        ).pack(fill=tk.X)

        self.norm_type.trace_add("write", lambda *a: self.plot_data())

        # ================= PLOT =================
        self.fig = Figure(figsize=(12, 7))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Intensity")
        self.ax.set_title("Average Spectral Data ±1σ")
        self.ax.grid(alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, right)

    # ================= FILE LOADING =================
    def load_asc_file(self, path):
        data = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or ':' in line:
                    continue
                try:
                    w, i = map(float, line.split()[:2])
                    data.append([w, i])
                except:
                    pass
        return pd.DataFrame(data, columns=["wl", "int"])

    def upload_files(self):
        files = filedialog.askopenfilenames(
            filetypes=[("Spectra", "*.csv *.asc *.txt")]
        )
        for f in files:
            if f.lower().endswith(".csv"):
                df = pd.read_csv(f, header=None).iloc[:, :2]
            else:
                df = self.load_asc_file(f)

            df.columns = ["wl", "int"]
            self.csv_data.append(df.dropna())
            self.file_listbox.insert(tk.END, f.split("/")[-1])

    # ================= GROUPING =================
    def pick_color(self):
        c = colorchooser.askcolor(initialcolor=self.selected_color)
        if c[1]:
            self.selected_color = c[1]
            self.color_btn.config(bg=c[1])

    def assign_group(self):
        sel = self.file_listbox.curselection()
        if not sel:
            return
        name = self.group_entry.get() or f"Group {len(self.groups)+1}"
        self.groups[name] = {"indices": list(sel), "color": self.selected_color}
        if name not in self.groups_listbox.get(0, tk.END):
            self.groups_listbox.insert(tk.END, name)
        self.plot_data()

    # ================= PLOTTING =================
    def plot_data(self):
        self.ax.clear()
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Intensity")
        self.ax.set_title("Average Spectral Data ±1σ")
        self.ax.grid(alpha=0.3)

        if not self.groups:
            self.canvas.draw()
            return

        group_means = {}
        group_stds = {}
        grids = {}

        # ---- compute means/stds first ----
        for g, info in self.groups.items():
            idxs = info["indices"]
            wl_all = np.concatenate([self.csv_data[i].wl.values for i in idxs])
            grid = np.linspace(wl_all.min(), wl_all.max(), 1000)

            spectra = []
            for i in idxs:
                df = self.csv_data[i]
                f = interp1d(df.wl, df.int, bounds_error=False, fill_value=np.nan)
                spectra.append(f(grid))

            arr = np.array(spectra)
            group_means[g] = np.nanmean(arr, axis=0)
            group_stds[g] = np.nanstd(arr, axis=0)
            grids[g] = grid

        # ================= GLOBAL MAX NORMALIZATION (KEY PART) =================
        if self.norm_type.get() == "Global Max (All Groups)":
            global_max = np.nanmax(np.concatenate(list(group_means.values())))
            if global_max != 0:
                for g in group_means:
                    group_means[g] /= global_max
                    group_stds[g] /= global_max

        # ---- plot ----
        for g in group_means:
            color = self.groups[g]["color"]
            self.ax.plot(grids[g], group_means[g], label=g, color=color, linewidth=1)
            self.ax.fill_between(
                grids[g],
                group_means[g] - group_stds[g],
                group_means[g] + group_stds[g],
                color=color,
                alpha=0.3
            )

        self.ax.legend()
        self.canvas.draw()

    # ================= CLEAR =================
    def clear_all(self):
        self.csv_data.clear()
        self.groups.clear()
        self.file_listbox.delete(0, tk.END)
        self.groups_listbox.delete(0, tk.END)
        self.ax.clear()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    CSVAveragePlotter(root)
    root.mainloop()

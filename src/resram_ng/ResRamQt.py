# Compilation mode, support OS-specific options
# nuitka-project-if: {OS} in ("Windows", "Linux", "Darwin", "FreeBSD"):
#    nuitka-project: --onefile
# nuitka-project-else:
#    nuitka-project: --mode=standalonealone

# The PySide6 plugin covers qt-plugins
# nuitka-project: --enable-plugin=pyqt6
# nuitka-project: --include-qt-plugins=qml
from PyQt6.QtCore import (
    Qt,
    QThreadPool,
    pyqtSlot,
    QRunnable,
    pyqtSignal,
    QTimer,
    QObject,
)
from PyQt6.QtWidgets import (
    QMessageBox,
    QLabel,
    QFileDialog,
    QCheckBox,
    QHeaderView,
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QHBoxLayout,
    QSpacerItem,
    QSizePolicy,
)
from PyQt6.QtGui import QIcon
import pyqtgraph as pg
import sys
from datetime import datetime
import os
from math import factorial
import numpy as np
import lmfit
# import toml
from .resram_core import (
    load_input,
    get_default_example_dir,
    g,
    A,
    R,
    cross_sections,
    resram_data,
    param_init,
    raman_residual,
    ensure_loss_coefficients,
    run_save,
)


"""    
    output_data = {
        "parameters": {
            "E00": obj.E0,
            "gamma": obj.gamma,
            "theta": obj.theta,
            "M": obj.M,
            "n": obj.n,
            "T": obj.T,
            "s_reorg": obj.s_reorg,
            "w_reorg": obj.w_reorg,
            "reorg": obj.reorg
        },
        "boltzmann": {
            "coefficients": obj.boltz_coef,
            "states": obj.boltz_state
        }
    }

    # Save the output data to a TOML file
    with open(current_time_str + "_data/output.toml", 'w') as toml_file:
        toml.dump(output_data, toml_file)    

    with open(current_time_str + "_data/inp_new.txt", 'w') as file:
        # Write the data to the file
        file.write(f"{obj.gamma} # gamma linewidth parameter (cm^-1)\n")
        file.write(
            f"{obj.theta} # theta static inhomogeneous linewidth parameter (cm^-1)\n")
        file.write(f"{obj.E0} # E0 (cm^-1)\n")
        file.write(f"{obj.k} # kappa solvent parameter\n")
        file.write(f"{obj.ts} # time step (ps)\n")
        file.write(f"{obj.ntime} # number of time steps\n")
        file.write(
            f"{obj.EL_reach} # range plus and minus E0 to calculate lineshapes\n")
        file.write(f"{obj.M} # transition length M (Angstroms)\n")
        file.write(f"{obj.n} # refractive index n\n")
        file.write(f"{obj.inp[9]} # start raman shift axis (cm^-1)\n")
        file.write(f"{obj.inp[10]} # end raman shift axis (cm^-1)\n")
        file.write(f"{obj.inp[11]} # rshift axis step size (cm^-1)\n")
        file.write(f"{obj.inp[12]} # raman spectrum resolution (cm^-1)\n")
        file.write(f"{obj.T} # Temperature (K)\n")
        file.write(
            f"{obj.inp[14]} # convergence for sums # no effect since order > 1 broken\n")
        file.write(f"{obj.inp[15]} # Boltz Toggle\n")
"""
# return resram_data(current_time_str + "_data")

class WorkerSignals(QObject):
    """Signals to be used in the Worker class

    Args:
        QObject (_type_): _description_
    """

    result_ready = pyqtSignal(str)
    finished = pyqtSignal(object)


class CalcSignals(QObject):
    """Signals for background calculation updates
    """
    finished = pyqtSignal(tuple)
    error = pyqtSignal(str)


class CalcWorker(QRunnable):
    """Worker for background cross_section calculations
    """
    def __init__(self, obj_load: load_input):
        super().__init__()
        self.obj_load = obj_load
        self.signals = CalcSignals()

    @pyqtSlot()
    def run(self):
        try:
            # Perform heavy calculations
            abs_cross, fl_cross, raman_cross, _, _ = cross_sections(self.obj_load)
            
            # Calculate raman_spec
            raman_spec = np.zeros((len(self.obj_load.rshift), len(self.obj_load.rpumps)))
            for i in range(len(self.obj_load.rpumps)):
                for l in np.arange(len(self.obj_load.wg)):
                    raman_spec[:, i] += (
                        np.real((raman_cross[l, self.obj_load.rp[i]]))
                        * (1 / np.pi)
                        * (0.5 * self.obj_load.res)
                        / (
                            (self.obj_load.rshift - self.obj_load.wg[l]) ** 2
                            + (0.5 * self.obj_load.res) ** 2
                        )
                    )
            
            self.signals.finished.emit((abs_cross, fl_cross, raman_cross, raman_spec))
        except Exception as e:
            self.signals.error.emit(str(e))


class Worker(QRunnable):
    """Worker class to run the fitting in a separate thread

    Args:
        QRunnable (_type_): _description_
    """

    def __init__(self, obj_load:load_input, tolerance:float, maxnfev:int, fit_alg:str, fit_switch:np.ndarray) -> None:
        """Initialize the Worker class

        Args:
            obj_load (load_input): load_input object containing all the parameters for the simulation
            tolerance (float): Tolerance for the fitting
            maxnfev (int): Maximum number of function evaluations
            fit_alg (str): Fitting algorithm to be used
            fit_switch (np.ndarray): Array of 0|1 for each parameter
        """
        super().__init__()
        self.signals = WorkerSignals()
        self.obj_load = obj_load
        self.tolerance = tolerance
        self.maxnfev = maxnfev
        self.fit_alg = fit_alg
        self.fit_switch = fit_switch

    @pyqtSlot()
    def run(self):
        """Run the fitting in a separate thread
        """
        # global delta, M, gamma, maxnfev, tolerance, fit_alg
        params_lmfit = param_init(self.fit_switch, self.obj_load)

        print("Fit is running, please wait...\n")
        fit_kws = dict(tol=self.tolerance)
        try:
            result = lmfit.minimize(
                raman_residual,
                params_lmfit,
                args=(self.obj_load,),
                method=self.fit_alg,
                **fit_kws,
                max_nfev=self.maxnfev,
            )  # max_nfev = 10000000, **fit_kws
        except Exception as e:
            print(
                "Something went wrong before fitting start. Use powell algorithm instead"
                + str(e)
            )
            result = lmfit.minimize(
                raman_residual,
                params_lmfit,
                args=(self.obj_load,),
                method="powell",
                **fit_kws,
                max_nfev=self.maxnfev,
            )
        print(lmfit.fit_report(result))
        for i in range(len(self.obj_load.delta)):
            self.obj_load.delta[i] = result.params.valuesdict()["delta" + str(i)]
        self.obj_load.gamma = result.params.valuesdict()["gamma"]
        self.obj_load.M = result.params.valuesdict()["transition_length"]
        self.obj_load.k = result.params.valuesdict()["kappa"]  # kappa parameter
        self.obj_load.theta = result.params.valuesdict()["theta"]  # kappa parameter
        self.obj_load.E0 = result.params.valuesdict()["E0"]  # kappa parameter
        run_save(self.obj_load)
        print("Fit done\n")
        self.signals.result_ready.emit("Fit done")
        self.signals.finished.emit(self.obj_load)


class SpectrumApp(QMainWindow):
    """Main class for the GUI
    """
    def __init__(self):
        super().__init__()
        self.dir = get_default_example_dir()
        # multithread
        self.threadpool = QThreadPool()
        # print("Multithreading with maximum %d threads" %self.threadpool.maxThreadCount())
        self.load_files()
        self.plot_switch = np.ones(len(self.obj_load.delta) + 18)
        self.fit_switch = np.ones(len(self.obj_load.delta) + 18)
        self.fit_alg = "powell"
        self.maxnfev = 100
        self.tolerance = 0.00000001
        self.setWindowTitle("Raman Spectrum Analyzer")
        self.setGeometry(100, 100, 960, 540)
        
        # Debounce timer
        self.update_debounce_timer = QTimer(self)
        self.update_debounce_timer.setSingleShot(True)
        self.update_debounce_timer.timeout.connect(self.trigger_calculation)
        
        # Plot item storage for efficient updates
        self.raman_plot_items = []
        self.rep_plot_items = []
        self.rep_scatter_items = []
        self.abs_plot_item = None
        self.fl_plot_item = None
        self.abs_exp_plot_item = None
        self.fl_exp_plot_item = None

        # main layout horizontal
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        # left layout vertical
        self.left_layout = QVBoxLayout()
        # Calculate the figure size in inches based on pixels and screen DPI
        # dpi = self.physicalDpiX()  # Get the screen's DPI
        # fig_width_pixels = 1280  # Desired figure width in pixels
        # fig_height_pixels = 720  # Desired figure height in pixels
        # fig_width = fig_width_pixels / dpi
        # fig_height = fig_height_pixels / dpi
        """
        self.canvas = FigureCanvas(plt.figure(
            figsize=(fig_width, fig_height)))  # fig profs
        self.canvas2 = FigureCanvas(plt.figure(
            figsize=(fig_width/2, fig_height)))  # fig raman spec
        self.canvas3 = FigureCanvas(plt.figure(
            figsize=(fig_width/2, fig_height)))  # fig abs
            """
        # Initialize PlotWidgets
        self.canvas = pg.PlotWidget()  # fig profs
        self.canvas.addLegend(colCount=2)
        self.canvas.setTitle("Raman Excitation Profiles")
        # self.ax.set_xlim(self.profs_xmin, self.profs_xmax)
        self.canvas.setLabel("bottom", "Wavenumber (cm-1)")
        self.canvas.setLabel(
            "left", "Raman Cross Section \n(1e-14 Angstrom**2/Molecule)"
        )
        self.canvas2 = pg.PlotWidget()  # fig raman spec
        self.canvas2.addLegend(offset=(-30, 30))
        self.canvas2.setTitle("Raman Spectra")
        # self.canvas2.set_xlim(self.raman_xmin, self.raman_xmax)
        self.canvas2.setLabel("bottom", "Raman Shift (cm-1)")
        self.canvas2.setLabel(
            "left", "Raman Cross Section \n(1e-14 Angstrom**2/Molecule)"
        )
        self.canvas3 = pg.PlotWidget()  # fig abs
        self.canvas3.addLegend()
        self.canvas3.setTitle("Absorption and Emission Spectra")
        # self.ax3.set_xlim(self.abs_xmin, self.abs_xmax)
        self.canvas3.setLabel("bottom", "Wavenumber (cm-1)")
        self.canvas3.setLabel("left", "Cross Section \n(1e-14 Angstrom**2/Molecule)")
        # self.ax3.set_ylabel('Cross Section \n(1e-14 Angstrom**2/Molecule)')
        self.canvas.setBackground("white")
        self.canvas2.setBackground("white")
        self.canvas3.setBackground("white")
        self.cm = pg.colormap.get("CET-R4")

        self.left_layout.addWidget(self.canvas, 5)
        self.main_layout.addLayout(self.left_layout, 3)
        self.left_bottom_layout = QHBoxLayout()
        self.left_bottom_layout.addWidget(self.canvas2, 7)
        self.left_bottom_layout.addWidget(self.canvas3, 3)
        self.left_layout.addLayout(self.left_bottom_layout, 3)

        # self.left_layout.addWidget(self.output_logger, 1)  # Use a stretch factor of 1.5
        self.right_layout = QVBoxLayout()

        # self.right_layout.addWidget(self.table_widget) #included in create_variable_table

        self.main_layout.addLayout(self.right_layout, 1)
        self.create_buttons()
        self.create_variable_table()
        # timer for updating plot
        self.update_timer = QTimer(self)
        self.plot_data()
        print("Initialized")
        self.showMaximized()

    def refresh_dir_indicator(self):
        """Sync current-directory label with the active loaded folder."""
        # Use the loader's resolved directory when available.
        if hasattr(self, "obj_load") and hasattr(self.obj_load, "dir"):
            self.dir = self.obj_load.dir
        else:
            self.dir = os.path.abspath(self.dir)

        if not self.dir.endswith(os.sep):
            self.dir += os.sep

        if hasattr(self, "dirlabel"):
            self.dirlabel.setText("Current data folder: " + self.dir)

    def sendto_table(self):
        """Send the data to the table
        """
        self.render_table_from_state()
        self.plot_data()

    def load_table(self):
        """Load the data from the table
        """
        self.apply_table_to_state()

    def clear_canvas(self):
        """Clear the canvas
        """
        if self.canvas is not None:
            self.canvas.clear()  # Redraw the canvas to clear it

        if self.canvas2 is not None:
            self.canvas2.clear()  # Redraw the canvas to clear it

        if self.canvas3 is not None:
            self.canvas3.clear()  # Redraw the canvas to clear it

    def reset_plot_items(self):
        """Clear plot canvases and cached plot item references for a new dataset."""
        self.clear_canvas()
        self.raman_plot_items = []
        self.rep_plot_items = []
        self.rep_scatter_items = []
        self.abs_plot_item = None
        self.fl_plot_item = None
        self.abs_exp_plot_item = None
        self.fl_exp_plot_item = None

    def start_update_timer(self):
        """Start the timer to update the plot
        """
        self.update_timer.start(3000)  # 3 seconds (in milliseconds)

    def stop_update_timer(self):
        """Stop the timer to update the plot
        """
        self.update_timer.stop()

    def fit(self):
        """Fit the data
        """
        self.load_table()
        print("Initial deltas: " + str(self.obj_load.delta))
        self.worker = Worker(
            self.obj_load, self.tolerance, self.maxnfev, self.fit_alg, self.fit_switch
        )  # thread for fitting
        self.fit_button.setEnabled(False)
        self.fit_button.setText("Fit Running")
        self.threadpool.start(self.worker)
        # after fitting, global variables should be optimized values
        # self.plot_data()
        self.worker.signals.finished.connect(self.handle_worker_result)
        self.worker.signals.result_ready.connect(self.update_fit)

    @pyqtSlot(object)
    def handle_worker_result(self, result_object:load_input):
        """Handle the result object received from the worker
        Args:
            result_object (load_input): The result object received from the worker
        """
        self.obj_load = result_object
        print("Received fitting result")
        # You can use the result_object in the main window as needed

    def update_fit(self, result):
        self.fit_button.setText("Fit")
        self.fit_button.setEnabled(True)  # Re-enable the button
        self.sendto_table()

    def on_toggle(self, state:bool):
        """Toggle the update timer on or off
        Args:
            state (bool): True to start the timer, False to stop it
        """
        if state:
            self.start_update_timer()
            self.update_timer.timeout.connect(self.sendto_table)
        else:
            self.update_timer.timeout.disconnect(self.sendto_table)
            self.stop_update_timer()

    def trigger_calculation(self):
        """Start background calculation if not already running
        """
        # We can optimize this by cancelling existing worker if possible, 
        # but for now we just start a new one.
        worker = CalcWorker(self.obj_load)
        worker.signals.finished.connect(self.on_calc_finished)
        worker.signals.error.connect(lambda e: print(f"Calculation error: {e}"))
        self.threadpool.start(worker)

    @pyqtSlot(tuple)
    def on_calc_finished(self, results):
        """Called when background calculation finishes
        """
        self.update_plots(results)

    def update_plots(self, results):
        """Update plots with calculated data using efficient setData calls
        """
        abs_cross, fl_cross, raman_cross, raman_spec = results
        fl_exp_y = self.normalized_fl_exp_y(fl_cross)
        
        # 1. Update Raman Spectra (canvas2)
        if not self.raman_plot_items:
            self.canvas2.clear()
            for i in range(len(self.obj_load.rpumps)):
                nm = 1e7 / self.obj_load.rpumps[i]
                pen = self.cm[i / len(self.obj_load.rpumps)]
                line = self.canvas2.plot(
                    self.obj_load.rshift,
                    np.real(raman_spec[:, i]),
                    pen=pen,
                    name=f"{nm:.3f} nm laser",
                )
                line.setDownsampling(ds=True, auto=True, method="subsample")
                self.raman_plot_items.append(line)
        else:
            for i, line in enumerate(self.raman_plot_items):
                line.setData(self.obj_load.rshift, np.real(raman_spec[:, i]))

        # 2. Update Raman Excitation Profiles (canvas)
        # Handle scatters (experimental)
        # For simplicity and correctness with toggle, we clear scatters and re-add them
        # if the total count doesn't match expected visible count.
        visible_modes_count = sum(self.plot_switch[:len(self.obj_load.wg)])
        if len(self.rep_scatter_items) != visible_modes_count * len(self.obj_load.rpumps):
            for item in self.rep_scatter_items:
                self.canvas.removeItem(item)
            self.rep_scatter_items = []
            for i in range(len(self.obj_load.rpumps)):
                for j in range(len(self.obj_load.wg)):
                    if self.plot_switch[j] == 1:
                        pen = self.cm[j / len(self.obj_load.wg)]
                        scatter = self.canvas.scatterPlot(
                            [self.obj_load.convEL[self.obj_load.rp[i]]],
                            [self.obj_load.profs_exp[j, i]],
                            symbol="o",
                            pen=pen,
                        )
                        scatter.setSymbolBrush(pen)
                        self.rep_scatter_items.append(scatter)

        # Handle lines (calculated)
        if not self.rep_plot_items:
            # Create ALL potential lines once
            for j in range(len(self.obj_load.wg)):
                pen = self.cm[j / len(self.obj_load.wg)]
                line = self.canvas.plot(
                    self.obj_load.convEL,
                    np.real(np.transpose(raman_cross))[:, j],
                    pen=pen,
                    name=f"{self.obj_load.wg[j]:.2f} cm-1",
                )
                line.setDownsampling(ds=True, auto=True, method="subsample")
                self.rep_plot_items.append(line)
        
        # Update data and visibility for all lines
        for j, line in enumerate(self.rep_plot_items):
            if self.plot_switch[j] == 1:
                line.setData(self.obj_load.convEL, np.real(np.transpose(raman_cross))[:, j])
                line.show()
            else:
                line.hide()

        # 3. Update Absorption/FL (canvas3)
        if self.abs_plot_item is None:
            self.canvas3.clear()
            self.abs_plot_item = self.canvas3.plot(
                self.obj_load.convEL, np.real(abs_cross), name="Abs", pen="red"
            )
            self.fl_plot_item = self.canvas3.plot(
                self.obj_load.convEL, np.real(fl_cross), name="FL", pen="green"
            )
            
            try:
                self.abs_exp_plot_item = self.canvas3.plot(
                    self.obj_load.abs_exp_raw[:, 0],
                    self.obj_load.abs_exp_raw[:, 1],
                    name="Abs expt.",
                    pen="blue",
                )
            except: pass
            
            try:
                if fl_exp_y is not None:
                    self.fl_exp_plot_item = self.canvas3.plot(
                        self.obj_load.fl_exp_raw[:, 0],
                        fl_exp_y,
                        name="FL expt.",
                        pen="yellow",
                    )
            except: pass
        else:
            self.abs_plot_item.setData(self.obj_load.convEL, np.real(abs_cross))
            self.fl_plot_item.setData(self.obj_load.convEL, np.real(fl_cross))
            if self.abs_exp_plot_item:
                self.abs_exp_plot_item.setData(self.obj_load.abs_exp_raw[:, 0], self.obj_load.abs_exp_raw[:, 1])
            if self.fl_exp_plot_item and fl_exp_y is not None:
                self.fl_exp_plot_item.setData(self.obj_load.fl_exp_raw[:, 0], fl_exp_y)

    def normalized_fl_exp_y(self, fl_cross):
        """Scale experimental fluorescence to the calculated fluorescence maximum."""
        if not hasattr(self.obj_load, "fl_exp_raw"):
            return None
        raw = self.obj_load.fl_exp_raw[:, 1]
        raw_max = np.max(raw)
        calc_max = np.max(np.real(fl_cross))
        if raw_max == 0 or calc_max == 0:
            return raw
        return raw * (calc_max / raw_max)

    def plot_data(self):
        """Old blocking plot_data, now redirects to trigger_calculation
        """
        self.trigger_calculation()

    def create_variable_table(self):
        """Create the variable table
        """
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(4)
        self.table_widget.setHorizontalHeaderLabels(
            ["Variables", "Values", "Plot Raman \nEx. Profile", "Fit?"]
        )
        self.table_widget.itemChanged.connect(self.update_spectrum)
        self.render_table_from_state()
        print("Initialized. Files loaded from the working folder.")
        # Set headers to resize to contents
        self.table_widget.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table_widget.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )

        self.right_layout.addWidget(self.table_widget)

    def reset_table_for_loaded_data(self):
        """Resize table state for the currently loaded dataset."""
        self.plot_switch = np.ones(len(self.obj_load.delta) + 18)
        self.fit_switch = np.ones(len(self.obj_load.delta) + 18)
        self.fit_alg = "powell"
        self.maxnfev = 100
        self.tolerance = 0.00000001
        self.render_table_from_state()

    def select_subfolder(self):
        """Select the subfolder
        """
        self.folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Subfolder",
            os.getcwd(),
            options=QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.ReadOnly,
        )

        if self.folder_path:
            print("Selected folder:", self.folder_path)
            self.dir = self.folder_path + "/"
            try:
                self.obj_load = load_input(self.dir)
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Folder Load Failed",
                    f"Could not load required files from:\n{self.folder_path}\n\n{e}",
                )
                return
            
            self.reset_plot_items()
            
            self.reset_table_for_loaded_data()
            self.refresh_dir_indicator()
            self.plot_data()

    def create_buttons(self):
        """Create the buttons
        """
        # self.add_button = QPushButton("Add Data")
        self.update_button = QPushButton("Update table")
        self.save_button = QPushButton("Save parameters")
        self.initialize_button = QPushButton("Intialize")
        self.fit_button = QPushButton("Fit")
        self.load_button = QPushButton("Load folder")

        # self.add_button.clicked.connect(self.add_data)
        self.update_button.clicked.connect(self.sendto_table)
        self.save_button.clicked.connect(self.save_data)
        self.initialize_button.clicked.connect(self.initialize)
        self.fit_button.clicked.connect(self.fit)
        self.load_button.clicked.connect(self.select_subfolder)

        # toggle switch
        self.updater_switch = QCheckBox("Auto Refresh")
        self.updater_switch.setCheckable(True)
        self.updater_switch.setStyleSheet(
            "QCheckBox::indicator { width: 40px; height: 20px; }"
        )
        self.updater_switch.toggled.connect(self.on_toggle)
        button_layout = QHBoxLayout()

        # Add a stretch to push widgets to the right
        spacer = QSpacerItem(
            40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        self.dirlabel = QLabel("Current data folder: " + self.dir)
        button_layout.addWidget(self.dirlabel)
        self.refresh_dir_indicator()
        button_layout.addItem(spacer)
        # button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.update_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.initialize_button)
        button_layout.addWidget(self.fit_button)
        button_layout.addWidget(self.updater_switch)
        # button_layout.addWidget(self.delete_button)
        self.left_layout.addLayout(button_layout)

    def update_spectrum(self):
        """Clear the previous series data and trigger debounced calculation
        """
        self.load_table()
        # Use debounce timer to prevent lag while typing
        self.update_debounce_timer.start(300) # 300ms delay

    """
    def add_data(self):#no use
        # Add a new row to the table
        row_position = self.table_widget.rowCount()
        self.table_widget.insertRow(row_position)
        self.table_widget.setItem(
            row_position, 0, QTableWidgetItem("New Data"))
        self.table_widget.setItem(row_position, 1, QTableWidgetItem("0.0"))
        self.update_spectrum()
        """

    def save_data(self):
        run_save(self.obj_load)

    def load_files(self):
        """Load the files from the directory
        """
        self.obj_load = load_input(self.dir)
        self.refresh_dir_indicator()
        return self.obj_load

    def initialize(self):
        """Initialize the GUI
        """
        self.dir = get_default_example_dir()
        self.load_files()
        
        self.reset_plot_items()
        
        self.reset_table_for_loaded_data()
        print("Initialized. Files loaded from the working folder.")
        self.plot_data()
        self.refresh_dir_indicator()

    def table_param_rows(self):
        """Return parameter rows after the mode rows."""
        loss_coefficients = ensure_loss_coefficients(self.obj_load)
        return [
            ("gamma", "gamma", self.obj_load.gamma, len(self.obj_load.delta)),
            ("M", "Transition Length (A)", self.obj_load.M, len(self.obj_load.delta) + 1),
            ("theta", "theta", self.obj_load.theta, len(self.obj_load.delta) + 2),
            ("kappa", "kappa", self.obj_load.k, len(self.obj_load.delta) + 3),
            ("n", "Refractive Index", self.obj_load.n, None),
            ("E0", "E00", self.obj_load.E0, len(self.obj_load.delta) + 5),
            ("ts", "Time step (ps)", self.obj_load.ts, None),
            ("ntime", "Time step number", self.obj_load.ntime, None),
            ("fit_alg", "Fitting algorithm", self.fit_alg, None),
            ("maxnfev", "Fitting maxnfev", self.maxnfev, None),
            ("tolerance", "Fitting tolerance", self.tolerance, None),
            ("T", "Temp (K)", self.obj_load.T, None),
            ("raman_maxcalc", "Raman maxcalc", getattr(self.obj_load, "raman_maxcalc", self.obj_load.inp[10]), None),
            ("EL_reach", "EL reach", getattr(self.obj_load, "EL_reach", self.obj_load.inp[6]), None),
            ("loss_total_sigma", "Loss coeff total_sigma", loss_coefficients[0], None),
            ("loss_abs_corr", "Loss coeff (1-abs_corr)", loss_coefficients[1], None),
            ("loss_fl_corr", "Loss coeff (1-fl_corr)", loss_coefficients[2], None),
        ]

    def row_for_key(self, key: str) -> int:
        """Return the current table row for a stable row key."""
        for row in range(self.table_widget.rowCount()):
            item = self.table_widget.item(row, 0)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == key:
                return row
        raise KeyError(key)

    def set_keyed_item(self, row: int, column: int, text: str, key: str | None = None):
        item = QTableWidgetItem(text)
        if key is not None:
            item.setData(Qt.ItemDataRole.UserRole, key)
        self.table_widget.setItem(row, column, item)
        return item

    def set_check_item(self, row: int, column: int, checked: bool | None):
        item = QTableWidgetItem()
        if checked is None:
            item.setText("N/A")
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        else:
            item.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.table_widget.setItem(row, column, item)
        return item

    def ensure_switch_lengths(self):
        needed = len(self.obj_load.delta) + 18
        if len(self.plot_switch) != needed:
            self.plot_switch = np.ones(needed)
        if len(self.fit_switch) != needed:
            self.fit_switch = np.ones(needed)

    def render_table_from_state(self):
        """Render a complete table snapshot from obj_load and GUI control state."""
        self.ensure_switch_lengths()
        rows = len(self.obj_load.delta) + len(self.table_param_rows())
        was_blocked = self.table_widget.blockSignals(True)
        try:
            self.table_widget.clearContents()
            self.table_widget.setRowCount(rows)

            for row in range(len(self.obj_load.delta)):
                key = f"delta:{row}"
                self.set_keyed_item(row, 0, f"delta@{self.obj_load.wg[row]:.2f} cm-1", key)
                self.set_keyed_item(row, 1, f"{self.obj_load.delta[row]:.4f}")
                self.set_check_item(row, 2, self.plot_switch[row] == 1)
                self.set_check_item(row, 3, self.fit_switch[row] == 1)

            start = len(self.obj_load.delta)
            for offset, (key, label, value, fit_index) in enumerate(self.table_param_rows()):
                row = start + offset
                self.set_keyed_item(row, 0, label, key)
                self.set_keyed_item(row, 1, str(value))
                self.set_check_item(row, 2, None)
                if fit_index is None:
                    self.set_check_item(row, 3, None)
                else:
                    self.set_check_item(row, 3, self.fit_switch[fit_index] == 1)
        finally:
            self.table_widget.blockSignals(was_blocked)

    def table_text(self, key: str) -> str:
        item = self.table_widget.item(self.row_for_key(key), 1)
        return "" if item is None else item.text()

    def restore_table_value(self, key: str, value):
        row = self.row_for_key(key)
        was_blocked = self.table_widget.blockSignals(True)
        try:
            self.set_keyed_item(row, 1, str(value))
        finally:
            self.table_widget.blockSignals(was_blocked)

    def read_float_value(self, key: str, current_value):
        text = self.table_text(key)
        try:
            return float(text)
        except ValueError:
            print(f"Invalid numeric value for {key}: {text!r}. Restoring {current_value}.")
            self.restore_table_value(key, current_value)
            return current_value

    def read_int_value(self, key: str, current_value: int):
        text = self.table_text(key)
        try:
            return int(text)
        except ValueError:
            print(f"Invalid integer value for {key}: {text!r}. Restoring {current_value}.")
            self.restore_table_value(key, current_value)
            return current_value

    def check_state_for_key(self, key: str, column: int) -> bool:
        item = self.table_widget.item(self.row_for_key(key), column)
        return item is not None and item.checkState() == Qt.CheckState.Checked

    def apply_table_to_state(self):
        """Apply editable table values back to obj_load using stable row keys."""
        self.ensure_switch_lengths()

        for i in range(len(self.obj_load.delta)):
            key = f"delta:{i}"
            self.obj_load.delta[i] = self.read_float_value(key, self.obj_load.delta[i])
            self.plot_switch[i] = 1 if self.check_state_for_key(key, 2) else 0
            self.fit_switch[i] = 1 if self.check_state_for_key(key, 3) else 0

        self.obj_load.gamma = self.read_float_value("gamma", self.obj_load.gamma)
        self.fit_switch[len(self.obj_load.delta)] = 1 if self.check_state_for_key("gamma", 3) else 0
        self.obj_load.M = self.read_float_value("M", self.obj_load.M)
        self.fit_switch[len(self.obj_load.delta) + 1] = 1 if self.check_state_for_key("M", 3) else 0
        self.obj_load.theta = self.read_float_value("theta", self.obj_load.theta)
        self.fit_switch[len(self.obj_load.delta) + 2] = 1 if self.check_state_for_key("theta", 3) else 0
        self.obj_load.k = self.read_float_value("kappa", self.obj_load.k)
        self.fit_switch[len(self.obj_load.delta) + 3] = 1 if self.check_state_for_key("kappa", 3) else 0
        self.obj_load.n = self.read_float_value("n", self.obj_load.n)
        self.obj_load.E0 = self.read_float_value("E0", self.obj_load.E0)
        self.fit_switch[len(self.obj_load.delta) + 5] = 1 if self.check_state_for_key("E0", 3) else 0
        self.obj_load.ts = self.read_float_value("ts", self.obj_load.ts)
        self.obj_load.ntime = self.read_float_value("ntime", self.obj_load.ntime)
        self.obj_load.T = self.read_float_value("T", self.obj_load.T)
        self.obj_load.raman_maxcalc = self.read_float_value(
            "raman_maxcalc", getattr(self.obj_load, "raman_maxcalc", float(self.obj_load.inp[10]))
        )
        self.obj_load.EL_reach = self.read_float_value(
            "EL_reach", getattr(self.obj_load, "EL_reach", float(self.obj_load.inp[6]))
        )
        self.fit_alg = self.table_text("fit_alg") or self.fit_alg
        self.maxnfev = self.read_int_value("maxnfev", self.maxnfev)
        self.tolerance = self.read_float_value("tolerance", self.tolerance)
        self.obj_load.loss_coefficients = np.array(
            [
                self.read_float_value("loss_total_sigma", self.obj_load.loss_coefficients[0]),
                self.read_float_value("loss_abs_corr", self.obj_load.loss_coefficients[1]),
                self.read_float_value("loss_fl_corr", self.obj_load.loss_coefficients[2]),
            ],
            dtype=float,
        )
        self.obj_load.update_params()
        self.obj_load.update_experimental_interpolants()

    def obj2table(self):
        """Compatibility wrapper for older table refresh call sites."""
        self.render_table_from_state()

    """
    def update_data(self):#nouse
        # Update the selected row in the table
        selected_rows = self.table_widget.selectionModel().selectedRows()
        for index in selected_rows:
            name = self.table_widget.item(index.row(), 0).text()
            value = float(self.table_widget.item(index.row(), 1).text())
            # You can edit the name and value here as needed
        self.update_spectrum()
        """

    def keyPressEvent(self, event):
        """Handle key press events
        Args:
            event (QKeyEvent): The key press event
        """
        if event.key() == Qt.Key.Key_F5:
            self.update_spectrum()


"""
class OutputWidget(QTextBrowser):#Not compatible with qrunner. 
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)  # Make the text browser read-only

    def write(self, text):
        # Append the text to the output panel
        self.insertPlainText(text)
        self.ensureCursorVisible()  # Scroll to the latest text
        """


def exception_hook(exctype, value, traceback):
    """
    Custom exception hook to handle uncaught exceptions.
    Display an error message box with the exception details.
    """
    msg = f"Unhandled exception: {exctype.__name__}\n{value}"
    QMessageBox.critical(None, "Unhandled Exception", msg)
    # Call default exception hook
    sys.__excepthook__(exctype, value, traceback)


def main():
    app = QApplication(sys.argv)
    # Set the custom exception hook
    sys.excepthook = exception_hook
    app.setWindowIcon(QIcon("ico.ico"))
    window = SpectrumApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

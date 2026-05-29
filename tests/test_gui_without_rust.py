import os
import subprocess
import sys
import textwrap


def run_python(script: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["QT_QPA_PLATFORM"] = "offscreen"
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def test_resram_disable_rust_forces_python_backend():
    result = run_python(
        "import os; "
        "os.environ['RESRAM_DISABLE_RUST'] = '1'; "
        "import resram_ng.resram_core as core; "
        "print(core.HAS_RUST)"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_resram_rust_namespace_package_does_not_enable_backend():
    result = run_python(
        "import resram_ng.resram_core as core; "
        "print(core.HAS_RUST)"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_gui_starts_when_resram_rust_cannot_be_imported():
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockResramRust(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "resram_rust":
                    raise ImportError("blocked resram_rust for test")
                return None

        sys.meta_path.insert(0, BlockResramRust())

        from PyQt6.QtWidgets import QApplication
        from resram_ng import resram_core
        from resram_ng.ResRamQt import SpectrumApp

        app = QApplication([])
        window = SpectrumApp()
        print(resram_core.HAS_RUST)
        print(window.windowTitle())
        window.close()
        app.quit()
        """
    )

    result = run_python(script)

    assert result.returncode == 0, result.stderr
    assert "False" in result.stdout
    assert "Raman Spectrum Analyzer" in result.stdout


def test_gui_table_rebuilds_when_loaded_folder_has_fewer_modes():
    script = textwrap.dedent(
        """
        import os

        from PyQt6.QtWidgets import QApplication
        from resram_ng.ResRamQt import SpectrumApp
        from resram_ng.resram_core import load_input

        app = QApplication([])
        window = SpectrumApp()
        window.obj_load = load_input(os.path.abspath("../pm546"))
        window.sendto_table()
        window.load_table()
        print(window.maxnfev)
        window.close()
        app.quit()
        """
    )

    result = run_python(script)

    assert result.returncode == 0, result.stderr
    assert "100" in result.stdout


def test_gui_table_contains_only_loaded_folder_delta_rows():
    script = textwrap.dedent(
        """
        import os

        from PyQt6.QtWidgets import QApplication
        from resram_ng.ResRamQt import SpectrumApp
        from resram_ng.resram_core import load_input

        app = QApplication([])
        window = SpectrumApp()
        window.obj_load = load_input(os.path.abspath("../pm546"))
        window.render_table_from_state()
        labels = [
            window.table_widget.item(row, 0).text()
            for row in range(window.table_widget.rowCount())
            if window.table_widget.item(row, 0) is not None
        ]
        delta_labels = [label for label in labels if label.startswith("delta@")]
        print("ROWS", window.table_widget.rowCount())
        print("DELTAS", len(delta_labels))
        print("MAXNFEV_LABEL", window.table_widget.item(window.row_for_key("maxnfev"), 0).text())
        print("MAXNFEV_VALUE", window.table_widget.item(window.row_for_key("maxnfev"), 1).text())
        window.close()
        app.quit()
        """
    )

    result = run_python(script)

    assert result.returncode == 0, result.stderr
    assert "ROWS 25" in result.stdout
    assert "DELTAS 11" in result.stdout
    assert "MAXNFEV_LABEL Fitting maxnfev" in result.stdout
    assert "MAXNFEV_VALUE 100" in result.stdout


def test_gui_table_rebuilds_cleanly_from_smaller_to_larger_folder():
    script = textwrap.dedent(
        """
        import os

        from PyQt6.QtWidgets import QApplication
        from resram_ng.ResRamQt import SpectrumApp
        from resram_ng.resram_core import get_default_example_dir, load_input

        app = QApplication([])
        window = SpectrumApp()
        window.obj_load = load_input(os.path.abspath("../pm546"))
        window.render_table_from_state()
        window.obj_load = load_input(get_default_example_dir())
        window.render_table_from_state()
        labels = [
            window.table_widget.item(row, 0).text()
            for row in range(window.table_widget.rowCount())
            if window.table_widget.item(row, 0) is not None
        ]
        print("ROWS", window.table_widget.rowCount())
        print("DELTAS", len([label for label in labels if label.startswith("delta@")]))
        print("MAXNFEV_VALUE", window.table_widget.item(window.row_for_key("maxnfev"), 1).text())
        window.close()
        app.quit()
        """
    )

    result = run_python(script)

    assert result.returncode == 0, result.stderr
    assert "ROWS 40" in result.stdout
    assert "DELTAS 26" in result.stdout
    assert "MAXNFEV_VALUE 100" in result.stdout


def test_gui_invalid_numeric_edit_does_not_crash_or_replace_state():
    script = textwrap.dedent(
        """
        from PyQt6.QtWidgets import QApplication, QTableWidgetItem
        from resram_ng.ResRamQt import SpectrumApp

        app = QApplication([])
        window = SpectrumApp()
        row = window.row_for_key("maxnfev")
        window.table_widget.setItem(row, 1, QTableWidgetItem("not-an-int"))
        window.load_table()
        print("MAXNFEV", window.maxnfev)
        print("CELL", window.table_widget.item(row, 1).text())
        window.close()
        app.quit()
        """
    )

    result = run_python(script)

    assert result.returncode == 0, result.stderr
    assert "MAXNFEV 100" in result.stdout
    assert "CELL 100" in result.stdout


def test_gui_table_edits_apply_to_model_by_key():
    script = textwrap.dedent(
        """
        import os

        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication, QTableWidgetItem
        from resram_ng.ResRamQt import SpectrumApp
        from resram_ng.resram_core import load_input

        app = QApplication([])
        window = SpectrumApp()
        window.obj_load = load_input(os.path.abspath("../pm546"))
        window.render_table_from_state()
        delta_row = window.row_for_key("delta:0")
        gamma_row = window.row_for_key("gamma")
        window.table_widget.setItem(delta_row, 1, QTableWidgetItem("0.1234"))
        window.table_widget.item(delta_row, 2).setCheckState(Qt.CheckState.Unchecked)
        window.table_widget.item(gamma_row, 3).setCheckState(Qt.CheckState.Unchecked)
        window.load_table()
        print("DELTA0", f"{window.obj_load.delta[0]:.4f}")
        print("PLOT0", int(window.plot_switch[0]))
        print("FIT_GAMMA", int(window.fit_switch[len(window.obj_load.delta)]))
        window.close()
        app.quit()
        """
    )

    result = run_python(script)

    assert result.returncode == 0, result.stderr
    assert "DELTA0 0.1234" in result.stdout
    assert "PLOT0 0" in result.stdout
    assert "FIT_GAMMA 0" in result.stdout


def test_raman_excitation_canvas_clears_old_folder_items_on_load():
    script = textwrap.dedent(
        """
        import os
        import numpy as np

        from PyQt6.QtWidgets import QApplication
        import resram_ng.ResRamQt as gui
        from resram_ng.resram_core import load_input

        gui.SpectrumApp.trigger_calculation = lambda self: None

        def fake_results(obj):
            return (
                np.ones(len(obj.convEL)),
                np.ones(len(obj.convEL)),
                np.ones((len(obj.wg), len(obj.convEL))),
                np.ones((len(obj.rshift), len(obj.rpumps))),
            )

        app = QApplication([])
        window = gui.SpectrumApp()
        window.update_plots(fake_results(window.obj_load))
        window.obj_load = load_input(os.path.abspath("../pm546"))
        window.reset_plot_items()
        window.reset_table_for_loaded_data()
        window.update_plots(fake_results(window.obj_load))

        expected_items = len(window.obj_load.wg) * (len(window.obj_load.rpumps) + 1)
        print("EXPECTED", expected_items)
        print("ACTUAL", len(window.canvas.listDataItems()))
        window.close()
        app.quit()
        """
    )

    result = run_python(script)

    assert result.returncode == 0, result.stderr
    assert "EXPECTED 33" in result.stdout
    assert "ACTUAL 33" in result.stdout


def test_experimental_abs_fl_raw_axes_do_not_follow_e00():
    script = textwrap.dedent(
        """
        import os
        import numpy as np

        from resram_ng.resram_core import load_input

        obj = load_input(os.path.abspath("../pm546"))
        abs_raw_x = obj.abs_exp_raw[:, 0].copy()
        fl_raw_x = obj.fl_exp_raw[:, 0].copy()
        old_conv_x = obj.convEL.copy()
        obj.E0 += 500
        obj.update_params()
        obj.update_experimental_interpolants()
        print("RAW_ABS_FIXED", np.array_equal(abs_raw_x, obj.abs_exp_raw[:, 0]))
        print("RAW_FL_FIXED", np.array_equal(fl_raw_x, obj.fl_exp_raw[:, 0]))
        print("CONVEL_SHIFTED", not np.array_equal(old_conv_x, obj.convEL))
        print("ABS_INTERP_ON_CONVEL", np.array_equal(obj.abs_exp[:, 0], obj.convEL))
        window.close() if False else None
        """
    )

    result = run_python(script)

    assert result.returncode == 0, result.stderr
    assert "RAW_ABS_FIXED True" in result.stdout
    assert "RAW_FL_FIXED True" in result.stdout
    assert "CONVEL_SHIFTED True" in result.stdout
    assert "ABS_INTERP_ON_CONVEL True" in result.stdout


def test_abs_fl_experimental_plot_x_data_stays_fixed_after_e00_change():
    script = textwrap.dedent(
        """
        import os
        import numpy as np

        from PyQt6.QtWidgets import QApplication
        import resram_ng.ResRamQt as gui
        from resram_ng.resram_core import load_input

        gui.SpectrumApp.trigger_calculation = lambda self: None

        def fake_results(obj):
            return (
                np.ones(len(obj.convEL)),
                np.ones(len(obj.convEL)),
                np.ones((len(obj.wg), len(obj.convEL))),
                np.ones((len(obj.rshift), len(obj.rpumps))),
            )

        app = QApplication([])
        window = gui.SpectrumApp()
        window.obj_load = load_input(os.path.abspath("../pm546"))
        window.reset_plot_items()
        window.reset_table_for_loaded_data()
        window.update_plots(fake_results(window.obj_load))
        original_exp_x = window.abs_exp_plot_item.xData.copy()
        original_calc_x = window.abs_plot_item.xData.copy()

        window.obj_load.E0 += 500
        window.obj_load.update_params()
        window.obj_load.update_experimental_interpolants()
        window.update_plots(fake_results(window.obj_load))

        print("EXP_X_FIXED", np.array_equal(original_exp_x, window.abs_exp_plot_item.xData))
        print("CALC_X_SHIFTED", not np.array_equal(original_calc_x, window.abs_plot_item.xData))
        window.close()
        app.quit()
        """
    )

    result = run_python(script)

    assert result.returncode == 0, result.stderr
    assert "EXP_X_FIXED True" in result.stdout
    assert "CALC_X_SHIFTED True" in result.stdout


def test_fl_experimental_plot_is_normalized_to_calculated_fl():
    script = textwrap.dedent(
        """
        import os
        import numpy as np

        from PyQt6.QtWidgets import QApplication
        import resram_ng.ResRamQt as gui
        from resram_ng.resram_core import load_input

        gui.SpectrumApp.trigger_calculation = lambda self: None

        def fake_results(obj):
            fl_cross = np.linspace(0.0, 42.0, len(obj.convEL))
            return (
                np.ones(len(obj.convEL)),
                fl_cross,
                np.ones((len(obj.wg), len(obj.convEL))),
                np.ones((len(obj.rshift), len(obj.rpumps))),
            )

        app = QApplication([])
        window = gui.SpectrumApp()
        window.obj_load = load_input(os.path.abspath("../pm546"))
        window.reset_plot_items()
        window.reset_table_for_loaded_data()
        window.update_plots(fake_results(window.obj_load))
        print("CALC_MAX", float(np.max(window.fl_plot_item.yData)))
        print("EXP_MAX", float(np.max(window.fl_exp_plot_item.yData)))
        window.close()
        app.quit()
        """
    )

    result = run_python(script)

    assert result.returncode == 0, result.stderr
    assert "CALC_MAX 42.0" in result.stdout
    assert "EXP_MAX 42.0" in result.stdout

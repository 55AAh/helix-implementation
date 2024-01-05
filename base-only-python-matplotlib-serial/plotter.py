import io
import json
import os.path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from tkinter import filedialog, messagebox
from typing import ClassVar, Tuple, List, Optional

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.widgets import Button, CheckButtons, AxesWidget, TextBox
from tqdm import tqdm


@dataclass
class DrawParams:
    line: bool = True

    t: bool = False
    xi: bool = False
    eta: bool = False

    n: bool = False
    beta: bool = False

    names: ClassVar[Tuple[str]] = ('t', 'ξ', 'η', 'n', 'β')

    def update(self, line: bool = True,
               t: bool = False, xi: bool = False, eta: bool = False,
               n: bool = False, beta: bool = False):
        self.line, self.t, self.xi, self.eta, self.n, self.beta = line, t, xi, eta, n, beta


class ISystem(ABC):
    @abstractmethod
    def plot(self, ax: Axes, current_state_num: int,
             elements_params: DrawParams, guess_params: DrawParams,
             plot_elements_segments, plot_basis_vectors_len: float, plot_basis_vectors_segments: int):
        """ Draws the frame """

    @abstractmethod
    def step(self):
        """ Advances system one step forward """

    @abstractmethod
    def is_adjusting_ratio(self) -> bool:
        """ Determines whether adjusting ratio """

    @abstractmethod
    def is_updating_task(self) -> bool:
        """ Determines whether updating task """

    @abstractmethod
    def is_task_boundary(self) -> bool:
        """ Determines whether generating new task """

    @abstractmethod
    def is_finished(self) -> bool:
        """ Determines whether finished """

    @abstractmethod
    def get_next(self) -> Optional['ISystem']:
        """ Retrieves the next state of the system """

    @abstractmethod
    def get_state_text(self) -> str:
        """ Represents the current state of a system in a text form """

    @abstractmethod
    def show_stats(self):
        """ Shows statistics """

    @abstractmethod
    def get_shared_data(self) -> dict:
        """ Retrieves shared data """

    @abstractmethod
    def set_shared_data(self, d: dict):
        """ Sets shared data """

    @abstractmethod
    def serialize(self) -> dict:
        """ Serializes current state of the system """

    @abstractmethod
    def deserialize(self, s: dict) -> 'ISystem':
        """ Deserializes current state of the system """


class Plotter:
    def __init__(self, system: ISystem, interactive: bool = True,
                 pause: float = 0, skip_plots: int = 0,
                 plot_elements_segments: int = 30,
                 plot_basis_vectors_len: float = 1, plot_basis_vectors_segments: int = 1,
                 next_cnt: int = 1, keep_lims: bool = False, lims=None,
                 major: dict = None, minor: dict = None):

        self.states = [system]
        self.current_state_num = 0

        self.interactive = interactive
        if self.interactive:
            self.fig: Figure = plt.figure()
            self.ax: Axes = self.fig.add_subplot(projection='3d')
            self.element_params = DrawParams()
            self.guess_params = DrawParams()
            self.guess_params.line = False
            self._widgets: List[AxesWidget] = []
            self.state_text: Optional[Text] = None
            self._plots_skipped = 0
            self._next_cnt = next_cnt
            self._keep_lims = keep_lims
            self._pause = pause
            self._skip_plots = skip_plots
            self._plot_elements_segments = plot_elements_segments
            self._plot_basis_vectors_len = plot_basis_vectors_len
            self._plot_basis_vectors_segments = plot_basis_vectors_segments
            self._lims = lims

        self._major = major

        if minor is None:
            minor = {}
        self._minor = minor

    @property
    def _current_state(self) -> ISystem:
        return self.states[self.current_state_num]

    def _plot(self, can_skip=False):
        if can_skip and self._plots_skipped < self._skip_plots:
            self._plots_skipped += 1
            return
        self._plots_skipped = 0

        if self._lims is not None:
            x, y, z = self._lims
        else:
            x, y, z = self.ax.get_xlim(), self.ax.get_ylim(), self.ax.get_zlim()
        self.ax.clear()

        # self.ax.set_autoscale_on(False)
        self._current_state.plot(self.ax, self.current_state_num,
                                 self.element_params, self.guess_params,
                                 self._plot_elements_segments,
                                 self._plot_basis_vectors_len, self._plot_basis_vectors_segments)

        if self._keep_lims:
            self.ax.set_xlim(*x)
            self.ax.set_ylim(*y)
            self.ax.set_zlim(*z)

        if self.state_text is not None:
            self.state_text.set_text(self._current_state.get_state_text())

        plt.draw()
        if self._pause:
            plt.pause(self._pause)

    def _add_widget(self, widget):
        self._widgets.append(widget)
        return widget

    def _redrawing_callback(self, f: callable, *args, **kwargs):
        def cb(*_args, **_kwargs):
            f(*args, **kwargs)
            self._plot()

        return cb

    def run_interactive(self):
        if not self.interactive:
            return self.run()

        self.fig.subplots_adjust(bottom=0.32)

        self._plot()

        # Draw params checkboxes

        for i, (name, draw_params) in enumerate(zip(['Elements', 'Guess'],
                                                    [self.element_params, self.guess_params])):
            cb = self._add_widget(CheckButtons(self.fig.add_axes([0.1 + 0.17 * i, 0.07, 0.15, 0.2]),
                                               labels=[name] + list(draw_params.names),
                                               actives=[draw_params.line] + [False for _ in draw_params.names]))
            cb.on_clicked(self._redrawing_callback(lambda _dp, _cb: _dp.update(*_cb.get_status()), draw_params, cb))

        # State text
        self.state_text = self.fig.text(0.43, 0.085, self._current_state.get_state_text())

        # Compute next & revert

        def _make_step():
            if not cb_compute.get_status()[0]:
                if self.current_state_num < len(self.states) - 1:
                    self.current_state_num += 1
            else:
                new = self._current_state.step()
                if new is not None:
                    del self.states[self.current_state_num + 1:]
                    self.states.append(new)
                    self.current_state_num += 1
            if not cb_skip_steps.get_status()[0]:
                self._plot(can_skip=True)

        def _step():
            for _ in range(int(next_steps_count_tb.text)):
                _make_step()

        b = self._add_widget(Button(self.fig.add_axes([0.65, 0.225, 0.15, 0.045]), 'Step'))
        b.on_clicked(self._redrawing_callback(_step))

        def _adjust_ratio():
            for _ in range(int(next_steps_count_tb.text)):
                while True:
                    _make_step()
                    if self._current_state.is_adjusting_ratio():
                        break

        b = self._add_widget(Button(self.fig.add_axes([0.65, 0.17, 0.15, 0.045]), 'Adjust'))
        b.on_clicked(self._redrawing_callback(_adjust_ratio))

        def _update_task():
            for _ in range(int(next_steps_count_tb.text)):
                while True:
                    _make_step()
                    if self._current_state.is_updating_task():
                        break

        b = self._add_widget(Button(self.fig.add_axes([0.65, 0.115, 0.15, 0.045]), 'Update'))
        b.on_clicked(self._redrawing_callback(_update_task))

        b = self._add_widget(Button(self.fig.add_axes([0.27, 0.015, 0.15, 0.045]), 'Show stats'))
        b.on_clicked(lambda _: self._current_state.show_stats())

        ax = self.fig.add_axes([0.65, 0.017, 0.1, 0.04])
        ax.axis('off')
        cb_skip_steps = self._add_widget(CheckButtons(ax, labels=['Skip s'], actives=[True]))

        ax = self.fig.add_axes([0.752, 0.017, 0.1, 0.04])
        ax.axis('off')
        cb_skip_apply_steps = self._add_widget(CheckButtons(ax, labels=['Skip /'], actives=[False]))

        ax = self.fig.add_axes([0.85, 0.017, 0.1, 0.04])
        ax.axis('off')
        cb_skip_percents_steps = self._add_widget(CheckButtons(ax, labels=['Skip %'], actives=[False]))

        def _update_next_steps_count(_cb):
            try:
                val = int(_cb.text)
                if val > 0:
                    if str(val) != _cb.text:
                        _cb.set_val(str(val))
                    return
            except ValueError:
                pass
            _cb.set_val(str(1))

        next_steps_count_tb = self._add_widget(TextBox(self.fig.add_axes([0.755, 0.06, 0.045, 0.045]),
                                                       label=' ', initial=str(1)))
        next_steps_count_tb.on_submit(self._redrawing_callback(_update_next_steps_count, next_steps_count_tb))

        next_steps_task_count_tb = self._add_widget(TextBox(self.fig.add_axes([0.915, 0.17, 0.045, 0.045]),
                                                            label=' ', initial=str(self._next_cnt)))
        next_steps_task_count_tb.on_submit(self._redrawing_callback(_update_next_steps_count, next_steps_task_count_tb))

        def _back():
            for _ in range(int(next_steps_count_tb.text)):
                if self.current_state_num > 0:
                    self.current_state_num -= 1
                    if not cb_skip_steps.get_status()[0]:
                        self._plot(can_skip=True)
                else:
                    break

        b = self._add_widget(Button(self.fig.add_axes([0.65, 0.06, 0.1, 0.045]), 'Back'))
        b.on_clicked(self._redrawing_callback(_back))

        def _back_task():
            for _ in range(int(next_steps_task_count_tb.text)):
                while self.current_state_num > 0:
                    self.current_state_num -= 1
                    if self._current_state.is_task_boundary():
                        break
                    if not cb_skip_steps.get_status()[0]:
                        self._plot(can_skip=True)
                if not cb_skip_percents_steps.get_status()[0]:
                    self._plot(can_skip=True)

        b = self._add_widget(Button(self.fig.add_axes([0.81, 0.225, 0.15, 0.045]), 'Revert p'))
        b.on_clicked(self._redrawing_callback(_back_task))

        def _next_t():
            for _ in range(int(next_steps_task_count_tb.text)):
                while True:
                    while True:
                        _make_step()
                        if self._current_state.is_updating_task():
                            break
                    if self._current_state.is_task_boundary():
                        break
                    if not cb_skip_apply_steps.get_status()[0]:
                        self._plot(can_skip=True)
                if not cb_skip_percents_steps.get_status()[0]:
                    self._plot(can_skip=True)

        b = self._add_widget(Button(self.fig.add_axes([0.81, 0.17, 0.1, 0.045]), 'Next p'))
        b.on_clicked(self._redrawing_callback(_next_t))

        ax = self.fig.add_axes([0.815, 0.11, 0.1, 0.04])
        ax.axis('off')
        cb_compute = self._add_widget(CheckButtons(ax, labels=['Compute'], actives=[True]))

        ax = self.fig.add_axes([0.1, 0.015, 0.1, 0.04])
        ax.axis('off')
        cb = self._add_widget(CheckButtons(ax, labels=['Keep lims'], actives=[self._keep_lims]))

        def _keep_lims(_cb):
            self._keep_lims = _cb.get_status()[0]

        cb.on_clicked(self._redrawing_callback(_keep_lims, cb))

        ax = self.fig.add_axes([0.02, 0.015, 0.1, 0.04])
        ax.axis('off')
        cb_save_all = self._add_widget(CheckButtons(ax, labels=['All'], actives=[True]))

        b_save = self._add_widget(Button(self.fig.add_axes([0.015, 0.18, 0.07, 0.09]), 'Save'))

        fd_options = dict(defaultextension='json',
                          filetypes=[('Geometry file', '.json')],
                          initialdir='out')

        def _save():
            file_path = filedialog.asksaveasfilename(**fd_options, title='Save geometry to file')
            if not file_path:
                print('Saving canceled')
                return

            if cb_save_all.get_status()[0]:
                states = self.states
            else:
                states = [self._current_state]
            serialized_states = []
            for state in tqdm(states, desc='Saving'):
                serialized_states.append(state.serialize())

            shared_data = self._current_state.get_shared_data()
            d = {
                'major': self._major,
                'minor': self._minor,
                'saved_states_count': len(states),
                'states': serialized_states,
                'shared_data': shared_data,
            }

            with open(file_path, 'w') as f:
                pbar = tqdm(desc='Writing file', unit="B", unit_scale=True)
                _write = f.write

                def write_wrapper(chunk):
                    _write(chunk)
                    pbar.update(len(chunk))

                f.write = write_wrapper

                json.dump(d, f, indent=4)
            pbar.close()

            states_plural_str = '(s)' if len(states) > 1 else ''
            print(f'Saved state{states_plural_str} to file "{file_path}"')

        b_save.on_clicked(self._redrawing_callback(_save))

        b_load = self._add_widget(Button(self.fig.add_axes([0.015, 0.07, 0.07, 0.09]), 'Load'))

        def _load():
            file_path = filedialog.askopenfilename(**fd_options, title='Load geometry from file')
            if not file_path:
                print('Loading canceled')
                return

            _total = os.path.getsize(file_path)
            pbar = tqdm(desc='Reading file', unit="B", unit_scale=True, total=_total)
            with open(file_path, 'rb') as f:
                _read = f.read

                def read_wrapper():
                    buffer = io.BytesIO()
                    while True:
                        chunk = _read(1024 * 1000)
                        pbar.update(len(chunk))
                        buffer.write(chunk)
                        if not chunk:
                            break
                    data = buffer.getvalue()
                    buffer.close()
                    return data

                f.read = read_wrapper
                d = json.load(f)
            pbar.close()

            serialized_states = d['states']
            assert len(serialized_states) > 0
            assert d['saved_states_count'] == len(serialized_states)
            if not cb_save_all.get_status()[0]:
                serialized_states = serialized_states[-1:]

            old_major_s = json.dumps(self._major)
            new_major_s = json.dumps(d['major'])
            if new_major_s != old_major_s:
                if not messagebox.askokcancel('Load geometry from file',
                                              f'Saved geometry has the following MAJOR:\n'
                                              f'{new_major_s}\n'
                                              f'while the current geometry has:\n'
                                              f'{old_major_s}\n'
                                              f'Continue loading?'):
                    print('Loading canceled')
                    return
            old_minor_s = json.dumps(self._minor)
            new_minor_s = json.dumps(d['minor'])
            if new_minor_s != old_minor_s:
                if not messagebox.askokcancel('Load geometry from file',
                                              f'Saved geometry has the following MINOR:\n'
                                              f'{new_minor_s}\n'
                                              f'while the current geometry has:\n'
                                              f'{old_minor_s}\n'
                                              f'Continue loading?'):
                    print('Loading canceled')
                    return

            shared_data = d['shared_data']
            states = []
            for ser_state in tqdm(serialized_states, desc='Loading'):
                state = self._current_state.deserialize(ser_state)
                state.set_shared_data(shared_data)
                states.append(state)
            self.states = states
            self.current_state_num = len(self.states) - 1

            states_plural_str = '(s)' if len(states) > 1 else ''
            print(f'Loaded state{states_plural_str} from file "{file_path}"')

        b_load.on_clicked(self._redrawing_callback(_load))

        def _skip_both():
            if not cb_skip_steps.get_status()[0]:
                cb_skip_steps.set_active(0)
            if not cb_skip_percents_steps.get_status()[0]:
                cb_skip_percents_steps.set_active(0)

        b = self._add_widget(Button(self.fig.add_axes([0.81, 0.06, 0.1, 0.045]), 'Skip'))
        b.on_clicked(lambda _: _skip_both())

        def _update_skip_plots_count(_cb):
            try:
                val = int(_cb.text)
                if val >= 0:
                    if str(val) != _cb.text:
                        _cb.set_val(str(val))
                    self._skip_plots = val
                    return
            except ValueError:
                pass
            _cb.set_val(str(0))

        skip_plots_tb = self._add_widget(TextBox(self.fig.add_axes([0.915, 0.06, 0.045, 0.045]),
                                                 label=' ', initial=str(self._skip_plots)))
        skip_plots_tb.on_submit(self._redrawing_callback(_update_skip_plots_count, skip_plots_tb))

        _make_step()
        self._plot()
        plt.show(block=True)

    def run(self):
        if self.interactive:
            self.run_interactive()
        else:
            while not self._current_state.is_finished():
                new = self._current_state.step()
                if new is not None:
                    self.states.append(new)
                    self.current_state_num += 1

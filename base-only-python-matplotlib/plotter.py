import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Tuple, List, Optional
from typing_extensions import Self
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import Button, CheckButtons, AxesWidget, TextBox
from tkinter import filedialog, messagebox


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
             elements_params: DrawParams, guess_params: DrawParams, final_guess_params: DrawParams,
             plot_elements_segments, plot_basis_vectors_len: float, plot_basis_vectors_segments: int):
        """ Draws the frame """

    @abstractmethod
    def apply_task(self):
        """ Applies task """

    @abstractmethod
    def join_guesses(self):
        """ Joins guesses """

    @abstractmethod
    def validate(self) -> bool:
        """ Validates guesses """

    @abstractmethod
    def get_next(self) -> Optional[Self]:
        """ Retrieves the next state of the system """

    @abstractmethod
    def serialize(self) -> Optional[dict]:
        """ Serializes current state of the system """

    @abstractmethod
    def deserialize(self, s: dict):
        """ Deserializes current state of the system """


i = [0]

class Plotter:
    def __init__(self, system: ISystem, pause: float = 0, skip_plots: int = 0,
                 plot_elements_segments: int = 30,
                 plot_basis_vectors_len: float = 1, plot_basis_vectors_segments: int = 1,
                 next_cnt: int = 1, keep_lims: bool = False, lims=None,
                 major: dict = None, minor: dict = None, state: dict = None):
        self.fig: Figure = plt.figure()
        self.ax: Axes = self.fig.add_subplot(projection='3d')

        self.states = [system]
        self.current_state_num = 0
        self.element_params = DrawParams()
        self.guess_params = DrawParams()
        self.final_guess_params = DrawParams()
        self.guess_params.line = self.final_guess_params.line = False
        self._widgets: List[AxesWidget] = []
        self._plots_skipped = 0
        self._keep_lims = keep_lims
        self._pause = pause
        self._skip_plots = skip_plots
        self._plot_elements_segments = plot_elements_segments
        self._plot_basis_vectors_len = plot_basis_vectors_len
        self._plot_basis_vectors_segments = plot_basis_vectors_segments
        self._next_cnt = next_cnt
        self._lims = lims

        self._major = major

        if minor is None:
            minor = {}
        self._minor = minor

        if state is None:
            state = {}

        self._state = state

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
                                 self.element_params, self.guess_params, self.final_guess_params,
                                 self._plot_elements_segments,
                                 self._plot_basis_vectors_len, self._plot_basis_vectors_segments)

        if self._keep_lims:
            self.ax.set_xlim(*x)
            self.ax.set_ylim(*y)
            self.ax.set_zlim(*z)

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

    def show_interactive(self):
        self.fig.subplots_adjust(bottom=0.3)

        self._plot()

        # Draw params checkboxes

        for i, (name, draw_params) in enumerate(zip(['Elements', 'Guess', 'Final guess'],
                                                    [self.element_params, self.guess_params, self.final_guess_params])):
            cb = self._add_widget(CheckButtons(self.fig.add_axes([0.1 + 0.2 * i, 0.05, 0.18, 0.2]),
                                               labels=[name] + list(draw_params.names),
                                               actives=[draw_params.line] + [False for _ in draw_params.names]))
            cb.on_clicked(self._redrawing_callback(lambda _dp, _cb: _dp.update(*_cb.get_status()), draw_params, cb))

        # Compute next & revert

        b = self._add_widget(Button(self.fig.add_axes([0.7, 0.205, 0.1, 0.045]), 'Guess'))
        b.on_clicked(self._redrawing_callback(lambda: self._current_state.apply_task()))

        b = self._add_widget(Button(self.fig.add_axes([0.7, 0.15, 0.1, 0.045]), 'Join'))
        b.on_clicked(self._redrawing_callback(lambda: [self._current_state.apply_task(),
                                                       self._current_state.join_guesses()]))

        b = self._add_widget(Button(self.fig.add_axes([0.7, 0.095, 0.1, 0.045]), 'Validate'))
        b.on_clicked(self._redrawing_callback(lambda: [self._current_state.apply_task(),
                                                       self._current_state.join_guesses(),
                                                       self._current_state.validate()]))

        ax = self.fig.add_axes([0.7, 0.05, 0.1, 0.04])
        ax.axis('off')
        cb_auto_guess_join = self._add_widget(CheckButtons(ax, labels=['Auto'], actives=[True]))

        def _update_next_steps_count(_cb):
            try:
                val = int(_cb.text)
                if val > 0:
                    if str(val) != _cb.text:
                        _cb.set_val(str(val))
                    return
            except ValueError:
                pass
            _cb.set_val('1')

        next_steps_count_tb = self._add_widget(TextBox(self.fig.add_axes([0.915, 0.15, 0.045, 0.045]),
                                                       label=' ', initial=str(self._next_cnt)))
        next_steps_count_tb.on_submit(self._redrawing_callback(_update_next_steps_count, next_steps_count_tb))

        def _revert():
            for _ in range(int(next_steps_count_tb.text)):
                if self.current_state_num > 0:
                    self.current_state_num -= 1

        b = self._add_widget(Button(self.fig.add_axes([0.81, 0.205, 0.15, 0.045]), 'Revert'))
        b.on_clicked(self._redrawing_callback(_revert))

        def _compute_next():
            for _ in range(int(next_steps_count_tb.text)):
                if cb_compute.get_status()[0]:
                    self._current_state.apply_task()
                    self._current_state.join_guesses()
                    if not self._current_state.validate():
                        continue
                    new_state = self._current_state.get_next()
                    if new_state is None:
                        break
                    self.current_state_num += 1
                    self.states = self.states[:self.current_state_num]
                    self.states.append(new_state)
                    if cb_auto_guess_join.get_status()[0]:
                        self._current_state.apply_task()
                        self._current_state.join_guesses()
                        self._current_state.validate()
                else:
                    if self.current_state_num >= len(self.states) - 1:
                        break
                    self.current_state_num += 1
                if self._pause:
                    self._plot(can_skip=True)

        b = self._add_widget(Button(self.fig.add_axes([0.81, 0.15, 0.1, 0.045]), 'Next'))
        b.on_clicked(self._redrawing_callback(_compute_next))

        ax = self.fig.add_axes([0.81, 0.09, 0.15, 0.04])
        ax.axis('off')
        cb_compute = self._add_widget(CheckButtons(ax, labels=['Compute'], actives=[True]))

        ax = self.fig.add_axes([0.81, 0.05, 0.15, 0.04])
        ax.axis('off')
        cb = self._add_widget(CheckButtons(ax, labels=['Keep lims'], actives=[self._keep_lims]))

        def _keep_lims(_cb):
            self._keep_lims = _cb.get_status()[0]

        cb.on_clicked(self._redrawing_callback(_keep_lims, cb))

        b_save = self._add_widget(Button(self.fig.add_axes([0.015, 0.16, 0.07, 0.09]), 'Save'))

        def _save():
            file = filedialog.asksaveasfile(**fd_options, title='Save geometry to file')
            if file is None:
                return
            gd = self._current_state.serialize()
            d = {
                'major': self._major,
                'minor': self._minor,
                'state': self._state,
                'geometry': gd,
            }
            with file as f:
                json.dump(d, f, indent=4)
            print('Saved geometry!')

        b_save.on_clicked(self._redrawing_callback(_save))

        b_load = self._add_widget(Button(self.fig.add_axes([0.015, 0.05, 0.07, 0.09]), 'Load'))

        fd_options = dict(defaultextension='json',
                          filetypes=[('Geometry file', '.json')],
                          initialdir='geom')

        def _load():
            file = filedialog.askopenfile(**fd_options, title='Load geometry from file')
            if file is None:
                return
            with file as f:
                d = json.load(f)
            same = True
            old_major_s = json.dumps(self._major)
            new_major_s = json.dumps(d['major'])
            if new_major_s != old_major_s:
                same = False
                if not messagebox.askokcancel('Load geometry from file',
                                              f'Saved geometry has the following MAJOR:\n'
                                              f'{new_major_s}\n'
                                              f'while the current geometry has:\n'
                                              f'{old_major_s}\n'
                                              f'Continue loading?'):
                    return
            old_minor_s = json.dumps(self._minor)
            new_minor_s = json.dumps(d['minor'])
            if new_minor_s != old_minor_s:
                same = False
                if not messagebox.askokcancel('Load geometry from file',
                                              f'Saved geometry has the following MINOR:\n'
                                              f'{new_minor_s}\n'
                                              f'while the current geometry has:\n'
                                              f'{old_minor_s}\n'
                                              f'Continue loading?'):
                    return
            old_state_s = json.dumps(self._state)
            new_state_s = json.dumps(d['state'])
            if new_state_s != old_state_s:
                if messagebox.askyesno('Load geometry from file',
                                       f'Saved geometry has the following STATE:\n'
                                       f'{new_state_s}\n'
                                       f'while the current geometry has:\n'
                                       f'{old_state_s}\n'
                                       f'Replace current state with saved?'):
                    if not same:
                        ans = messagebox.askyesnocancel('Load geometry from file',
                                                        f'Saved geometry has the following STATE:\n'
                                                        f'{new_state_s}\n'
                                                        f'while the current geometry has:\n'
                                                        f'{old_state_s}\n'
                                                        f'Merge current state with saved?')
                        if ans is None:
                            return
                        if not ans:
                            self._state.clear()
                    self._state.update(d['state'])
            self._current_state.deserialize(d['geometry'])
            print('Loaded geometry!')

        b_load.on_clicked(self._redrawing_callback(_load))

        plt.show(block=True)

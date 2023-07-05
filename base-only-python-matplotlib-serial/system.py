from dataclasses import dataclass, asdict
from enum import auto, Enum
from typing import List, Generator, Optional

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.widgets import Button
from numpy import array as point, log10, arccos, clip
from numpy import array as vector
from numpy import linspace, cross, array, dot, pi, cos
from numpy.linalg import norm

from element import Element
from plotter import ISystem, DrawParams


def dprint(*_args, **_kwargs):
    # print(*_args, **_kwargs)
    pass


@dataclass
class Params:
    arm_position: float = 0.5
    percent_steps: int = 1
    criteria_goal: float = 1e-12
    criteria_final_goal: float = 1e-12
    dTh_diff_angle_max_limit: float = 1.0
    dTh_diff_angle_mean_limit: float = 1.0
    T_diff_max_limit: float = 1.0
    T_diff_mean_limit: float = 1.0
    last_point_diff_prop_limit: float = 0.1

    def serialize(self) -> dict:
        return asdict(self)

    @staticmethod
    def deserialize(d: dict):
        return Params(**d)


@dataclass
class Task:
    m0: vector | List[vector]
    f: vector
    fp: point

    def serialize(self) -> dict:
        return {
            'm0': array(self.m0).tolist(),
            'f': self.f.tolist(),
            'fp': self.fp.tolist(),
        }

    @staticmethod
    def deserialize(d: dict):
        m0 = array(d['m0'])
        if len(m0.shape) > 1:
            m0 = list(m0)
        return Task(m0, vector(d['f']), point(d['fp']))


class Stats:
    def __init__(self, shared_data: dict):
        self._shared_data = shared_data

    def reset(self):
        stats = dict()
        stats['g_cosine'] = []
        stats['g_C_cos'] = []
        stats['g_mu'] = []
        stats['g_log10_ratio'] = []
        stats['g_log10_criteria'] = []
        stats['g_window_start'] = 0
        self._shared_data['stats'] = stats

    def start_window(self):
        stats = self._shared_data['stats']
        stats['g_window_start'] = len(stats['g_cosine'])

    def add(self, cosine, C_cos, mu, log10_ratio, log10_criteria):
        stats = self._shared_data['stats']
        stats['g_cosine'].append(cosine)
        stats['g_C_cos'].append(C_cos)
        stats['g_mu'].append(mu)
        stats['g_log10_ratio'].append(log10_ratio)
        stats['g_log10_criteria'].append(log10_criteria)

    def show(self):
        stats = self._shared_data['stats']

        fig1, ax = plt.subplots()
        ax.set_title('All')
        x = range(len(stats['g_cosine']))
        ax.plot(x, stats['g_cosine'], '.-', label='cosine')
        ax.plot(x, stats['g_C_cos'], '.-', label='C_cos')
        ax.plot(x, stats['g_mu'], '.-', label='mu')
        ax.plot(x, stats['g_log10_ratio'], '.-', label='log10_ratio')
        ax.plot(x, stats['g_log10_criteria'], '.-', label='log10_criteria')
        plt.legend()
        plt.pause(0.1)

        fig2, ax = plt.subplots()
        ax.set_title('Last')
        _window_start = stats['g_window_start']
        x = range(_window_start, len(stats['g_cosine']))
        ax.plot(x, stats['g_cosine'][_window_start:], '.-', label='cosine')
        ax.plot(x, stats['g_C_cos'][_window_start:], '.-', label='C_cos')
        ax.plot(x, stats['g_mu'][_window_start:], '.-', label='mu')
        ax.plot(x, stats['g_log10_ratio'][_window_start:], '.-', label='log10_ratio')
        ax.plot(x, stats['g_log10_criteria'][_window_start:], '.-', label='log10_criteria')
        plt.legend()

        _stop = [False]

        def stop(*_args):
            _stop[0] = True

        b1 = Button(fig1.add_axes([0.89, 0.95, 0.1, 0.045]), 'CLOSE')
        b1.on_clicked(stop)

        b2 = Button(fig2.add_axes([0.89, 0.95, 0.1, 0.045]), 'CLOSE')
        b2.on_clicked(stop)

        while not _stop[0]:
            plt.pause(1)

        plt.pause(0.1)
        plt.close(fig1)
        plt.close(fig2)
        plt.pause(0.1)


class States(Enum):
    @staticmethod
    def _generate_next_value_(name: str, *_args):
        return name

    Initial = auto()
    IncrementingLoad = auto()
    UpdatingTask = auto()
    SolvingTask = auto()
    Applying = auto()
    AdjustingRatio = auto()
    SatisfyingCriteria = auto()
    SolvedTask = auto()
    Finished = auto()


@dataclass
class System(ISystem):
    state: States
    params: Params
    shared_data: dict

    current_task: Optional[Task]
    applying_to_num: Optional[int]
    ratio: Optional[float]
    max_criteria: Optional[float]
    last_point_diff_prev: Optional[vector]
    iteration: Optional[int]
    percent: Optional[float]

    builtin_moments: List[vector]
    applied_moments: Optional[List[vector]]

    elements: List[Element]
    guess: Optional[List[Element]]

    def __init__(self, task_gen: Generator[Task, 'System', None],
                 elements: List[Element], params: Params, shared_data: dict = None,
                 builtin_moments: Optional[List[vector]] = None):
        """ Створює систему елементів

         :param elements: набір елементів
        """

        self.task_gen = task_gen
        self.state = States.Initial
        self.params = params
        if shared_data is None:
            shared_data = dict()
        self.shared_data = shared_data

        self.current_task = None
        self.applying_to_num = None
        self.ratio = None
        self.max_criteria = None
        self.last_point_diff_prev = None
        self.iteration = None
        self.percent = None

        if builtin_moments is None:
            builtin_moments = [vector([0, 0, 0]) for _ in elements]
        self.builtin_moments = builtin_moments
        self.applied_moments = None

        self.elements = elements
        self.guess = None

    def clone_initial(self) -> 'System':
        clone = System(self.task_gen,
                       self.elements,
                       params=self.params,
                       shared_data=self.shared_data,
                       builtin_moments=self.builtin_moments)
        return clone

    def get_last_point(self, guess: bool = False) -> point:
        if guess:
            return self.guess[-1].point(self.guess[-1].s)
        else:
            return self.elements[-1].point(self.elements[-1].s)

    def plot(self, ax: Axes, current_state_num: int,
             elements_params: DrawParams, guess_params: DrawParams,
             plot_elements_segments: int, plot_basis_vectors_len: float, plot_basis_vectors_segments: int):

        plots = [(self.elements, '-', 'black', 'Elements', elements_params)]
        if self.guess is not None:
            plots.append((self.guess, ':', 'tab:orange', 'Guess', guess_params))

        plt_objs = []

        class ECL:
            def __init__(self, label, color=None):
                self._color = color
                self._label = label

            def color(self):
                return self._color

            def label(self):
                _l = self._label
                self._label = None
                return _l

            def add(self, obj):
                if self._color is None:
                    self._color = obj[0].get_color()
                return obj

        for elements, line_style, def_line_color, name, params in plots:
            if params.line:
                ecl_el = ECL(name, def_line_color)
                for e in elements:
                    p = [e.point(s) for s in linspace(0, e.s, plot_elements_segments)]
                    plt_objs.append(ecl_el.add(ax.plot([p[0] for p in p], [p[1] for p in p], [p[2] for p in p],
                                                       color=ecl_el.color(), label=None))[0])
                    plt_objs.append(ecl_el.add(ax.scatter([p[0][0]], [p[0][1]], [p[0][2]], s=3,
                                                          color=ecl_el.color(), label=None)))
                    ax.plot([], [], '-.', color=ecl_el.color(), label=ecl_el.label())

        for _ in range(3):
            # noinspection PyProtectedMember
            next(ax._get_lines.prop_cycler)

        for elements, line_style, _def_line_color, name, params in plots:
            if params.t or params.xi or params.eta or params.n or params.beta:
                ecl_t = ECL(name + '.t') if params.t else None
                ecl_xi = ECL(name + '.ξ') if params.xi else None
                ecl_eta = ECL(name + '.η') if params.eta else None
                ecl_n = ECL(name + '.n') if params.n else None
                ecl_beta = ECL(name + '.β') if params.beta else None

                for e in elements:
                    for s in linspace(0, e.s, plot_basis_vectors_segments):
                        p = e.point(s)
                        mat, nat = e.bases(s)

                        for vec, ecl in [(mat[0], ecl_t),
                                         (mat[1], ecl_xi),
                                         (mat[2], ecl_eta),
                                         (nat[1], ecl_n),
                                         (nat[2], ecl_beta)]:
                            if ecl is None:
                                continue
                            m_vec = vec * plot_basis_vectors_len * e.s / plot_basis_vectors_segments
                            plt_objs.append(ecl.add(ax.plot([p[0], p[0] + m_vec[0]],
                                                            [p[1], p[1] + m_vec[1]],
                                                            [p[2], p[2] + m_vec[2]],
                                                            color=ecl.color(),
                                                            label=ecl.label())))

        if ax.lines:
            ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Геометрія номер {current_state_num}')
        ax.axis('equal')

        return plt_objs

    def step(self):
        if self.state == States.Initial:
            next(self.task_gen)
            self.percent = 0
            Stats(self.shared_data).reset()
            self.state = States.IncrementingLoad
            return

        if self.state == States.IncrementingLoad:
            new_state = self.clone_initial()
            new_state.percent = self.percent + 100 / self.params.percent_steps
            new_state.iteration = 0
            new_state.ratio = 1.0
            Stats(self.shared_data).start_window()
            new_state.state = States.UpdatingTask
            return new_state

        if self.state == States.UpdatingTask:
            try:
                self.current_task = self.task_gen.send(self)
            except StopIteration:
                print('StopIteration')
                self.state = States.Finished
                return
            self.state = States.SolvingTask
            return

        if self.state == States.SolvingTask:
            self.guess = [e.copy() for e in self.elements]
            self.applied_moments = [None for _e in self.elements]
            self.applying_to_num = 0
            self.state = States.Applying
            return

        if self.state == States.Applying:
            if self.applying_to_num > 0:
                e_p = self.guess[self.applying_to_num - 1]
                e_n = self.guess[self.applying_to_num]
                e_n.join_after(e_p)

            i = self.applying_to_num
            m0, force, force_point = self.current_task.m0, self.current_task.f, self.current_task.fp
            if isinstance(m0, list):
                m0 = m0[i]
            e = self.guess[i]

            apply_point = e.point(e.s * self.params.arm_position)
            arm = force_point - apply_point
            force_moment = cross(arm, force)
            new_moment = m0 + force_moment

            builtin_moment = self.builtin_moments[i]
            applied_moment = builtin_moment + (new_moment - builtin_moment) * self.ratio
            g = e.apply_moment(applied_moment)

            self.guess[i] = g
            self.applied_moments[i] = applied_moment

            self.applying_to_num += 1
            if self.applying_to_num == len(self.elements):
                self.applying_to_num = None
                self.state = States.AdjustingRatio
            return

        if self.state == States.AdjustingRatio:
            cos_angle = lambda a, b: clip(dot(a, b) / norm(a) / norm(b), -1, 1) if norm(a) * norm(b) > 0 else 1.0
            dTh_diff_angles = [arccos(cos_angle(g.dTh, e.dTh)) for e, g in zip(self.elements, self.guess)]
            dTh_diff_angle_max = max(dTh_diff_angles)
            dTh_diff_angle_mean = sum(dTh_diff_angles) / len(self.elements)

            T_diffs = [g.T - e.T for e, g in zip(self.elements, self.guess)]
            T_diff_max = max(T_diffs)
            T_diff_mean = sum(T_diffs) / len(self.elements)

            last_point_e = self.get_last_point()
            last_point_g = self.get_last_point(guess=True)
            last_point_diff = last_point_e - last_point_g
            total_length = sum([e.s for e in self.elements])
            last_point_diff_prop = norm(last_point_diff) / total_length

            criteria = [
                dTh_diff_angle_max / self.params.dTh_diff_angle_max_limit,
                dTh_diff_angle_mean / self.params.dTh_diff_angle_mean_limit,
                T_diff_max / self.params.T_diff_max_limit,
                T_diff_mean / self.params.T_diff_mean_limit,
                last_point_diff_prop / self.params.last_point_diff_prop_limit
            ]
            self.max_criteria = max(criteria)

            if self.max_criteria >= 1:
                self.state = States.SatisfyingCriteria
                return

            goal = self.params.criteria_goal if self.percent < 100 else self.params.criteria_final_goal
            if self.max_criteria < goal:
                self.state = States.SolvedTask
                return

            if self.last_point_diff_prev is not None:
                cosine = cos_angle(self.last_point_diff_prev, last_point_diff)
                C_cos = pow(cosine, 2) / 3 + cos(cosine * pi / 2) / 100
                cosine_limit = C_cos
                mu = 0.7
                if cosine > 0.4:
                    mu = 1.3
                elif cosine < -0.4:
                    mu = 0.5
                # cos -1 -> 0.5
                # cos  1 -> 1.3
                log10_ratio = log10(self.ratio)
                log10_criteria = log10(self.max_criteria)
                Stats(self.shared_data).add(cosine, C_cos, mu, log10_ratio, log10_criteria)
                dprint(f'cosine: {cosine:.3g}\t'
                       f'C_cos: {C_cos:.3g}\t'
                       f'mu: {mu:.3g}\t'
                       f'ratio: {self.ratio:.3g}\t'
                       f'max_criteria: {self.max_criteria:.3g}')

                min_limit = min(self.ratio, cosine_limit)
                new_ratio = min(min_limit * mu, 1 / 3)
            else:
                new_ratio = self.ratio

            new_state = System(self.task_gen, self.guess, params=self.params, shared_data=self.shared_data,
                               builtin_moments=self.applied_moments)
            new_state.last_point_diff_prev = last_point_diff
            new_state.percent = self.percent
            new_state.iteration = self.iteration + 1
            new_state.ratio = new_ratio
            new_state.max_criteria = self.max_criteria
            new_state.state = States.UpdatingTask
            return new_state

        if self.state == States.SatisfyingCriteria:
            self.ratio /= (self.max_criteria * 1.001)
            self.state = States.SolvingTask
            return

        if self.state == States.SolvedTask:
            self.guess = None
            if self.percent >= 100:
                self.state = States.Finished
                return
            else:
                self.state = States.IncrementingLoad
                return

        if self.state == States.Finished:
            return

        raise Exception('unknown system state')

    def is_adjusting_ratio(self) -> bool:
        return self.state == States.AdjustingRatio

    def is_updating_task(self) -> bool:
        return self.state == States.UpdatingTask or self.is_task_boundary()

    def is_task_boundary(self) -> bool:
        return self.state == States.IncrementingLoad or self.is_finished()

    def is_finished(self) -> bool:
        return self.state == States.Finished

    def get_next(self) -> Optional['System']:
        if self.state != 'guessed':
            return

        new_state = System(self.task_gen, self.guess, params=self.params, shared_data=self.shared_data,
                           builtin_moments=self.builtin_moments)
        return new_state

    def get_state_text(self) -> str:
        pp = lambda val, precision: format(val, f'.{precision}g')

        t = self.state.name
        if self.state == States.Applying:
            t += f'({self.applying_to_num + 1})'
        t += f'\npercent: {round(self.percent, 7) if self.percent is not None else None}'
        t += f'\niteration: {self.iteration}'
        t += f'\nratio: {pp(self.ratio, 5) if self.ratio is not None else None}'
        t += f'\ncrit: {pp(self.max_criteria, 5) if self.max_criteria is not None else None}'
        return t

    def show_stats(self):
        Stats(self.shared_data).show()

    def get_shared_data(self) -> dict:
        return self.shared_data

    def set_shared_data(self, d: dict):
        self.shared_data = d

    def serialize(self) -> dict:
        obj_ser = lambda o: o.serialize() if o is not None else None
        obj_list_ser = lambda l: [obj_ser(o) for o in l] if l is not None else None
        np_ser = lambda a: a.tolist() if a is not None else None
        np_list_ser = lambda l: [np_ser(a) for a in l] if l is not None else None

        d = {
            'state': self.state.name,
            'params': obj_ser(self.params),

            'current_task': obj_ser(self.current_task),
            'applying_to_num': self.applying_to_num,
            'ratio': self.ratio,
            'max_criteria': self.max_criteria,
            'last_point_diff_prev': np_ser(self.last_point_diff_prev),
            'iteration': self.iteration,
            'percent': self.percent,

            'builtin_moments': np_list_ser(self.builtin_moments),
            'applied_moments': np_list_ser(self.applied_moments),

            'elements': obj_list_ser(self.elements),
            'guess': obj_list_ser(self.guess),
        }

        return d

    def deserialize(self, d: dict) -> 'System':
        obj_des = lambda cls, o: cls.deserialize(o) if o is not None else None
        obj_list_des = lambda cls, l: [obj_des(cls, o) for o in l] if l is not None else None
        np_des = lambda a: array(a) if a is not None else None
        np_list_des = lambda l: [np_des(a) for a in l] if l is not None else None

        system = System(self.task_gen, elements=[], params=obj_des(Params, d['params']))
        system.state = States[d['state']]

        system.current_task = obj_des(Task, d['current_task'])
        system.applying_to_num = d['applying_to_num']
        system.ratio = d['ratio']
        system.max_criteria = d['max_criteria']
        system.last_point_diff_prev = np_des(d['last_point_diff_prev'])
        system.iteration = d['iteration']
        system.percent = d['percent']

        system.builtin_moments = np_list_des(d['builtin_moments'])
        system.applied_moments = np_list_des(d['applied_moments'])

        system.elements = obj_list_des(Element, d['elements'])
        system.guess = obj_list_des(Element, d['guess'])

        return system

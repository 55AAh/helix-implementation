from typing import List, Generator, Tuple, Optional

from matplotlib.axes import Axes
from numpy import array as point
from numpy import array as vector
from numpy import linspace, cross, array, arccos, clip, dot, pi, cos, sin, vstack
from numpy.linalg import norm
from typing_extensions import Self

from element import Element
from plotter import ISystem, DrawParams


def dprint(*_args, **_kwargs):
    # print(*_args, **_kwargs)
    pass


class System(ISystem):
    def __init__(self, elements: List[Element], valid_threshold: float, arm_position: float = 0.5,
                 applied_moments: Optional[List[vector]] = None):
        """ Створює систему елементів

         :param elements: набір елементів
        """

        self.elements = elements
        self.guess = self.final_guess = None
        self.task_gen = None
        self.state = 'start'
        self.valid_threshold = valid_threshold
        self.arm_position = arm_position

        if applied_moments is None:
            applied_moments = array([vector([0, 0, 0]) for _ in elements])
        self.applied_moments = applied_moments

        self.guess_moments = None
        self.doing_final_guess = True

    def set_task(self, task_gen: Generator[Tuple[float, vector, vector, point], None, None]):
        self.task_gen = task_gen
        try:
            next(task_gen)
        except StopIteration:
            pass

    def plot(self, ax: Axes, current_state_num: int,
             elements_params: DrawParams, guess_params: DrawParams, final_guess_params: DrawParams,
             plot_elements_segments: int, plot_basis_vectors_len: float, plot_basis_vectors_segments: int):

        plots = [(self.elements, '-', 'black', 'Elements', elements_params)]
        if self.guess is not None:
            plots.append((self.guess, ':', 'tab:orange', 'Guess', guess_params))
        if self.final_guess is not None:
            plots.append((self.final_guess, '-.', 'tab:green', 'Final', final_guess_params))

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

    def apply_task(self):
        if self.state != 'start':
            return

        _ratio = [1]

        if self.doing_final_guess:
            try:
                mult, m0, force, force_point = self.task_gen.send(self)
            except StopIteration:
                print('SHOULD NOT PRINT')
                return

            if not isinstance(m0, list):
                m0 = [m0] * len(self.elements)  # TODO remove

            final_moments = []
            self.final_guess = []
            for i, e in enumerate(self.elements):
                apply_point = e.point(e.s * self.arm_position)
                arm = force_point - apply_point
                force_moment = cross(arm, force)
                total_moment = m0[i] + force_moment

                final_moments.append(total_moment)
                self.final_guess.append(e.apply_moment(total_moment))

                dprint(f'Для елементу номер {i + 1}:\n'
                       f'\tсередина елемента = {apply_point}\n'
                       f'\tсила прикладена в = {force_point}\n'
                       f'\tплече             = {arm}\n'
                       f'\tсила              = {force}\n'
                       f'\tмомент від сили   = {force_moment}\n'
                       f'\tпочатковий момент = {m0}\n'
                       f'\tсумарний момент   = {total_moment}\n'
                       f'\tK було            = {float(e.K):.5}\n'
                       f'\tK стало           = {float(self.final_guess[-1].K):.5}\n'
                       f'\tn було            = {e.nat[1]}\n'
                       f'\tn стало           = {self.final_guess[-1].nat[1]}\n')

            final_moments = array(final_moments)
            # TODO remove
            self._final_moments = final_moments
            self.guess = self.final_guess

            a = self.applied_moments
            g = final_moments
            n = a + (g - a) * mult
            self.guess_moments = n

            _ratio[0] = mult

        # self.guess = [e.apply_moment(_m) for e, _m in zip(self.elements, self.guess_moments)]

        self.guess = [e.interpolate(f, _ratio[0]) for e, f in zip(self.elements, self.final_guess)]

        # self.guess = self.final_guess

        # diff = []
        # for new, old in zip(self.final_guess, self.elements):
        #     for s in linspace(0, 1, 30):
        #         diff.append(norm(new.point(new.s * s) - old.point(old.s * s)))
        # mean_diff = mean(diff)
        # if mean_diff != 0:
        #     mean_diff_c = mult / mean_diff
        # else:
        #     mean_diff_c = multf
        # mean_diff_cl = clip(mean_diff_c, 0, multf)
        # mean_diff_cl = 1
        # # dprint(f'\t\tMD = {mean_diff}, {mean_diff_c}, {mean_diff_cl}')
        #
        # self.moments = [p + (n - p) * mean_diff_cl for p, n in zip(self.moments, m)]
        # self.guess = [e.apply_moment(_m) for e, _m in zip(self.elements, self.moments)]

        self.state = 'guessed'

    def join_guesses(self):
        """ Поєднує елементи (кінці та матеріальні базиси) """

        if self.state != 'guessed':
            return

        if self.doing_final_guess:
            el = [self.final_guess, self.guess]
        else:
            el = [self.guess]

        for el in el:
            for e_p, e_n in zip(el, el[1:]):
                border_p = e_p.point(e_p.s)
                e_n.p = border_p

                border_mat, border_nat = e_p.bases(e_p.s)

                n_begin_xi = e_n.mat[1]
                n_begin_eta = e_n.mat[2]
                n_begin_n = e_n.nat[1]
                n_begin_beta = e_n.nat[2]

                # Кут між матеріальним та природним базисами
                psi = arccos(clip(dot(n_begin_n, n_begin_xi), -1, 1))
                if arccos(clip(dot(n_begin_n, n_begin_eta), -1, 1)) < pi / 2:
                    psi *= -1

                end_xi = border_mat[1]
                end_eta = border_mat[2]

                end_t = border_mat[0]
                end_n = end_xi * cos(psi) - end_eta * sin(psi)
                end_beta = end_xi * sin(psi) + end_eta * cos(psi)

                e_n.mat = border_mat
                e_n.nat = vstack((end_t, end_n, end_beta))

        self.state = 'joined'

        try:
            next(self.task_gen)
        except StopIteration:
            print('SHOULD NOT PRINT')

    def validate(self) -> bool:
        if self.state == 'valid':
            return True

        if self.state != 'joined':
            return False

        pts_c = 11
        elements_pts = [e.point(e.s * i) for e in self.elements for i in linspace(0, 1, pts_c)]
        guess_pts = [e.point(e.s * i) for e in self.guess for i in linspace(0, 1, pts_c)]
        diff = [norm(p - fg_p) for p, fg_p in zip(elements_pts, guess_pts)]
        max_diff = array(diff).max(initial=0)

        if max_diff < self.valid_threshold:

            e = self.elements[-1]
            g = self.guess[-1]

            K_before = e.K
            n_before = e.nat[1]

            K_after = g.K
            n_after = g.nat[1]

            dprint(f'Для останнього елементу:\n'
                   f'\tМомент приклали   = {self._final_moments[-1]}\n'
                   f'\tМомент був        = {self.applied_moments[-1]}\n'
                   f'\tМомент став       = {self.guess_moments[-1]}\n'
                   f'\tK було            = {float(K_before):.5}\n'
                   f'\tK стало           = {float(K_after):.5}\n'
                   f'\tn було            = {n_before}\n'
                   f'\tn стало           = {n_after}\n'
                   f'================================')

            self.state = 'valid'
            return True
        else:
            self.state = 'start'

            a = self.applied_moments
            g = self.guess_moments
            n = a + (g - a) / 2
            self.guess_moments = n

            self.doing_final_guess = False
            return False

    def get_next(self) -> Optional[Self]:
        if self.state != 'valid':
            return

        new_state = System(self.guess, self.valid_threshold, arm_position=self.arm_position,
                           applied_moments=self.guess_moments.copy())
        new_state.set_task(self.task_gen)
        return new_state

    def serialize(self) -> Optional[dict]:
        e_ser = lambda e: [e.serialize() for e in e] if e is not None else None
        a_ser = lambda a: a.tolist() if a is not None else None

        d = {
            'state': self.state,
            'doing_final_guess': self.doing_final_guess,
            'elements': e_ser(self.elements),
            'guess': e_ser(self.guess),
            'final_guess': e_ser(self.final_guess),
            'applied_moments': a_ser(self.applied_moments),
            'guess_moments': a_ser(self.guess_moments),
        }

        return d

    def deserialize(self, d: dict):
        e_des = lambda e: [Element.deserialize(e) for e in e] if e is not None else None
        a_des = lambda a: array(a) if a is not None else None

        self.state = d['state']
        self.doing_final_guess = d['doing_final_guess']
        self.elements = e_des(d['elements'])
        self.guess = e_des(d['guess'])
        self.final_guess = e_des(d['final_guess'])
        self.applied_moments = a_des(d['applied_moments'])
        self.guess_moments = a_des(d['guess_moments'])

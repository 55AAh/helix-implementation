import io
import os

from matplotlib import pyplot as plt
from matplotlib.widgets import Button, CheckButtons
from numpy import array as point
from numpy import array as vector, pi, sqrt, cos, sin, cross, printoptions
from numpy import dot
from numpy import linspace
from numpy import round, array_equal
from numpy.linalg import norm

from element import Element
from plotter import Plotter
from system import System, Params, Task


class GlobalParams:
    def __init__(self):
        self.interactive = True
        self.batch = False
        self.save_analytic_plots = False
        self.show_analytic_plots = True
        self.use_threshold = True
        self.arm_position = 0.0
        self.arm_position_i = False
        self.arm_position_m = False

    def pick(self):
        if input('Use gui? (y/n): ') == 'y':
            fig, _ax = plt.subplots()
            plt.axis('off')
            cb_params = {
                'Interactive': 'interactive',
                'Run tests in batch': 'batch',
                'Save analytic plots': 'save_analytic_plots',
                'Show analytic plots': 'show_analytic_plots',
                'Use threshold': 'use_threshold',
                'Arm position inverted': 'arm_position_i',
                'Arm position at middle': 'arm_position_m',
            }
            cb = CheckButtons(fig.add_axes([0.05, 0.3, 0.9, 0.6]),
                              labels=list(cb_params.keys()),
                              actives=[self.__dict__[k] for k in cb_params.values()])
            cb.on_clicked(lambda *e, **k: print(cb.get_status()))

            save_button = Button(fig.add_axes([0.05, 0.1, 0.2, 0.1]), 'Save')
            save_button.on_clicked(lambda _: plt.close())
            plt.show(block=True)
            self.arm_position = 0.5 if self.arm_position_m else (1.0 if self.arm_position_i else 0.0)
        else:
            self.interactive = input('Interactive? (y/n): ') == 'y'
            self.batch = input('Run tests in batch? (y/n): ') == 'y'
            self.save_analytic_plots = input('Save analytic plots? (y/n): ') == 'y'
            self.show_analytic_plots = input('Show analytic plots? (y/n): ') == 'y'
            self.use_threshold = input('Use threshold? (y/n): ') == 'y'
            self.arm_position = float(input('Arm position? (float): '))


global_params = GlobalParams()


def dprint(*_args, **_kwargs):
    print(*_args, **_kwargs)
    pass


def save_fig(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path)


class SilentPrinter:
    rewrite_last_printed = False
    original_print = None
    last_printed = ''

    def __enter__(self):
        SilentPrinter.rewrite_last_printed = True

    def __exit__(self, *_):
        print()
        SilentPrinter.rewrite_last_printed = False

    @staticmethod
    def print(*args, **kwargs):
        if SilentPrinter.rewrite_last_printed:
            output = io.StringIO()
            SilentPrinter.original_print(*args, file=output, end='', **kwargs)
            contents = output.getvalue().replace('\n', '\t').replace('\t', ' ' * 4)[-100:]
            output.close()
            to_print = '\r' + ' ' * len(SilentPrinter.last_printed) + '\r' + contents
            SilentPrinter.original_print(to_print, end='')
            SilentPrinter.last_printed = contents
        else:
            SilentPrinter.original_print(*args, **kwargs)
            SilentPrinter.last_printed = ''


SilentPrinter.original_print = print
print = SilentPrinter.print


def analyze_results(results, precision=6):
    rounded = [round(p, precision) for p in results]
    while len(rounded) > 2 and array_equal(rounded[-2], rounded[-1]):
        del rounded[-1]
        del results[-1]
    print(f'\n\n\tTotal iterations: {len(rounded)}')
    with printoptions(suppress=True):
        print(f'\tResult : {results[-1]}')
        print(f'\tRounded: {rounded[-1]}')
    return rounded[-1]


def upward_force(elements_count: int = 10, la: float = 1):
    percent_steps = 10
    total_length = 10
    each_length = total_length / elements_count
    start_s_values = linspace(0, total_length, elements_count + 1)[:-1]
    EI = 1
    GJ = 1

    elements = [Element(
        p=point([s, 0, 0]),
        s=each_length,
        t=vector([1, 0, 0]),
        xi=vector([0, 1, 0]),
        K0=0,
        T0=0,
        EI=EI,
        GJ=GJ,
    ) for s in start_s_values]

    results = []

    def task():
        system = yield
        while system.percent <= 100:
            task_mult = system.percent / 100
            m0 = vector([0, 0, 0])
            f = task_mult * la * vector([0, 0, 1])
            fp = system.get_last_point()
            print(f'{system.percent}% load, iteration {system.iteration}:\tlast point = {fp}')
            results.append(fp)
            system = yield Task(m0, f, fp)

    initial_system = System(
        task(),
        elements,
        params=Params(
            use_threshold=global_params.use_threshold,
            arm_position=global_params.arm_position,
            percent_steps=percent_steps,
            criteria_goal=0.001,
            criteria_final_goal=0.0000001,
            threshold_goal=0.01,
        ),
        record_stats=global_params.interactive,
    )

    # noinspection PyArgumentEqualDefault
    visual = Plotter(
        initial_system, global_params.interactive,
        pause=0.01,
        skip_plots=0,
        plot_elements_segments=30,
        plot_basis_vectors_len=0.5,
        next_cnt=percent_steps,
        major={
            'problem': 'upward_force',
            'elements_count': elements_count,
        },
        minor={
            'total_length': total_length,
            'EI': EI, 'GJ': GJ,
            'la': la,
        },
    )
    visual.run()
    return analyze_results(results)


def Bathe(elements_count: int = 10, la: float = 7.2):
    angle = pi / 4
    r = 100
    each_length = angle * r / elements_count
    EI = 1
    GJ = 1

    elements = [Element(
        p=point([r * (1 - cos(s)), r * sin(s), 0]),
        s=each_length,
        t=vector([sin(s), cos(s), 0]),
        xi=vector([cos(s), sin(s), 0]),
        K0=1 / r, T0=0,
        EI=EI, GJ=GJ,
    ) for s in linspace(0, angle, elements_count + 1)[:-1]]

    force = la * EI / r ** 2

    results = []

    def task():
        system = yield
        while system.percent <= 100:
            m0 = vector([0, 0, 0])
            f = force * vector([0, 0, 1])
            fp = system.get_last_point()
            with printoptions(precision=10, suppress=True):
                print(f'Ітерація {system.iteration}, остання точка = {fp}')
            results.append(fp)
            if len(results) >= 2 and array_equal(round(results[-2], 8), round(results[-1], 8)):
                break
            system = yield Task(m0, f, fp)
        yield

    initial_system = System(
        task(),
        elements,
        params=Params(
            use_threshold=global_params.use_threshold,
            arm_position=global_params.arm_position,
            criteria_goal=1e-7,
            criteria_final_goal=1e-7,
            threshold_goal=0.1,
        ),
        record_stats=global_params.interactive,
    )

    # noinspection PyArgumentEqualDefault
    visual = Plotter(
        initial_system, global_params.interactive,
        pause=0.01,
        skip_plots=0,
        plot_elements_segments=30, plot_basis_vectors_len=0.5,
        major={
            'problem': 'Bathe',
            'elements_count': elements_count,
        },
        minor={
            'angle': angle,
            'r': r,
            'EI': EI, 'GJ': GJ,
            'la': la,
        },
    )
    visual.run()
    return analyze_results(results)


def Ibrahimbegovich_small(elements_count: int = 10, la: float = 0):
    total_length = 10

    each_length = total_length / elements_count
    start_s_values = linspace(0, total_length, elements_count + 1)[:-1]

    elements = [Element(
        p=vector([s, 0, 0]),
        s=each_length,
        t=vector([1, 0, 0]),
        xi=vector([0, 1, 0]),
        K0=0, T0=0,
        EI=100, GJ=100,
    ) for s in start_s_values]

    results = []

    def task():
        title = f'\t\t\t\t\t\tElements: {elements_count}\n\n\t\tDisplacement components under end moment'
        if la != 0:
            title += ' and pert. force'
        print(title + '\n')

        system = yield
        while system.percent <= 100:
            m0 = 2.5 * pi * vector([0, 0, 1])
            f = la * vector([0, 0, 1])
            fp = system.get_last_point()

            offset = system.get_last_point() - initial_system.get_last_point()
            with printoptions(precision=10, suppress=True):
                print(f'\tIteration {system.iteration:<5}', offset)
            results.append(offset)
            if len(results) >= 2 and array_equal(round(results[-2], 8), round(results[-1], 8)):
                break
            system = yield Task(m0, f, fp)

    initial_system = System(
        task(),
        elements,
        params=Params(
            use_threshold=global_params.use_threshold,
            arm_position=global_params.arm_position,
            threshold_goal=0.01,
        ),
        record_stats=global_params.interactive,
    )

    # noinspection PyArgumentEqualDefault
    visual = Plotter(
        initial_system, global_params.interactive,
        pause=0.01,
        skip_plots=0,
        plot_elements_segments=30,
        plot_basis_vectors_len=0.5,
        major={
            'problem': 'Ibrahimbegovich_small',
            'elements_count': elements_count,
        },
        minor={
            'total_length': total_length,
        },
    )
    visual.run()
    return analyze_results(results)


def Ibrahimbegovich_big(elements_count: int = 10, percent_steps: int = 1):
    total_length = 10
    each_length = total_length / elements_count
    start_s_values = linspace(0, total_length, elements_count + 1)[:-1]
    EI = 100
    GJ = 100

    elements = [Element(
        p=vector([s, 0, 0]),
        s=each_length,
        t=vector([1, 0, 0]),
        xi=vector([0, 1, 0]),
        K0=0, T0=0,
        EI=EI, GJ=GJ,
    ) for s in start_s_values]

    mode_m = '>'
    mode_f = '>'

    def task():
        system = yield

        data['displacements'].append((system.get_last_point() - system.elements[0].point(0)).tolist())
        data['displacements_marks'].append(0)
        data['displacements_p'].append(data['displacements'][0])
        data['displacements_p_marks'].append(0)

        while system.percent <= 100:
            task_mult = system.percent / 100

            if mode_m == '0':
                mult_m = 0
            elif mode_m == '>':
                mult_m = task_mult
            elif mode_m == '1':
                mult_m = 1
            else:
                raise Exception('Unknown m mode')

            if mode_f == '0':
                mult_f = 0
            elif mode_f == '>':
                mult_f = task_mult
            elif mode_f == '1':
                mult_f = 1
            else:
                raise Exception('Unknown f mode')

            m0 = mult_m * 200 * pi * vector([0, 0, 1])
            f = mult_f * -50 * vector([0, 0, 1])
            fp = system.get_last_point()

            _percent = system.percent
            print(f'{round(_percent, 5)}% load, iteration {system.iteration:<5} last point = {fp}')
            system = yield Task(m0, f, fp)

            c_begin = system.elements[0].point(0)
            c_end = system.get_last_point()
            data['displacements'].append((c_end - c_begin).tolist())
            if system.percent != _percent or system.state.name == 'Finished':
                data['displacements_marks'].append(len(data['displacements']) - 1)
                data['displacements_p_marks'].append(len(data['displacements_p']))
                data['displacements_p'].append(data['displacements'][-1])

    initial_system = System(
        task(),
        elements,
        params=Params(
            use_threshold=global_params.use_threshold,
            arm_position=global_params.arm_position,
            percent_steps=percent_steps,
            criteria_goal=1e-7,
            criteria_final_goal=1e-7,
            threshold_goal=1.1,
        ),
        record_stats=global_params.interactive,
    )

    data = dict()
    initial_system.shared_data['Ibrahimbegovich_big'] = data
    data['displacements'] = []
    data['displacements_marks'] = []
    data['displacements_p'] = []
    data['displacements_p_marks'] = []

    # noinspection PyArgumentEqualDefault
    visual = Plotter(
        initial_system, global_params.interactive,
        pause=0.01,
        skip_plots=0,
        plot_elements_segments=max(1, int(30 * each_length)),
        plot_basis_vectors_len=each_length / 3,
        next_cnt=percent_steps,
        lims=((-0.17443378249741953, 0.18093407976867604),
              (-0.01892765241459829, 0.33644020985149725),
              (-0.007159928998741932, 0.25936596770082976)),
        major={
            'problem': 'Ibrahimbegovich_big',
            'elements_count': elements_count,
        },
        minor={
            'total_length': total_length,
            'EI': EI, 'GJ': GJ,
        },
    )
    visual.run()

    data = visual.states[-1].shared_data['Ibrahimbegovich_big']

    if mode_f != '0':
        fn = f'out/Ibrahimbegovich_big/ap={global_params.arm_position}/ps={percent_steps}_ec={elements_count}'
        if percent_steps > 1 and len(data['displacements_p']) > 0:
            disp_final = data['displacements_p'][-1][2]
            displacements_p_z_i = [disp_final - d[2] for d in vector(data['displacements_p'])]
            plt.plot(displacements_p_z_i, range(len(displacements_p_z_i)))
            plt.axvline(linestyle='dotted', alpha=0.5)
            plt.title(
                'Free-end displacement component in the direction of applied force\n'
                'Ibrahimbegovich style (shifted & flipped & only percent steps)')
            plt.margins(y=0)
            if global_params.save_analytic_plots:
                save_fig(fn + '_paper_style')
            if global_params.show_analytic_plots:
                plt.show()
            plt.close()

        displacements_z = [d[2] for d in vector(data['displacements'])]
        plt.plot(displacements_z, range(len(displacements_z)))
        for mark in data['displacements_marks'][::10]:
            plt.axhline(mark, linestyle='--', alpha=0.5)
        plt.title('Free-end displacement component in the direction of applied force')
        if global_params.save_analytic_plots:
            save_fig(fn)
        if global_params.show_analytic_plots:
            plt.show()
        plt.close()

    return data['displacements_p']


def ideal_helix(elements_count: int = 1, use_m0: bool = False, guess_needed_criteria: bool = False):
    a = 2
    h = 0.05

    EI = 1
    GJ = 0.8

    ah_s = a ** 2 + h ** 2
    ah_sr = sqrt(ah_s)

    K = a / ah_s
    T = h / ah_s

    KT_s = K ** 2 + T ** 2
    KT_sr = sqrt(KT_s)

    winds = 5
    total_length = 2 * pi * winds * ah_sr

    each_length = total_length / elements_count

    phi_values = linspace(0, 2 * pi * winds, elements_count + 1)[:-1]
    start_s_values = [phi * ah_sr for phi in phi_values]

    ideal_ts = [vector([a * cos(phi), a * sin(phi), h]) / ah_sr for phi in phi_values]
    ideal_betas = [vector([-h * cos(phi), -h * sin(phi), a]) / ah_sr for phi in phi_values]

    if use_m0:
        ideal_moments = []
        for t, beta in zip(ideal_ts, ideal_betas):
            # mt = -T * GJ * t
            # mb = -K * EI * beta
            mt = T * GJ * t
            mb = K * EI * beta
            m0 = mt + mb
            ideal_moments.append(m0)
        f = vector([0, 0, 0])
        m0 = ideal_moments
        _mm = ideal_moments[0]
    else:
        # P = (EI - GJ) * T * KT_sr
        # Z = -(EI * K ** 2 + GJ * T ** 2) / KT_sr
        # # P = (K * h * EI - T * a * GJ) / ah_sr
        # # Z = (-K * a * EI - T * h * GJ) / ah_sr

        P = -(EI - GJ) * T * KT_sr
        Z = (EI * K ** 2 + GJ * T ** 2) / KT_sr
        # P = -(K * h * EI - T * a * GJ) / ah_sr
        # Z = -(-K * a * EI - T * h * GJ) / ah_sr
        m0 = Z * vector([0, 0, 1])
        f = P * vector([0, 0, 1])
        _mm = vector([a * P, 0, Z])
    fp = point([0, 0, 0])

    ideal_end = point([a * sin(2 * pi * winds), -a * cos(2 * pi * winds), 2 * pi * winds * h])

    xi = vector([0, 1, 0])
    angle = pi / 3
    xi = cos(angle) * xi + sin(angle) * cross(ideal_ts[0], xi)

    elements = [Element(
        p=ideal_ts[0] * s - vector([0, a, 0]),
        s=each_length,
        t=ideal_ts[0],
        xi=xi,
        K0=0, T0=0,
        EI=EI, GJ=GJ,
    ) for s in start_s_values]

    # _beta = (_mm - dot(_mm, ideal_ts[0]) * ideal_ts[0]) / EI
    # _dTh_norm = norm(_beta * each_length)
    # _dTh_diff_angle = pi / 2
    _T = dot(_mm, ideal_ts[0]) / GJ
    _last_point_diff_ratio = a / ah_sr

    results = []

    def task():
        system = yield
        while system.percent <= 100:
            dprint()
            got_end = system.get_last_point()
            with printoptions(precision=10, suppress=True):
                dprint(f'Обчислюємо похибку:\n'
                       f'\tостання точка: {got_end}\n'
                       f'\tмає бути:      {ideal_end}\n'
                       f'\tрізниця:       {got_end - ideal_end}\n'
                       f'\tнорма різниці: {norm(got_end - ideal_end)}\n')
            results.append(got_end - ideal_end)

            system = yield Task(m0, f, fp)

    criteria_guesses = {
        'dTh_diff_angle_max_limit': 0.001,
        'dTh_diff_angle_mean_limit': 0.001,
        'T_diff_max_limit': _T * 1.001,
        'T_diff_mean_limit': _T * 1.001,
        'last_point_diff_prop_limit': _last_point_diff_ratio * 1.001,
    }

    initial_system = System(
        task(),
        elements,
        params=Params(
            use_threshold=global_params.use_threshold,
            arm_position=global_params.arm_position,
            **(criteria_guesses if guess_needed_criteria else dict()),
            criteria_goal=0.000001,
            criteria_final_goal=0.000001,
            threshold_goal=1.1,
        ),
        record_stats=global_params.interactive,
    )

    # noinspection PyArgumentEqualDefault
    visual = Plotter(
        initial_system, global_params.interactive,
        pause=0.01,
        skip_plots=0,
        plot_elements_segments=max(1, int(30 * each_length)),
        plot_basis_vectors_len=a / each_length / 3,
        lims=((-a * 1.1, a * 1.1), (-a * 1.1, a * 1.1), (-0.1, 2 * pi * winds * h * 1.1)),
        major={
            'problem': 'ideal_helix',
            'elements_count': elements_count,
            'use_m0': use_m0,
        },
        minor={
            'a': a, 'h': h,
            'EI': EI, 'GJ': GJ,
            'winds': winds,
        },
    )
    visual.run()
    return analyze_results(results)


# noinspection PyArgumentEqualDefault
def main():
    while True:
        test = input('''
1 - вертикальна сила до горизонтальної прямої балки
2 - вертикальна сила до балки-дуги кола в горизонтальній площині - задача Бате
3 - спіраль з прямої балки - ідеальний хелікс
4 - вигинання прямої балки - спрощена задача Ібрахімбеговича
5 - спіраль з прямої балки - сила, прикладена до краю - задача Ібрахімбеговича
0 - змінити параметри
  - press Enter to exit
> ''')
        if not test:
            break

        ELEMENTS = [3, 10, 30, 100, 300, 1000]

        if test == '0':
            global_params.pick()
        elif test == '1':
            if global_params.batch:
                for la in [1, 3, 10]:
                    print(f'\nUpward force, λ = {la}')
                    for elements_count in ELEMENTS:
                        with SilentPrinter():
                            result = upward_force(elements_count, la)
                        with printoptions(suppress=True):
                            print(f'{elements_count:<4} elements: {result}')
            else:
                elements_count = int(input('Elements count? (int): '))
                la = float(input('λ? (float): '))
                upward_force(elements_count, la)

        elif test == '2':
            if global_params.batch:
                for la, av, bv in [(0, point([29.3, 70.7, 0.0]), point([29.3, 70.7, 0.0])),
                                   (3.6, point([22.2, 58.8, 40.2]), point([22.5, 59.2, 39.5])),
                                   (7.2, point([15.6, 47.1, 53.6]), point([15.9, 47.2, 53.4]))]:
                    print(f'\nBathe, λ = {la}')
                    for elements_count in ELEMENTS:
                        with SilentPrinter():
                            result = Bathe(elements_count, la)
                        with printoptions(suppress=True):
                            print(f'{elements_count:<4} elements: '
                                  f'{result},'
                                  f'\tdiff_albino: {norm(result - av):.3},'
                                  f'\tdiff_bathe: {norm(result - bv):.3}')
            else:
                elements_count = int(input('Elements count? (int): '))
                la = float(input('λ? (float): '))
                Bathe(elements_count, la)

        elif test == '3':
            if global_params.batch:
                for use_m0 in [False, True]:
                    for guess_needed_criteria in [False, True]:
                        print(f'\nideal helix, use_m0 = {use_m0}, guess_needed_criteria = {guess_needed_criteria}')
                        for elements_count in ELEMENTS:
                            with SilentPrinter():
                                result = ideal_helix(elements_count, use_m0, guess_needed_criteria)
                            with printoptions(suppress=True):
                                print(f'{elements_count:<4} elements: {result}')
            else:
                variant = input('''
1   - один елемент
1*  - один елемент, m0
2   - 20 елементів
2*  - 20 елементів, m0
i1  - один елемент       1 ітерація    
i1* - один елемент, m0   1 ітерація
i2  - 20 елементів       1 ітерація
i2* - 20 елементів, m0   1 ітерація
  - press Enter to go back
> ''')
                if variant == '1':
                    ideal_helix(elements_count=1)
                elif variant == '1*':
                    ideal_helix(elements_count=1, use_m0=True)
                elif variant == '2':
                    ideal_helix(elements_count=20)
                elif variant == '2*':
                    ideal_helix(elements_count=20, use_m0=True)
                elif variant == 'i1':
                    ideal_helix(elements_count=1, guess_needed_criteria=True)
                elif variant == 'i1*':
                    ideal_helix(elements_count=1, use_m0=True, guess_needed_criteria=True)
                elif variant == 'i2':
                    ideal_helix(elements_count=20, guess_needed_criteria=True)
                elif variant == 'i2*':
                    ideal_helix(elements_count=20, use_m0=True, guess_needed_criteria=True)

        elif test == '4':
            if global_params.batch:
                for la in [0, 0.001]:
                    print(f'\nIbrahimbegovich small, λ = {la}')
                    for elements_count in ELEMENTS:
                        with SilentPrinter():
                            result = Ibrahimbegovich_small(elements_count, la)
                        with printoptions(suppress=True):
                            print(f'{elements_count:<4} elements: {result}')
            else:
                variant = input('''
1 - без сили
2 - мала сила
  - press Enter to go back
> ''')
                if variant == '1':
                    Ibrahimbegovich_small(elements_count=10, la=0)
                elif variant == '2':
                    Ibrahimbegovich_small(elements_count=10, la=0.001)

        elif test == '5':
            if global_params.batch:
                for percent_steps in [1, 10, 100, 1000]:
                    print(f'\nIbrahimbegovich big, percent_steps = {percent_steps}')
                    for elements_count in ELEMENTS:
                        with SilentPrinter():
                            result = Ibrahimbegovich_big(elements_count, percent_steps)
                        with printoptions(suppress=True):
                            print(f'{elements_count:<4} elements')
                            for i, r in enumerate(result):
                                print(f'\t{i / (len(result) - 1):<6.1%}:\t{r}')
            else:
                variant = input('''
1  - 1 крок         10  елементів
2  - 10 кроків      10  елементів
3  - 100 кроків     10  елементів
4  - 1000 кроків    10  елементів
1* - 1 крок         100 елементів
2* - 10 кроків      100 елементів
3* - 100 кроків     100 елементів
4* - 1000 кроків    100 елементів
  - press Enter to go back
> ''')
                if variant == '1':
                    Ibrahimbegovich_big(elements_count=10, percent_steps=1)
                elif variant == '2':
                    Ibrahimbegovich_big(elements_count=10, percent_steps=10)
                elif variant == '3':
                    Ibrahimbegovich_big(elements_count=10, percent_steps=100)
                elif variant == '4':
                    Ibrahimbegovich_big(elements_count=10, percent_steps=1000)
                elif variant == '1*':
                    Ibrahimbegovich_big(elements_count=100, percent_steps=1)
                elif variant == '2*':
                    Ibrahimbegovich_big(elements_count=100, percent_steps=10)
                elif variant == '3*':
                    Ibrahimbegovich_big(elements_count=100, percent_steps=100)
                elif variant == '4*':
                    Ibrahimbegovich_big(elements_count=100, percent_steps=1000)


if __name__ == '__main__':
    main()

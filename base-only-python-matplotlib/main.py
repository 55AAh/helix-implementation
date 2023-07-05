import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from numpy import array as point
from numpy import array as vector, pi, sqrt, cos, sin, cross, printoptions
from numpy import dot
from numpy import linspace
from numpy.linalg import norm

from element import Element
from plotter import Plotter
from system import System


def dprint(*_args, **_kwargs):
    # print(*_args, **_kwargs)
    pass


# noinspection PyPep8Naming
# noinspection SpellCheckingInspection
# noinspection Duplicates
def Ibrahimbegovich_small(force: bool = False):
    total_length = 10
    elements_count = 10

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

    initial_system = System(elements, valid_threshold=each_length * 10)

    state = dict()

    def task():
        mult = 1

        title = f'\t\t\t\t\t\tElements: {elements_count}\n\n\t\tDisplacement components under end moment'
        if force:
            title += ' and pert. force'
        print(title + '\n')

        state['i'] = 1
        while True:
            current_system: System = yield
            if current_system is None:
                continue

            force_val = 0.001 if force else 0

            m0 = 2.5 * pi * vector([0, 0, 1])
            f = force_val * vector([0, 0, 1])
            fp = current_system.elements[-1].point(current_system.elements[-1].s)

            yield mult, m0, f, fp

            with printoptions(precision=10, suppress=True):
                _i_str = str(state['i']) + ':'
                print(f'\tIteration {_i_str:<5}',
                      current_system.guess[-1].point(current_system.guess[-1].s) -
                      initial_system.elements[-1].point(initial_system.elements[-1].s))
            state['i'] += 1

    initial_system.set_task(task())

    # noinspection PyArgumentEqualDefault
    visual = Plotter(initial_system, pause=0.01,
                     major={'problem': 'Ibrahimbegovich_small',
                            'elements_count': elements_count},
                     minor={'total_length': total_length},
                     state=state)
    visual.show_interactive()


# noinspection PyPep8Naming
# noinspection SpellCheckingInspection
# noinspection Duplicates
def Ibrahimbegovich_big(task_steps=100):
    total_length = 10
    elements_count = 10
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

    initial_system = System(elements, valid_threshold=each_length * 100000)

    displacements = [initial_system.elements[-1].point(initial_system.elements[-1].s) -
                     initial_system.elements[0].point(0)]
    displacements_marks = [0]

    displacements_s = [displacements[0]]
    displacements_s_marks = [0]

    g_fp_inc_norm = []
    g_cosine = []
    g_C_cos = []
    g_mu = []
    g_mult = []
    g_mult_scale_limit = []

    state = dict()

    def task():
        for mode in ['mt']:

            state['percent'] = int(100 / task_steps)
            while state['percent'] <= 100:
                task_mult = state['percent'] / 100
                if task_mult < 0.41:
                    state['percent'] += int(100 / task_steps)
                    continue
                print(f'{task_mult:%} сили')

                fp_prev = None
                fp_inc_prev = None
                fp_inc_norm_prev = None
                mult = 0.01

                state['i'] = 1
                while state['i'] <= 1000:
                    current_system = None
                    while current_system is None:
                        current_system = yield

                    if mode == 'm':
                        m0 = task_mult * 200 * pi * vector([0, 0, 1])
                        f = 0 * 50 * vector([0, 0, 1])
                    elif mode == 't':
                        m0 = 1 * 200 * pi * vector([0, 0, 1])
                        f = -task_mult * 50 * vector([0, 0, 1])
                    else:
                        m0 = task_mult * 200 * pi * vector([0, 0, 1])
                        f = -task_mult * 50 * vector([0, 0, 1])
                    fp = current_system.elements[-1].point(current_system.elements[-1].s)

                    print(f'i = {state["i"]}', end='')
                    _append = lambda a, v: a.append(v) if state['i'] >= 3 else None
                    if fp_prev is not None:
                        fp_inc = fp - fp_prev
                        fp_inc_norm = norm(fp_inc)

                        if fp_inc_prev is not None:
                            cosine = dot(fp_inc, fp_inc_prev) / fp_inc_norm / fp_inc_norm_prev
                            C_cos = pow(cosine, 2) / 3 + cos(cosine * pi / 2) / 100
                            print(f'\tcos={cosine:.3}\tC_cos={C_cos:.3}', end='')

                            mu = 0.7
                            if cosine > 0.4:
                                mu = 1.1
                            elif cosine < -0.4:
                                mu = 0.9
                            # cos -1 -> 0.5
                            # cos  1 -> 1.3

                            print(f'\tmu={mu}', end='')

                            mult *= mu

                            # print(f'\tmult={mult:.3}', end='')
                            #
                            # if mult < 1e-6:
                            #     mult = 0.1
                            #     print('\tBOOST', end='')

                            _append(g_fp_inc_norm, fp_inc_norm)
                            _append(g_cosine, cosine)
                            _append(g_C_cos, C_cos)
                            _append(g_mu, mu)
                            _append(g_mult, np.log10(mult))

                        fp_inc_prev = fp_inc
                        fp_inc_norm_prev = fp_inc_norm

                        mult_scale_limit = total_length / 10 / fp_inc_norm
                        _append(g_mult_scale_limit, np.log10(mult_scale_limit))
                        if mult > mult_scale_limit:
                            print(f'\tscale lim: {mult:.3} -> {mult_scale_limit:.3}', end='')
                        mult = min(mult, mult_scale_limit)

                        mult_limit = 0.5
                        if mult > mult_limit:
                            print(f'\tlim: {mult:.3} -> {mult_limit:.3}', end='')
                        mult = min(mult, mult_limit)

                        print(f'\tmult={mult:.3}', end='')

                    fp_prev = fp
                    print(flush=True)

                    yield mult, m0, f, fp
                    state['i'] += 1

                    g_window = 100
                    if state['i'] % g_window == 0:
                        fig1, ax = plt.subplots()
                        ax.set_title(f'Last {g_window}')
                        x = range(max(state['i']-g_window, 3), state['i'])
                        ax.plot(x, g_fp_inc_norm[-g_window:], label='fp_inc_norm')
                        ax.plot(x, g_cosine[-g_window:], label='cosine')
                        ax.plot(x, g_C_cos[-g_window:], label='C_cos')
                        ax.plot(x, g_mu[-g_window:], label='mu')
                        ax.plot(x, g_mult[-g_window:], label='log10(mult)')
                        ax.plot(x, g_mult_scale_limit[-g_window:], label='log10(mult_scale_limit)')
                        plt.legend()

                        fig2, ax = plt.subplots()
                        ax.set_title('All')
                        x = range(3, state['i'])
                        ax.plot(x, g_fp_inc_norm, label='fp_inc_norm')
                        ax.plot(x, g_cosine, label='cosine')
                        ax.plot(x, g_C_cos, label='C_cos')
                        ax.plot(x, g_mu, label='mu')
                        ax.plot(x, g_mult, label='log10(mult)')
                        ax.plot(x, g_mult_scale_limit, label='log10(mult_scale_limit)')
                        plt.legend()
                        plt.pause(0.1)

                        _stop = [False]

                        def stop(*_args):
                            _stop[0] = True

                        b = Button(fig2.add_axes([0.81, 0.15, 0.1, 0.045]), 'Close')
                        b.on_clicked(stop)

                        while not _stop[0]:
                            plt.pause(1)

                        plt.pause(0.1)
                        plt.close(fig1)
                        plt.close(fig2)
                        plt.pause(0.1)

                state['percent'] += int(100 / task_steps)

    initial_system.set_task(task())

    cnt = 100
    skip = 0
    if task_steps == 10:
        cnt = 1000
        # skip = 4
    elif task_steps == 100:
        cnt = 10000
        skip = 99

    # noinspection PyArgumentEqualDefault
    visual = Plotter(initial_system, pause=0.01, skip_plots=skip, plot_elements_segments=max(1, int(30 * each_length)),
                     next_cnt=cnt, keep_lims=False,
                     lims=((-0.17443378249741953, 0.18093407976867604),
                           (-0.01892765241459829, 0.33644020985149725),
                           (-0.007159928998741932, 0.25936596770082976)),
                     major={'problem': 'Ibrahimbegovich_big',
                            'elements_count': elements_count},
                     minor={'total_length': total_length,
                            'EI': EI, 'GJ': GJ},
                     state=state)
    visual.show_interactive()

    # if task_steps == 100 or True:
    #     displacements_s = [d[2] for d in vector(displacements_s)]
    #     plt.plot(displacements_s, range(len(displacements_s)))
    #     for mark in displacements_s_marks[::10]:
    #         plt.axhline(mark, linestyle='--', alpha=0.5)
    #     plt.title('Free-end displacement component in the direction of applied force (only for task steps)')
    #     plt.show()
    #
    #     displacements = [d[2] for d in vector(displacements)]
    #     plt.plot(displacements, range(len(displacements)))
    #     for mark in displacements_marks[::10]:
    #         plt.axhline(mark, linestyle='--', alpha=0.5)
    #     plt.title('Free-end displacement component in the direction of applied force')
    #     plt.show()


# noinspection PyPep8Naming
# noinspection SpellCheckingInspection
# noinspection Duplicates
def ideal_helix_m0():
    a = 2
    h = 0.05

    EI = 1
    GJ = 0.8

    ah_s = a ** 2 + h ** 2
    ah_sr = sqrt(ah_s)

    K = a / ah_s
    T = h / ah_s

    elements_count = 20
    winds = 5
    total_length = 2 * pi * winds * ah_sr

    each_length = total_length / elements_count

    phi_values = linspace(0, 2 * pi * winds, elements_count + 1)[:-1]
    start_s_values = [phi * ah_sr for phi in phi_values]

    ideal_ts = [vector([a * cos(phi), a * sin(phi), h]) / ah_sr for phi in phi_values]
    ideal_betas = [vector([-h * cos(phi), -h * sin(phi), a]) / ah_sr for phi in phi_values]

    ideal_moments = []
    for t, beta in zip(ideal_ts, ideal_betas):
        # mt = -T * GJ * t
        # mb = -K * EI * beta
        mt = T * GJ * t
        mb = K * EI * beta
        m0 = mt + mb
        ideal_moments.append(m0)

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

    initial_system = System(elements, valid_threshold=each_length * 1000, arm_position=0)

    state = dict()

    def task():
        mult = 1

        state['i'] = 1
        while True:
            current_system: System = yield
            if current_system is None:
                continue

            # dprint('\n' * 20)
            # dprint('Обчислюємо наступну ітерацію')

            # dprint(f'Закладаємо фіксовані моменти 1:\n'
            #        f'\tT            = {T}\n'
            #        f'\tGJ           = {GJ}\n'
            #        f'\tt            = {ideal_ts[0]}\n'
            #        f'\tmt = -T·GJ·t = {m1t}\n'
            #        f'\n'
            #        f'\tK            = {K}\n'
            #        f'\tEI           = {EI}\n'
            #        f'\tβ            = {ideal_betas[0]}\n'
            #        f'\tmβ = -K·EI·β = {m1b}\n'
            #        f'\n'
            #        f'\tm0 = mt + mβ = {m10}\n')
            #
            # dprint(f'Закладаємо фіксовані моменти 2:\n'
            #        f'\tT            = {T}\n'
            #        f'\tGJ           = {GJ}\n'
            #        f'\tt            = {ideal_ts[1]}\n'
            #        f'\tmt = -T·GJ·t = {m2t}\n'
            #        f'\n'
            #        f'\tK            = {K}\n'
            #        f'\tEI           = {EI}\n'
            #        f'\tβ            = {ideal_betas[1]}\n'
            #        f'\tmβ = -K·EI·β = {m2b}\n'
            #        f'\n'
            #        f'\tm0 = mt + mβ = {m20}\n')

            f = vector([0, 0, 0])
            fp = point([0, 0, 0])

            yield mult, ideal_moments, f, fp

            # dprint(f'Має бути природний базис:\n'
            #        f'{vstack([ideal_ts[0], ideal_ns[0], ideal_betas[0]])}\n')
            #
            # dprint(f'Отримали\n'
            #        f'\tT = {current_system.guess[0].T}\n'
            #        f'\tK = {current_system.guess[0].K}\n'
            #        f'Має бути\n'
            #        f'\tT = {T}\n'
            #        f'\tK = {K}\n')

            dprint()
            got_end = current_system.guess[-1].point(current_system.guess[-1].s)
            with printoptions(precision=10, suppress=True):
                dprint(f'Обчислюємо похибку:\n'
                       f'\tостання точка: {got_end}\n'
                       f'\tмає бути:      {ideal_end}\n'
                       f'\tрізниця:       {got_end - ideal_end}\n'
                       f'\tнорма різниці: {norm(got_end - ideal_end)}\n')
            state['i'] += 1

    initial_system.set_task(task())

    # noinspection PyArgumentEqualDefault
    visual = Plotter(initial_system, pause=0.01, skip_plots=0,
                     plot_elements_segments=max(1, int(30 * each_length)), plot_basis_vectors_len=a / each_length / 3,
                     next_cnt=elements_count, keep_lims=True,
                     lims=((-a * 1.1, a * 1.1), (-a * 1.1, a * 1.1), (-0.1, 2 * pi * winds * h * 1.1)),
                     major={'problem': 'ideal_helix_1e',
                            'elements_count': elements_count},
                     minor={'a': a, 'h': h,
                            'EI': EI, 'GJ': GJ,
                            'winds': winds},
                     state=state)
    visual.show_interactive()


# noinspection PyPep8Naming
# noinspection SpellCheckingInspection
# noinspection Duplicates
def ideal_helix():
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

    elements_count = 20
    winds = 5
    total_length = 2 * pi * winds * ah_sr

    each_length = total_length / elements_count

    phi_values = linspace(0, 2 * pi * winds, elements_count + 1)[:-1]
    start_s_values = [phi * ah_sr for phi in phi_values]

    start_t = vector([a, 0, h]) / ah_sr

    # P = (EI - GJ) * T * KT_sr
    # Z = -(EI * K ** 2 + GJ * T ** 2) / KT_sr
    # # P = (K * h * EI - T * a * GJ) / ah_sr
    # # Z = (-K * a * EI - T * h * GJ) / ah_sr

    P = -(EI - GJ) * T * KT_sr
    Z = (EI * K ** 2 + GJ * T ** 2) / KT_sr
    # P = -(K * h * EI - T * a * GJ) / ah_sr
    # Z = -(-K * a * EI - T * h * GJ) / ah_sr

    ideal_end = point([a * sin(2 * pi * winds), -a * cos(2 * pi * winds), 2 * pi * winds * h])

    xi = vector([0, 1, 0])
    angle = pi / 3
    xi = cos(angle) * xi + sin(angle) * cross(start_t, xi)

    elements = [Element(
        p=start_t * s - vector([0, a, 0]),
        s=each_length,
        t=start_t,
        xi=xi,
        K0=0, T0=0,
        EI=EI, GJ=GJ,
    ) for s in start_s_values]

    initial_system = System(elements, valid_threshold=each_length * 10, arm_position=0)

    state = dict()

    def task():
        mult = 1

        state['i'] = 1
        while True:
            current_system: System = yield
            if current_system is None:
                continue

            # dprint('\n' * 20)
            # dprint('Обчислюємо наступну ітерацію')

            # dprint(f'Закладаємо фіксовані моменти 1:\n'
            #        f'\tT            = {T}\n'
            #        f'\tGJ           = {GJ}\n'
            #        f'\tt            = {ideal_ts[0]}\n'
            #        f'\tmt = -T·GJ·t = {m1t}\n'
            #        f'\n'
            #        f'\tK            = {K}\n'
            #        f'\tEI           = {EI}\n'
            #        f'\tβ            = {ideal_betas[0]}\n'
            #        f'\tmβ = -K·EI·β = {m1b}\n'
            #        f'\n'
            #        f'\tm0 = mt + mβ = {m10}\n')
            #
            # dprint(f'Закладаємо фіксовані моменти 2:\n'
            #        f'\tT            = {T}\n'
            #        f'\tGJ           = {GJ}\n'
            #        f'\tt            = {ideal_ts[1]}\n'
            #        f'\tmt = -T·GJ·t = {m2t}\n'
            #        f'\n'
            #        f'\tK            = {K}\n'
            #        f'\tEI           = {EI}\n'
            #        f'\tβ            = {ideal_betas[1]}\n'
            #        f'\tmβ = -K·EI·β = {m2b}\n'
            #        f'\n'
            #        f'\tm0 = mt + mβ = {m20}\n')

            m0 = Z * vector([0, 0, 1])
            f = P * vector([0, 0, 1])
            fp = point([0, 0, 0])

            yield mult, m0, f, fp

            # dprint(f'Має бути природний базис:\n'
            #        f'{vstack([ideal_ts[0], ideal_ns[0], ideal_betas[0]])}\n')
            #
            # dprint(f'Отримали\n'
            #        f'\tT = {current_system.guess[0].T}\n'
            #        f'\tK = {current_system.guess[0].K}\n'
            #        f'Має бути\n'
            #        f'\tT = {T}\n'
            #        f'\tK = {K}\n')

            dprint()
            got_end = current_system.guess[-1].point(current_system.guess[-1].s)
            with printoptions(precision=10, suppress=True):
                dprint(f'Обчислюємо похибку:\n'
                       f'\tостання точка: {got_end}\n'
                       f'\tмає бути:      {ideal_end}\n'
                       f'\tрізниця:       {got_end - ideal_end}\n'
                       f'\tнорма різниці: {norm(got_end - ideal_end)}\n')
            state['i'] += 1

    initial_system.set_task(task())

    # noinspection PyArgumentEqualDefault
    visual = Plotter(initial_system, pause=0.01, skip_plots=0,
                     plot_elements_segments=max(1, int(30 * each_length)), plot_basis_vectors_len=a / each_length / 3,
                     next_cnt=int(2.2 * elements_count), keep_lims=True,
                     lims=((-a * 1.1, a * 1.1), (-a * 1.1, a * 1.1), (-0.1, 2 * pi * winds * h * 1.1)),
                     major={'problem': 'ideal_helix',
                            'elements_count': elements_count},
                     minor={'a': a, 'h': h,
                            'EI': EI, 'GJ': GJ,
                            'winds': winds},
                     state=state)
    visual.show_interactive()


# noinspection Duplicates
def ideal_helix_1e():
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

    elements_count = 1
    winds = 5
    total_length = 2 * pi * winds * ah_sr

    each_length = total_length / elements_count

    phi_values = linspace(0, 2 * pi * winds, elements_count + 1)[:-1]
    start_s_values = [phi * ah_sr for phi in phi_values]

    start_t = vector([a, 0, h]) / ah_sr

    # P = (EI - GJ) * T * KT_sr
    # Z = -(EI * K ** 2 + GJ * T ** 2) / KT_sr
    # # P = (K * h * EI - T * a * GJ) / ah_sr
    # # Z = (-K * a * EI - T * h * GJ) / ah_sr

    P = -(EI - GJ) * T * KT_sr
    Z = (EI * K ** 2 + GJ * T ** 2) / KT_sr
    # P = -(K * h * EI - T * a * GJ) / ah_sr
    # Z = -(-K * a * EI - T * h * GJ) / ah_sr

    ideal_end = point([a * sin(2 * pi * winds), -a * cos(2 * pi * winds), 2 * pi * winds * h])

    xi = vector([0, 1, 0])
    angle = pi / 3
    xi = cos(angle) * xi + sin(angle) * cross(start_t, xi)

    elements = [Element(
        p=start_t * s - vector([0, a, 0]),
        s=each_length,
        t=start_t,
        xi=xi,
        K0=0, T0=0,
        EI=EI, GJ=GJ,
    ) for s in start_s_values]

    initial_system = System(elements, valid_threshold=each_length * 10, arm_position=0)

    state = dict()

    def task():
        mult = 1

        state['i'] = 1
        while True:
            current_system: System = yield
            if current_system is None:
                continue

            # dprint('\n' * 20)
            # dprint('Обчислюємо наступну ітерацію')

            # dprint(f'Закладаємо фіксовані моменти 1:\n'
            #        f'\tT            = {T}\n'
            #        f'\tGJ           = {GJ}\n'
            #        f'\tt            = {ideal_ts[0]}\n'
            #        f'\tmt = -T·GJ·t = {m1t}\n'
            #        f'\n'
            #        f'\tK            = {K}\n'
            #        f'\tEI           = {EI}\n'
            #        f'\tβ            = {ideal_betas[0]}\n'
            #        f'\tmβ = -K·EI·β = {m1b}\n'
            #        f'\n'
            #        f'\tm0 = mt + mβ = {m10}\n')
            #
            # dprint(f'Закладаємо фіксовані моменти 2:\n'
            #        f'\tT            = {T}\n'
            #        f'\tGJ           = {GJ}\n'
            #        f'\tt            = {ideal_ts[1]}\n'
            #        f'\tmt = -T·GJ·t = {m2t}\n'
            #        f'\n'
            #        f'\tK            = {K}\n'
            #        f'\tEI           = {EI}\n'
            #        f'\tβ            = {ideal_betas[1]}\n'
            #        f'\tmβ = -K·EI·β = {m2b}\n'
            #        f'\n'
            #        f'\tm0 = mt + mβ = {m20}\n')

            m0 = Z * vector([0, 0, 1])
            f = P * vector([0, 0, 1])
            fp = point([0, 0, 0])

            yield mult, m0, f, fp

            # dprint(f'Має бути природний базис:\n'
            #        f'{vstack([ideal_ts[0], ideal_ns[0], ideal_betas[0]])}\n')
            #
            # dprint(f'Отримали\n'
            #        f'\tT = {current_system.guess[0].T}\n'
            #        f'\tK = {current_system.guess[0].K}\n'
            #        f'Має бути\n'
            #        f'\tT = {T}\n'
            #        f'\tK = {K}\n')

            dprint()
            got_end = current_system.guess[-1].point(current_system.guess[-1].s)
            with printoptions(precision=10, suppress=True):
                dprint(f'Обчислюємо похибку:\n'
                       f'\tостання точка: {got_end}\n'
                       f'\tмає бути:      {ideal_end}\n'
                       f'\tрізниця:       {got_end - ideal_end}\n'
                       f'\tнорма різниці: {norm(got_end - ideal_end)}\n')
            state['i'] += 1

    initial_system.set_task(task())

    # noinspection PyArgumentEqualDefault
    visual = Plotter(initial_system, pause=0.01, skip_plots=0,
                     plot_elements_segments=max(1, int(30 * each_length)), plot_basis_vectors_len=a / each_length / 3,
                     next_cnt=int(2.2 * elements_count), keep_lims=True,
                     lims=((-a * 1.1, a * 1.1), (-a * 1.1, a * 1.1), (-0.1, 2 * pi * winds * h * 1.1)),
                     major={'problem': 'ideal_helix_1e',
                            'elements_count': elements_count},
                     minor={'a': a, 'h': h,
                            'EI': EI, 'GJ': GJ,
                            'winds': winds},
                     state=state)
    visual.show_interactive()


# noinspection Duplicates
# noinspection PyPep8Naming
# noinspection SpellCheckingInspection
def Bathe():
    # Last point = [17.55634512 59.03995987 42.13639299]
    #              [22.2        58.8        40.2]
    # Last point = [19.12206718 59.43061039 40.63625546]
    # Last point = [19.9351499  59.66925304 40.55072743]
    # Last point = [25.54253596 51.91710079 44.172872  ]
    # Last point = [22.13089535 52.31283358 45.62013139]

    # [16.89398632 58.88046202 42.5273269 ]

    elements_count = 10
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
        K0=1 / r,
        T0=0,
        EI=EI,
        GJ=GJ
    ) for s in linspace(0, angle, elements_count + 1)[:-1]]

    initial_system = System(elements, valid_threshold=each_length * 300)

    la = 7.2
    force = la * EI / r ** 2

    state = dict()

    def task():
        mult = 0.5

        state['i'] = 1
        while state['i'] <= 100:
            current_system = None
            while current_system is None:
                current_system = yield

            m0 = vector([0, 0, 0])
            f = force * vector([0, 0, 1])
            fp = current_system.elements[-1].point(current_system.elements[-1].s)
            with printoptions(precision=10, suppress=True):
                print(f'Ітерація {state["i"]}, остання точка = {fp}')

            yield mult, m0, f, fp
            state['i'] += 1

    initial_system.set_task(task())

    # noinspection PyArgumentEqualDefault
    visual = Plotter(initial_system, pause=0.01, skip_plots=0,
                     plot_elements_segments=30, plot_basis_vectors_len=0.5,
                     next_cnt=10,
                     major={'problem': 'Bathe',
                            'elements_count': elements_count},
                     minor={'angle': angle,
                            'r': r,
                            'EI': EI, 'GJ': GJ,
                            'la': la},
                     state=state)
    visual.show_interactive()


def upward_force():
    total_length = 10
    elements_count = 10
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
        GJ=GJ
    ) for s in start_s_values]

    initial_system = System(elements, valid_threshold=each_length * 300000)

    la = 1

    state = dict()

    def task():
        mult = 0.01

        state['percent'] = 10
        while state['percent'] <= 100:
            task_mult = state['percent'] / 100
            print(f'{task_mult:%} сили')

            state['i'] = 1
            while state['i'] <= 10:
                current_system = None
                while current_system is None:
                    current_system = yield

                m0 = vector([0, 0, 0])
                f = task_mult * la * vector([0, 0, 1])
                fp = current_system.elements[-1].point(current_system.elements[-1].s)
                print(f'Last point = {fp}')

                yield mult, m0, f, fp
                state['i'] += 1

            state['percent'] += 10

    initial_system.set_task(task())

    # noinspection PyArgumentEqualDefault
    visual = Plotter(initial_system, pause=0.01, skip_plots=0,
                     plot_elements_segments=30, plot_basis_vectors_len=0.5,
                     next_cnt=100,
                     major={'problem': 'upward_force',
                            'elements_count': elements_count},
                     minor={'total_length': total_length,
                            'EI': EI, 'GJ': GJ,
                            'la': la},
                     state=state)
    visual.show_interactive()


def main():
    # noinspection PyArgumentEqualDefault
    while True:
        test = input('''
    Select test:

1 - вертикальна сила до горизонтальної прямої балки
2 - вертикальна сила до балки-дуги кола в горизонтальній площині - задача Бате
3 - спіраль з прямої балки - ідеальний хелікс m0
4 - спіраль з прямої балки - ідеальний хелікс
5 - спіраль з прямої балки - ідеальний хелікс, 1 елемент
6 - спіраль з прямої балки - сила, прикладена до краю - задача Ібрахімбеговича
7 - вигинання прямої балки - спрощена задача Ібрахімбеговича
  - press Enter to exit

> ''')
        if not test:
            break

        if test == '1':
            upward_force()
        elif test == '2':
            Bathe()
        elif test == '3':
            ideal_helix_m0()
        elif test == '4':
            ideal_helix()
        elif test == '5':
            ideal_helix_1e()
        elif test == '6':
            ff = input('1 - 1 step\n2 - 10 steps\n3 - 100 steps\n - press Enter to go back\n> ')
            if ff == '1':
                Ibrahimbegovich_big(task_steps=1)
            elif ff == '2':
                Ibrahimbegovich_big(task_steps=10)
            elif ff == '3':
                Ibrahimbegovich_big(task_steps=100)
            return
        elif test == '7':
            ff = input('1 - no force\n2 - small force\n> ')
            if ff == '1':
                Ibrahimbegovich_small()
            elif ff == '2':
                Ibrahimbegovich_small(force=True)


if __name__ == '__main__':
    main()

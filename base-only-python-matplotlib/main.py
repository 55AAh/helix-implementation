from matplotlib import pyplot as plt
from numpy import array as vector, pi, sqrt, cos, sin, cross, clip, array, ceil, printoptions
from numpy import array as point
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

    def task():
        mult = 1

        title = f'\t\t\t\t\t\tElements: {elements_count}\n\n\t\tDisplacement components under end moment'
        if force:
            title += ' and pert. force'
        print(title + '\n')

        i = 0
        while True:
            current_system: System = yield
            if current_system is None:
                continue

            force_val = 0.001 if force else 0

            m0 = 2.5 * pi * vector([0, 0, 1])
            f = force_val * vector([0, 0, 1])
            fp = current_system.elements[-1].point(current_system.elements[-1].s)

            yield mult, m0, f, fp

            i += 1
            with printoptions(precision=10, suppress=True):
                _i_str = str(i) + ':'
                print(f'\tIteration {_i_str:<5}',
                      current_system.guess[-1].point(current_system.guess[-1].s) -
                      initial_system.elements[-1].point(initial_system.elements[-1].s))

    initial_system.set_task(task())

    # noinspection PyArgumentEqualDefault
    visual = Plotter(initial_system, pause=0.01)
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

    def task():
        pts_c = 11

        for mode in ['mt']:
            for task_mult in linspace(0, 1, task_steps + 1)[1:]:
                if task_steps < 100:
                    if task_mult < 0.6:
                        mult_clip = 0.003 / task_mult
                    else:
                        mult_clip = 0.1
                    max_diff_c = 20
                    max_diff_c_hist = list()
                    max_diff_c_hist_buf_size = 20
                    wdt = 5

                    if task_mult <= 0.1:
                        mult_clip = 0.2

                    while max_diff_c >= wdt:
                        current_system: System = yield
                        if current_system is None:
                            continue

                        current_pts = [e.point(e.s * i) for e in current_system.elements for i in linspace(0, 1, pts_c)]

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

                        yield mult_clip, m0, f, fp

                        c_begin = current_system.elements[0].point(0)
                        c_end = current_system.elements[-1].point(current_system.elements[-1].s)
                        displacements.append(c_end - c_begin)

                        final_guess_pts = [e.point(e.s * i) for e in current_system.final_guess for i in
                                           linspace(0, 1, pts_c)]

                        diff = [norm(p - fg_p) for p, fg_p in zip(current_pts, final_guess_pts)]
                        max_diff = array(diff).max(initial=0)
                        max_diff_c = max_diff / each_length * 1000

                        max_diff_c_hist.append(max_diff_c)
                        if len(max_diff_c_hist) > max_diff_c_hist_buf_size:
                            max_diff_c_hist.pop(0)

                        _mult = 10 / max_diff_c
                        mult = clip(_mult, 0.001, mult_clip)

                        _si = int(ceil(len(max_diff_c_hist) / 2))
                        max_diff_c_hist_c = array(max_diff_c_hist[:_si]).mean() / array(max_diff_c_hist[::-1][:_si]).mean()
                        print(f'task_mult = {float(task_mult):.3} mult_clip = {float(mult_clip):.5} '
                              f'hist_bs = {max_diff_c_hist_buf_size} '
                              f'max_diff_c = [[{float(max_diff_c):.5}]] max_diff_c_hist_c = {float(max_diff_c_hist_c): .5}')

                        if len(max_diff_c_hist) >= max_diff_c_hist_buf_size:
                            if max_diff_c_hist_c < 1.01:
                                max_diff_c_hist = list()
                                mult_clip /= 10
                                if mult_clip <= 1e-7:
                                    mult_clip = 0.01 / task_mult
                                    max_diff_c_hist_buf_size *= 3
                            tt = 1.001
                            if max_diff_c_hist_c > tt and mult_clip < 0.01 / task_mult:
                                max_diff_c_hist = list()
                                if task_mult < 0.6:
                                    mult_clip *= 0.6 / task_mult
                                else:
                                    mult_clip *= 1.3
                else:
                    mult = 1

                    mult_clip = 0.1
                    max_diff_c = 2
                    max_diff_c_hist = list()
                    max_diff_c_hist_buf_size = 100
                    while max_diff_c >= 1:
                        current_system: System = yield
                        if current_system is None:
                            continue

                        current_pts = [e.point(e.s * i) for e in current_system.elements for i in linspace(0, 1, pts_c)]

                        if mode == 'm':
                            m0 = task_mult * -200 * pi * vector([0, 0, 1])
                            f = 0 * 50 * vector([0, 0, 1])
                        elif mode == 't':
                            m0 = 1 * -200 * pi * vector([0, 0, 1])
                            f = task_mult * 50 * vector([0, 0, 1])
                        else:
                            m0 = task_mult * -200 * pi * vector([0, 0, 1])
                            f = task_mult * 50 * vector([0, 0, 1])
                        fp = current_system.elements[-1].point(current_system.elements[-1].s)

                        yield mult_clip, m0, f, fp

                        c_begin = current_system.elements[0].point(0)
                        c_end = current_system.elements[-1].point(current_system.elements[-1].s)
                        displacements.append(c_end - c_begin)

                        final_guess_pts = [e.point(e.s * i) for e in current_system.final_guess for i in
                                           linspace(0, 1, pts_c)]

                        diff = [norm(p - fg_p) for p, fg_p in zip(current_pts, final_guess_pts)]
                        max_diff = array(diff).max(initial=0)
                        max_diff_c = max_diff / each_length * 1000

                        max_diff_c_hist.append(max_diff_c)
                        if len(max_diff_c_hist) > max_diff_c_hist_buf_size:
                            max_diff_c_hist.pop(0)

                        _mult = 10 / max_diff_c
                        mult = clip(_mult, 0.001, mult_clip)

                        _si = int(ceil(len(max_diff_c_hist) / 2))
                        max_diff_c_hist_c = array(max_diff_c_hist[:_si]).mean() / array(
                            max_diff_c_hist[::-1][:_si]).mean()
                        print(f'task_mult = {task_mult} mult_clip = {mult_clip} hist_bs = {max_diff_c_hist_buf_size} '
                              f'max_diff_c = {max_diff_c:.5} max_diff_c_hist_c = {max_diff_c_hist_c: .5}')

                        if len(max_diff_c_hist) >= max_diff_c_hist_buf_size and max_diff_c_hist_c < 1.1:
                            max_diff_c_hist = list()
                            mult_clip /= 10
                            if mult_clip <= 1e-10:
                                mult_clip = 1
                                max_diff_c_hist_buf_size *= 10

                print('Bingo', flush=True)
                displacements_marks.append(len(displacements) - 1)
                displacements_s_marks.append(len(displacements_s))
                displacements_s.append(displacements[-1])

    initial_system.set_task(task())

    cnt = 100
    skip = 0
    if task_steps == 10:
        cnt = 5730
        skip = 9
    elif task_steps == 100:
        cnt = 10000
        skip = 99

    # noinspection PyArgumentEqualDefault
    visual = Plotter(initial_system, pause=0.01, skip_plots=skip, plot_elements_segments=max(1, int(30 * each_length)),
                     next_cnt=cnt, keep_lims=False,
                     lims=((-0.17443378249741953, 0.18093407976867604),
                           (-0.01892765241459829, 0.33644020985149725),
                           (-0.007159928998741932, 0.25936596770082976)))
    visual.show_interactive()

    if task_steps == 100:
        displacements_s = [d[2] for d in vector(displacements_s)]
        plt.plot(displacements_s, range(len(displacements_s)))
        for mark in displacements_s_marks[::10]:
            plt.axhline(mark, linestyle='--', alpha=0.5)
        plt.title('Free-end displacement component in the direction of applied force (only for task steps)')
        plt.show()

        displacements = [d[2] for d in vector(displacements)]
        plt.plot(displacements, range(len(displacements)))
        for mark in displacements_marks[::10]:
            plt.axhline(mark, linestyle='--', alpha=0.5)
        plt.title('Free-end displacement component in the direction of applied force')
        plt.show()


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

    def task():
        mult = 1

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

    initial_system.set_task(task())

    # noinspection PyArgumentEqualDefault
    visual = Plotter(initial_system, pause=0.01, skip_plots=0,
                     plot_elements_segments=max(1, int(30 * each_length)), plot_basis_vectors_len=a / each_length / 3,
                     next_cnt=elements_count, keep_lims=True,
                     lims=((-a * 1.1, a * 1.1), (-a * 1.1, a * 1.1), (-0.1, 2 * pi * winds * h * 1.1)))
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

    def task():
        mult = 1

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

    initial_system.set_task(task())

    # noinspection PyArgumentEqualDefault
    visual = Plotter(initial_system, pause=0.01, skip_plots=0,
                     plot_elements_segments=max(1, int(30 * each_length)), plot_basis_vectors_len=a / each_length / 3,
                     next_cnt=int(2.2 * elements_count), keep_lims=True,
                     lims=((-a * 1.1, a * 1.1), (-a * 1.1, a * 1.1), (-0.1, 2 * pi * winds * h * 1.1)))
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

    def task():
        mult = 1

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

    initial_system.set_task(task())

    # noinspection PyArgumentEqualDefault
    visual = Plotter(initial_system, pause=0.01, skip_plots=0,
                     plot_elements_segments=max(1, int(30 * each_length)), plot_basis_vectors_len=a / each_length / 3,
                     next_cnt=int(2.2 * elements_count), keep_lims=True,
                     lims=((-a * 1.1, a * 1.1), (-a * 1.1, a * 1.1), (-0.1, 2 * pi * winds * h * 1.1)))
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

    def task():
        mult = 0.5

        for i in range(101):
            current_system = None
            while current_system is None:
                current_system = yield

            m0 = vector([0, 0, 0])
            f = force * vector([0, 0, 1])
            fp = current_system.elements[-1].point(current_system.elements[-1].s)
            with printoptions(precision=10, suppress=True):
                print(f'Ітерація {i}, остання точка = {fp}')

            yield mult, m0, f, fp

    initial_system.set_task(task())

    # noinspection PyArgumentEqualDefault
    visual = Plotter(initial_system, pause=0.01, skip_plots=0,
                     plot_elements_segments=30, plot_basis_vectors_len=0.5,
                     next_cnt=10)
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

    def task():
        mult = 0.01

        for task_mult in linspace(0, 1, 11)[1:]:
            print(f'{task_mult:%} сили')
            for i in range(10):
                current_system = None
                while current_system is None:
                    current_system = yield

                m0 = vector([0, 0, 0])
                f = task_mult * la * vector([0, 0, 1])
                fp = current_system.elements[-1].point(current_system.elements[-1].s)
                print(f'Last point = {fp}')

                yield mult, m0, f, fp

    initial_system.set_task(task())

    # noinspection PyArgumentEqualDefault
    visual = Plotter(initial_system, pause=0.01, skip_plots=0,
                     plot_elements_segments=30, plot_basis_vectors_len=0.5,
                     next_cnt=100)
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
        elif test == '7':
            ff = input('1 - no force\n2 - small force\n> ')
            if ff == '1':
                Ibrahimbegovich_small()
            elif ff == '2':
                Ibrahimbegovich_small(force=True)


if __name__ == '__main__':
    main()

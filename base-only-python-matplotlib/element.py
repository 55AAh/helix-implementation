from typing import Tuple
from numpy.linalg import norm
from numpy import vstack, pi, cross, sqrt, sin, cos, arccos, dot
from numpy import array as point
from numpy import array as vector
from numpy import array as matrix
from dataclasses import dataclass
from numpy import clip


def dprint(*_args, **_kwargs):
    # print(*_args, **_kwargs)
    pass


@dataclass
class Element:
    p: point  # Початок (перша точка)
    s: float  # Довжина

    # Початкові кривизна та скрут
    K0: float
    T0: float

    # Набуті кривизна та скрут (з урахуванням початкових)
    K: float
    T: float

    mat: matrix  # Матеріальний базис
    nat: matrix  # Природний базис

    def __init__(self, p: point, s: float,
                 t: vector, xi: vector,
                 K0: float = 0.0, T0: float = 0.0,
                 EI: float = 1.0, GJ: float = 1.0):
        """ Створює елемент

        :param p: початок (перша точка)
        :param s: довжина
        :param t: вектор дотичної
        :param xi: вектор матеріальної нормалі
        :param K0: початкова кривизна
        :param T0: початковий скрут
        """

        self.p = p
        self.s = s
        self.K0, self.T0 = K0, T0
        self.EI, self.GJ = EI, GJ

        # Нормуємо вектори базису
        t, xi = (t / norm(t), xi / norm(xi))
        # Знаходимо третій вектор (бінормаль) матеріального базису за відомими двома (дотична та нормаль)
        eta = cross(t, xi)
        # Маємо матеріальний базис
        self.mat = vstack((t, xi, eta))

        # Вважаємо, що спочатку елемент не має додаткової кривизни та скруту
        self.K, self.T = K0, T0
        # А природний та матеріальний базиси співпадають
        self.nat = self.mat

    def point(self, s: float) -> point:
        """ Обчислює положення будь-якої точки елементу

        :param s: параметр довжини
        :return: положення точки
        :rtype: point
        """

        kt_s = self.K ** 2 + self.T ** 2

        # Знаходимо вектор переміщення
        if kt_s == 0:
            # Якщо елемент - пряма лінія, маємо особливий випадок
            t = self.mat[0]
            offset = t * s

        else:
            phi = sqrt(kt_s) * s
            a, h = self.K / kt_s, self.T / kt_s

            m = vector([
                (a ** 2 * sin(phi) + h ** 2 * phi) / sqrt(a ** 2 + h ** 2),
                a * (1 - cos(phi)),
                a * h * (phi - sin(phi)) / sqrt(a ** 2 + h ** 2)
            ])

            # offset = m @ self.nat

            t, n, beta = self.nat
            offset = m[0] * t + m[1] * n + m[2] * beta

        position = self.p + offset

        return position

    def bases(self, s: float) -> Tuple[matrix, matrix]:
        """ Обчислює матеріальний та природний базиси в будь-якій точці елементу

        :param s: параметр довжини
        :return: матеріальний та природний базиси
        :rtype: Tuple[matrix, matrix]
        """

        begin_t, begin_xi, begin_eta = self.mat.copy()
        _, begin_n, begin_beta = self.nat.copy()

        kt_s = self.K ** 2 + self.T ** 2

        if kt_s == 0:
            # Якщо елемент - пряма лінія, маємо особливий випадок
            # end_nat = self.nat
            end_t = begin_t
            end_n = begin_n
            end_beta = begin_beta

        else:
            phi = sqrt(kt_s) * s

            a = self.K / kt_s
            h = self.T / kt_s

            m = matrix([
                vector([(a ** 2 * cos(phi) + h ** 2) / (a ** 2 + h ** 2),
                        a * sin(phi) / sqrt(a ** 2 + h ** 2),
                        a * h * (1 - cos(phi)) / (a ** 2 + h ** 2)]),

                vector([-a * sin(phi) / sqrt(a ** 2 + h ** 2),
                        cos(phi),
                        h * sin(phi) / sqrt(a ** 2 + h ** 2)]),

                vector([a * h * (1 - cos(phi)) / (a ** 2 + h ** 2),
                        -h * sin(phi) / sqrt(a ** 2 + h ** 2),
                        (a ** 2 + h ** 2 * cos(phi)) / (a ** 2 + h ** 2)]),
            ])

            # end_nat = m @ self.nat

            end_t = m[0][0] * begin_t + m[0][1] * begin_n + m[0][2] * begin_beta
            end_n = m[1][0] * begin_t + m[1][1] * begin_n + m[1][2] * begin_beta
            end_beta = m[2][0] * begin_t + m[2][1] * begin_n + m[2][2] * begin_beta

        begin_t, begin_xi, begin_eta = self.mat.copy()

        # Кут між матеріальним та природним базисами
        psi = arccos(clip(dot(begin_xi, begin_n), -1, 1))
        if arccos(clip(dot(begin_xi, begin_beta), -1, 1)) < pi / 2:
            psi *= -1

        end_xi = end_n * cos(psi) - end_beta * sin(psi)
        end_eta = end_n * sin(psi) + end_beta * cos(psi)

        end_mat = vstack((end_t, end_xi, end_eta))
        end_nat = vstack((end_t, end_n, end_beta))

        return end_mat, end_nat

    def apply_moment(self, m: vector) -> 'Element':
        """ Прикладає момент до елементу

        :param m: вектор моменту
        :return: новий елемент
        :rtype: Element
        """

        # # Розкладемо момент на складові за матеріальним базисом
        # proj = m @ self.mat.T
        #
        # # Моменти треба поділити на фізичні константи
        # m_t = proj[0] / self.GJ
        # m_xi = proj[1] / self.EI
        # m_eta = proj[2] / self.EI

        dprint(f'Дали момент:\n'
               f'\tm = {m}\n')

        t, xi, eta = self.mat
        dprint(f'Маємо матеріальні базиси:\n'
               f'\tt = {t}\n'
               f'\tξ = {xi}\n'
               f'\tη = {eta}\n')

        m_t = dot(m, t) / self.GJ
        m_xi = dot(m, xi) / self.EI
        m_eta = dot(m, eta) / self.EI
        dprint(f'Розкладаємо за базисами:\n'
               f'\tm_t = {m_t}\n'
               f'\tm_ξ = {m_xi}\n'
               f'\tm_η = {m_eta}\n')
        dprint(f'\tK0  = {self.K0}\n'
               f'\tT0  = {self.T0}\n')

        # T = self.T0 - m_t
        # _beta = -m_xi * xi + (self.K0 - m_eta) * eta
        # dprint(f'Маємо\n'
        #        f'\t-m_ξ·ξ + (K0 - m_η)·η = {_beta}\n')
        # TODO мінуси? В мене M = arm x F
        T = self.T0 + m_t
        _beta = m_xi * xi + (self.K0 + m_eta) * eta
        dprint(f'Маємо\n'
               f'\tm_ξ·ξ + (K0 + m_η)·η = {_beta}\n')

        K = norm(_beta)

        dprint(f'Отримали кривизну та скрут:\n'
               # f'\tT = -m_t                 = {T}\n'
               f'\tT = m_t                 = {T}\n'
               f'\tK = norm(_beta) = {norm(_beta)}\n'
               f'\tK = sqrt(m_ξ^2 + m_η^2) = {K}\n')

        if K != 0:  # TODO Його можна визначити лише в цьому випадку
            beta = _beta / K
        else:
            beta = eta

        dprint(f'Маємо\n'
               f'\tβ               = {beta}\n')

        n = cross(beta, t)
        dprint(f'Знаходимо n як векторний добуток:\n'
               f'\tn = β x t = {n}\n')

        nat = vstack((t, n, beta))
        dprint(f'Отже, новий природний базис:\n'
               f'{nat}\n')

        guess = Element(self.p.copy(), self.s, t.copy(), xi.copy(), self.K0, self.T0, self.EI, self.GJ)
        guess.K = K
        guess.T = T
        guess.nat = nat.copy()

        return guess

    def serialize(self) -> dict:
        d = {
            'p': self.p.tolist(), 's': self.s,
            'K0': self.K0, 'T0': self.T0,
            'K': self.K, 'T': self.T,
            'mat': self.mat.tolist(), 'nat': self.nat.tolist(),
            'EI': self.EI, 'GJ': self.GJ,
        }

        return d

    @staticmethod
    def deserialize(d: dict):
        p, s = point(d['p']), d['s']
        K0, T0 = d['K0'], d['T0']
        K, T = d['K'], d['T']
        mat, nat = matrix(d['mat']), matrix(d['nat'])
        EI, GJ = d['EI'], d['GJ']

        e = Element(p, s, mat[0], mat[1], K0, T0, EI, GJ)
        e.K, e.T = K, T
        e.mat, e.nat = mat, nat

        return e

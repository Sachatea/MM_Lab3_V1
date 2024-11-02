import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import *

class Solver:
    def __init__(self):  # константи
        # Область S0
        self.height = 1          # висота прямокутника x1  Додатні
        self.width = 1           # ширира прямокутника x2
        self.lbr = [0,0]         # лівий нижній кут прямокутника
        
        self.T = 2               # проміжок часу [0,T]
        self.t_inst = 0          # вибраний час для показу результату

        self.k = 1               # з умови для G

        # кількість початково-граничних операторі (доступних)
        self.Ln1 = 1             
        self.Ln2 = 1

        #приклад для Set_L_0_r: [[x1,x2,t,op_num],[1,1,0,1]]  точки з S0
        self.Set_L_0_r = [[0.5,0.5,0,1], [0.4,0.6,0,1], [0.6,0.4,0,1], [0.1,0.1,0,1]]    
        #приклад для Set_L_G_r: [[x1,x2,t,op_num],[1,1,1,1]]  точки з SG де SG це прямокутник з 2x сторонами навколо S0, без S0
        self.Set_L_G_r = [[1.5,1.5,1,1], [-1.5,1.5,0.6,1]]

        self.n = 3               # щільність сітки по якій ми інтегруємо

        self.t, self.x1, self.x2 = symbols('t x1 x2')
        self.t_sh, self.x1_sh, self.x2_sh = symbols('t_sh x1_sh x2_sh')

        self.v = [0, 0]          # вектор функція

        self.Eps_s = 0
        self.Det_s = "?"



    def y_primary(self):         # шукана функція
        return self.x1 * self.x1 + self.x2 * self.x2 + self.t
    
    def L(self, func):           # оператор з умови
        return func.diff(self.t) + self.k * (func.diff(self.x1).diff(self.x1) + func.diff(self.x2).diff(self.x2))
    
    def G_sh(self):
        return (1 / ((4 * math.pi * self.k * (self.t - self.t_sh)) ** 0.5)) * exp(-(((self.x1 - self.x1_sh) ** 2 + (self.x2 - self.x2_sh) **2) ** 0.5) / (4 * self.k * (self.t - self.t_sh)))

    def L_0_r(self, func, i):  # початкові оператори
        if i == 1:
            return func
        elif i == 2:
            return func.diff(self.t)
        elif i == 3:
            return func.diff(self.x1)
    
    def L_G_r(self, func, i):  # граничні оператори
        if i == 1:
            return func
        elif i == 2:
            return func.diff(self.t)
        elif i == 3:
            return func.diff(self.x1)
    
    def shov_y_primary(self):
        x1 = np.linspace( self.lbr[0], self.lbr[0] + self.height, 100)
        x2 = np.linspace( self.lbr[1], self.lbr[1] + self.width, 100)
        x1, x2 = np.meshgrid(x1, x2)
        z = np.array([[Solver.y_primary(self).subs({self.x1: x1, self.x2: x2, self.t: self.t_inst}).evalf()
                    for x1, x2 in zip(row_x1, row_x2)] 
                    for row_x1, row_x2 in zip(x1, x2)])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x1, x2, z, cmap='viridis')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Z')
        #plt.show()
        return ax
    
    def Net_S0T(self):
        x1_min, x2_min, t_min = self.lbr[0], self.lbr[1], 0
        x1_max = x1_min + self.height
        x2_max = x2_min + self.width
        t_max = self.T
        x1_vals = np.linspace(x1_min, x1_max, self.n + 1)
        x2_vals = np.linspace(x2_min, x2_max, self.n + 1)
        t_vals = np.linspace(t_min, t_max, self.n + 1)
        points = np.array([[x1, x2, t] for x2 in x2_vals for x1 in x1_vals for t in t_vals if t != 0])
        return points

    def Net_S0(self):
        x1_min, x2_min, t_min = self.lbr[0], self.lbr[1], -self.T
        x1_max = x1_min + self.height
        x2_max = x2_min + self.width
        t_max = 0
        x1_vals = np.linspace(x1_min, x1_max, self.n + 1)
        x2_vals = np.linspace(x2_min, x2_max, self.n + 1)
        t_vals = np.linspace(t_min, t_max, self.n + 1)
        points = np.array([[x1, x2, t] for x2 in x2_vals for x1 in x1_vals for t in t_vals if t != 0])
        return points
    
    def Net_SG(self):
        x1_min, x2_min, t_min = self.lbr[0] - self.height / 2, self.lbr[1] - self.width / 2, 0
        x1_max = x1_min + self.height * 2
        x2_max = x2_min + self.width * 2
        t_max = self.T
        x1_vals = np.linspace(x1_min, x1_max, self.n + 3)
        x2_vals = np.linspace(x2_min, x2_max, self.n + 3)
        t_vals = np.linspace(t_min, t_max, self.n + 1)
        points = np.array([[x1, x2, t] for x2 in x2_vals for x1 in x1_vals for t in t_vals
                           if (x1 < self.lbr[0] or x1 > self.lbr[0] + self.height) and (x2 < self.lbr[1] or x2 > self.lbr[1] + self.width) and t != 0])
        return points

    def Aij_Y(self):
        A11 = []
        A12 = []
        A21 = []
        A22 = []
        Y = []
        Net_S0T = Solver.Net_S0T(self)
        u = Solver.L(self, Solver.y_primary(self))
        V_delta_inf = (self.T / self.n) * (self.height / (self.n + 1)) * (self.width / (self.n + 1))
        sum_y_inf = 0
        SYI = []
        for i in Net_S0T:
            sum_y_inf = Solver.G_sh(self).subs([(self.x1_sh, i[0]), (self.x2_sh, i[1]), (self.t_sh, i[2])]) * u.subs([(self.x1_sh, i[0]), (self.x2_sh, i[1]), (self.t_sh, i[2])]) * V_delta_inf 
            SYI.append([sum_y_inf, -i[2]])
        res = 0
        for i in self.Set_L_0_r:
            A11.append([Solver.L_0_r(self, Solver.G_sh(self), i[3]).subs([(self.x1, i[0]), (self.x2, i[1]), (self.t, i[2])]), i[2]])
            A12.append([Solver.L_0_r(self, Solver.G_sh(self), i[3]).subs([(self.x1, i[0]), (self.x2, i[1]), (self.t, i[2])]), i[2]])
            for j in SYI:
                if (i[2] + j[1]) > 0:
                    res += Solver.L_0_r(self, j[0], i[3]).subs([(self.x1, i[0]), (self.x2, i[1]), (self.t, i[2])]).evalf()
            Y.append(Solver.L_0_r(self, Solver.y_primary(self), i[3]).subs([(self.x1, i[0]), (self.x2, i[1]), (self.t, i[2])]).evalf() - res)
            res = 0
        for i in self.Set_L_G_r:
            A21.append([Solver.L_G_r(self, Solver.G_sh(self), i[3]).subs([(self.x1, i[0]), (self.x2, i[1]), (self.t, i[2])]), i[2]])
            A22.append([Solver.L_G_r(self, Solver.G_sh(self), i[3]).subs([(self.x1, i[0]), (self.x2, i[1]), (self.t, i[2])]), i[2]])
            for j in SYI:
                if (i[2] + j[1]) > 0:
                    res += Solver.L_G_r(self, j[0], i[3]).subs([(self.x1, i[0]), (self.x2, i[1]), (self.t, i[2])]).evalf()
            Y.append(Solver.L_G_r(self, Solver.y_primary(self), i[3]).subs([(self.x1, i[0]), (self.x2, i[1]), (self.t, i[2])]).evalf() - res)
            res = 0
        A11 = np.array(A11)
        A12 = np.array(A12)
        A21 = np.array(A21)
        A22 = np.array(A22)
        Y = np.array(Y)
        return A11, A12, A21, A22, Y
    
    def P(self, A11, A12, A21, A22):
        Net_S0 = Solver.Net_S0(self)
        Net_SG = Solver.Net_SG(self)     # self.n + 2 густіше точки в площині x1*x2 для Net_SG
        V_delta_S0 =  (self.height / (self.n + 1)) * (self.width / (self.n + 1)) * (self.T / self.n) 
        V_delta_SG =  ((self.height * 2) / (self.n + 3)) * ((self.width * 2) / (self.n + 3)) * (self.T / self.n)
        
        P_res = []
        row = []
        res = 0

        ########## P11  можна винести pij в окремий метод, але лінь

        for i in A11:
            row = []
            for j in A11:
                for k in Net_S0:
                    if (i[1] - k[2]) > 0 and (j[1] - k[2]) > 0:
                        res += (i[0] * j [0]).subs([(self.x1_sh, k[0]), (self.x2_sh, k[1]), (self.t_sh, k[2])]) *  V_delta_S0
                row.append(res)
                res = 0
            P_res.append(row)
        P11 = np.array(P_res)
        P_res = []

        for i in A12:
            row = []
            for j in A12:
                for k in Net_SG:
                    if (i[1] - k[2]) > 0 and (j[1] - k[2]) > 0:
                        res += (i[0] * j [0]).subs([(self.x1_sh, k[0]), (self.x2_sh, k[1]), (self.t_sh, k[2])]) *  V_delta_SG
                row.append(res)
                res = 0
            P_res.append(row)
        P_res = np.array(P_res)
        P11 = P11 + P_res
        P_res = []

        ############## P12

        for i in A11:
            row = []
            for j in A21:
                for k in Net_S0:
                    if (i[1] - k[2]) > 0 and (j[1] - k[2]) > 0:
                        res += (i[0] * j [0]).subs([(self.x1_sh, k[0]), (self.x2_sh, k[1]), (self.t_sh, k[2])]) *  V_delta_S0
                row.append(res)
                res = 0
            P_res.append(row)
        P12 = np.array(P_res)
        P_res = []

        for i in A12:
            row = []
            for j in A22:
                for k in Net_SG:
                    if (i[1] - k[2]) > 0 and (j[1] - k[2]) > 0:
                        res += (i[0] * j [0]).subs([(self.x1_sh, k[0]), (self.x2_sh, k[1]), (self.t_sh, k[2])]) *  V_delta_SG
                row.append(res)
                res = 0
            P_res.append(row)
        P_res = np.array(P_res)
        P12 = P12 + P_res
        P_res = []

        ############## P21
        
        for i in A21:
            row = []
            for j in A11:
                for k in Net_S0:
                    if (i[1] - k[2]) > 0 and (j[1] - k[2]) > 0:
                        res += (i[0] * j [0]).subs([(self.x1_sh, k[0]), (self.x2_sh, k[1]), (self.t_sh, k[2])]) *  V_delta_S0
                row.append(res)
                res = 0
            P_res.append(row)
        P21 = np.array(P_res)
        P_res = []

        for i in A22:
            row = []
            for j in A12:
                for k in Net_SG:
                    if (i[1] - k[2]) > 0 and (j[1] - k[2]) > 0:
                        res += (i[0] * j [0]).subs([(self.x1_sh, k[0]), (self.x2_sh, k[1]), (self.t_sh, k[2])]) *  V_delta_SG
                row.append(res)
                res = 0
            P_res.append(row)
        P_res = np.array(P_res)
        P21 = P21 + P_res
        P_res = []

        ############## P22
        
        for i in A21:
            row = []
            for j in A21:
                for k in Net_S0:
                    if (i[1] - k[2]) > 0 and (j[1] - k[2]) > 0:
                        res += (i[0] * j [0]).subs([(self.x1_sh, k[0]), (self.x2_sh, k[1]), (self.t_sh, k[2])]) *  V_delta_S0
                row.append(res)
                res = 0
            P_res.append(row)
        P22 = np.array(P_res)
        P_res = []

        for i in A22:
            row = []
            for j in A22:
                for k in Net_SG:
                    if (i[1] - k[2]) > 0 and (j[1] - k[2]) > 0:
                        res += (i[0] * j [0]).subs([(self.x1_sh, k[0]), (self.x2_sh, k[1]), (self.t_sh, k[2])]) *  V_delta_SG
                row.append(res)
                res = 0
            P_res.append(row)
        P_res = np.array(P_res)
        P22 = P22 + P_res

        return P11, P12, P21, P22

    def Av(self, A11, A12, A21, A22):
        Net_S0 = Solver.Net_S0(self)
        Net_SG = Solver.Net_SG(self)     # self.n + 2 густіше точки в площині x1*x2 для Net_SG
        V_delta_S0 =  (self.height / (self.n + 1)) * (self.width / (self.n + 1)) * (self.T / self.n) 
        V_delta_SG =  ((self.height * 2) / (self.n + 3)) * ((self.width * 2) / (self.n + 3)) * (self.T / self.n)
        
        A_res = []
        res = 0

        ########## Av0

        for i in A11:
            for k in Net_S0:
                if (i[1] - k[2]) > 0:
                    res += (i[0] * self.v[0]).subs([(self.x1_sh, k[0]), (self.x2_sh, k[1]), (self.t_sh, k[2])]) *  V_delta_S0
            A_res.append(res)
            res = 0
        Av0 = np.array(A_res)
        A_res = []

        for i in A12:
            for k in Net_SG:
                if (i[1] - k[2]) > 0:
                    res += (i[0] * self.v[1]).subs([(self.x1_sh, k[0]), (self.x2_sh, k[1]), (self.t_sh, k[2])]) *  V_delta_SG
            A_res.append(res)
            res = 0
        A_res = np.array(A_res)
        Av0 = Av0 + A_res
        A_res = []

        ############## AvG

        for i in A21:
            for k in Net_S0:
                if (i[1] - k[2]) > 0:
                    res += (i[0] * self.v[0]).subs([(self.x1_sh, k[0]), (self.x2_sh, k[1]), (self.t_sh, k[2])]) *  V_delta_S0
            A_res.append(res)
            res = 0
        AvG = np.array(A_res)
        A_res = []

        for i in A22:
            for k in Net_SG:
                if (i[1] - k[2]) > 0:
                    res += (i[0] * self.v[1]).subs([(self.x1_sh, k[0]), (self.x2_sh, k[1]), (self.t_sh, k[2])]) *  V_delta_SG
            A_res.append(res)
            res = 0
        A_res = np.array(A_res)
        AvG = AvG + A_res
        Av = np.concatenate((Av0, AvG))
        return Av

    def shov_y_found(self, u0, uG):
        Net_S0T = Solver.Net_S0T(self)
        Net_S0 = Solver.Net_S0(self)
        Net_SG = Solver.Net_SG(self)     # self.n + 2 густіше точки в площині x1*x2 для Net_SG
        V_delta_S0 =  (self.height / (self.n + 1)) * (self.width / (self.n + 1)) * (self.T / self.n) 
        V_delta_SG =  ((self.height * 2) / (self.n + 3)) * ((self.width * 2) / (self.n + 3)) * (self.T / self.n)
        V_delta_inf = (self.T / self.n) * (self.height / (self.n + 1)) * (self.width / (self.n + 1))

        u = Solver.L(self, Solver.y_primary(self))

        SYI = []
        for i in Net_S0T:
            sum_y_inf = Solver.G_sh(self).subs([(self.x1_sh, i[0]), (self.x2_sh, i[1]), (self.t_sh, i[2])]) * u.subs([(self.x1_sh, i[0]), (self.x2_sh, i[1]), (self.t_sh, i[2])]) * V_delta_inf 
            SYI.append([sum_y_inf, -i[2]])

        SY0 = []
        for i in Net_S0:
            sum_y_inf = Solver.G_sh(self).subs([(self.x1_sh, i[0]), (self.x2_sh, i[1]), (self.t_sh, i[2])]) * u0.subs([(self.x1_sh, i[0]), (self.x2_sh, i[1]), (self.t_sh, i[2])]) * V_delta_S0 
            SY0.append([sum_y_inf, -i[2]])

        SYG = []
        for i in Net_SG:
            sum_y_inf = Solver.G_sh(self).subs([(self.x1_sh, i[0]), (self.x2_sh, i[1]), (self.t_sh, i[2])]) * uG.subs([(self.x1_sh, i[0]), (self.x2_sh, i[1]), (self.t_sh, i[2])]) * V_delta_SG 
            SYG.append([sum_y_inf, -i[2]])

        def y(self, x11, x22, t33):
            y = 0
            for j in SYI:
                if (t33 + j[1]) > 0:
                    y += j[0].subs([(self.x1, x11), (self.x2, x22), (self.t, t33)]).evalf()
            for j in SY0:
                if (t33 + j[1]) > 0:
                    y += j[0].subs([(self.x1, x11), (self.x2, x22), (self.t, t33)]).evalf()
            for j in SYG:
                if (t33 + j[1]) > 0:
                    y += j[0].subs([(self.x1, x11), (self.x2, x22), (self.t, t33)]).evalf()
            return float(y)

        x1 = np.linspace( self.lbr[0], self.lbr[0] + self.height, 10)
        x2 = np.linspace( self.lbr[1], self.lbr[1] + self.width, 10)
        x1, x2 = np.meshgrid(x1, x2)

        z = np.array([[y(self, x1, x2, self.t_inst)
                    for x1, x2 in zip(row_x1, row_x2)] 
                    for row_x1, row_x2 in zip(x1, x2)])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x1, x2, z, cmap='viridis')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Z')

        # Define your title data
        precision = self.Eps_s
        uniqueness = self.Det_s
        title_str = f"Data Visualization\nPrecision: {precision}\nUniqueness: {uniqueness}"

        # Set the title
        plt.title(title_str)


        plt.show()

    def pidstava(self, A11, A12, A21, A22, s):
        A110 = []
        for i in A11:
            if (i[1] - s[2]) > 0:
                A110.append(i[0].subs([(self.x1_sh, s[0]), (self.x2_sh, s[1]), (self.t_sh, s[2])]).evalf())
            else:
                A110.append(0)
        A120 = []
        for i in A12:
            if (i[1] - s[2]) > 0:
                A120.append(i[0].subs([(self.x1_sh, s[0]), (self.x2_sh, s[1]), (self.t_sh, s[2])]).evalf())
            else:
                A120.append(0)
        A210 = []
        for i in A21:
            if (i[1] - s[2]) > 0:
                A210.append(i[0].subs([(self.x1_sh, s[0]), (self.x2_sh, s[1]), (self.t_sh, s[2])]).evalf())
            else:
                A210.append(0)
        A220 = []
        for i in A22:
            if (i[1] - s[2]) > 0:
                A220.append(i[0].subs([(self.x1_sh, s[0]), (self.x2_sh, s[1]), (self.t_sh, s[2])]).evalf())
            else:
                A220.append(0)
        
        A = np.zeros((len(A11) + len(A21), 2))
        for i in range(0, len(A11)):
            A[i][0] = A110[i]
            A[i][1] = A120[i]

        for i in range(len(A11), len(A21)):
            A[i][0] = A210[i]
            A[i][1] = A220[i]

        return A
        
    def det(self, A11, A12, A21, A22):
        elements = [(0.5, 0.5, 0), (0.4, 0.6, 0), (0.6, 0.4, 0), (0.1, 0.1, 0)]
        count = len(elements)
        matrix_zeros = np.zeros((2 * count, 2 * count))

        Solver.pidstava(self, A11, A12, A21, A22, (0.5, 0.5, 0))

        for i in range(0, count):
            for j in range(0, count):
                Aij = Solver.pidstava(self, A11, A12, A21, A22, elements[i]).T @ Solver.pidstava(self, A11, A12, A21, A22, elements[j])
                for a in range(0, 2):
                    for b in range(0, 2):
                        matrix_zeros[a+2*(i-1),b+2*(j-1)] = Aij[a,b]
     
        if np.linalg.det(matrix_zeros) > 0.01:
            return "єдиний"
        else:
            return  "неоднозначний"

    def main(self):
        A11, A12, A21, A22, Y = Solver.Aij_Y(self)
        P11, P12, P21, P22 = Solver.P(self, A11, A12, A21, A22)

        top = np.hstack((P11, P12))
        bottom = np.hstack((P21, P22))
        P = np.vstack((top, bottom))
        P = P.astype(np.float64)
        P_plus = np.linalg.pinv(P)

        Av = Solver.Av(self, A11, A12, A21, A22)

        A0 = []
        for i in A11:
            A0.append(i[0])
        for i in A21:
            A0.append(i[0])
        A0 = np.array(A0)
    
        AG = []
        for i in A12:
            AG.append(i[0])
        for i in A22:
            AG.append(i[0])
        AG = np.array(AG)

        u0 = A0 @ P_plus @ (Y - Av) + self.v[0]
        uG = AG @ P_plus @ (Y - Av) + self.v[1]

        self.Eps_s = Y @ Y.T - Y @ P @ P_plus @ Y.T
        print("Точність - ", self.Eps_s)

        self.Det_s = Solver.det(self, A11, A12, A21, A22)
        print("єдиність - ", self.Det_s)

        Solver.shov_y_found(self, u0, uG)

        
               
#s = Solver()
#s.main()
#s.shov_y_primary()

import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import sys
import csv
import pathlib
from example_tester import extract_dataset_traj_active
# sys.path.append(str(pathlib.Path(__file__).parent.parent))
sys.path.append('C:/Users/skart/PycharmProjects/interaction-dataset/data')

from QuinticPolynomialsPlanner import \
    QuinticPolynomial
import cubic_spline_planner

SIM_LOOP = 100

# Parameter
#MAX_SPEED = 50.0 / 3.6  # maximum speed [m/s]
MAX_SPEED = 50.0
MAX_ACCEL = 2.0  # maximum acceleration [m/ss]
#MAX_CURVATURE = 1.0  # maximum curvature [1/m]
MAX_CURVATURE = 5.0
MAX_ROAD_WIDTH = 7.0  # maximum road width [m]
D_ROAD_W = 1.0  # road width sampling length [m]
DT = 0.2  # time tick [s]
MAX_T = 5.0  # max prediction time [m]
MIN_T = 4.0  # min prediction time [m]
TARGET_SPEED = 30.0 / 3.6  # target speed [m/s]
D_T_S = 5.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 1  # sampling number of target speed
ROBOT_RADIUS = 2.0  # robot radius [m]

# cost weights
K_J = 0.1
K_T = 0.1
K_D = 1.0
K_LAT = 1.0
K_LON = 1.0

show_animation = True


class QuarticPolynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt


class FrenetPath:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []


def calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0):
    frenet_paths = []

    # generate path to each offset goal
    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):

        # Lateral motion planning
        for Ti in np.arange(MIN_T, MAX_T, DT):
            fp = FrenetPath()

            # lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            # Longitudinal motion planning (Velocity keeping)
            for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE,
                                TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = copy.deepcopy(fp)
                lon_qp = QuarticPolynomial(s0, c_speed, c_accel, tv, 0.0, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                # square of diff from target speed
                ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2

                tfp.cd = K_J * Jp + K_T * Ti + K_D * tfp.d[-1] ** 2
                tfp.cv = K_J * Js + K_T * Ti + K_D * ds
                tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv

                frenet_paths.append(tfp)

    return frenet_paths


def calc_global_paths(fplist, csp):
    for fp in fplist:

        # calc global positions
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            i_yaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
            fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)

        # calc yaw and ds
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.hypot(dx, dy))

        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

    return fplist


def check_collision(fp, ob):
    for i in range(len(ob[:, 0])):
        d = [((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2)
             for (ix, iy) in zip(fp.x, fp.y)]

        collision = any([di <= ROBOT_RADIUS ** 2 for di in d])

        if collision:
            return False

    return True


def check_paths(fplist, ob):
    ok_ind = []
    for i, _ in enumerate(fplist):
        if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
            continue
        elif any([abs(a) > MAX_ACCEL for a in
                  fplist[i].s_dd]):  # Max accel check
            continue
        elif any([abs(c) > MAX_CURVATURE for c in
                  fplist[i].c]):  # Max curvature check
            continue
        elif not check_collision(fplist[i], ob):
            continue

        ok_ind.append(i)

    return [fplist[i] for i in ok_ind]


def frenet_optimal_planning(csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, ob):
    fplist = calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0)

    fplist = calc_global_paths(fplist, csp)

    fplist = check_paths(fplist, ob)
    #print(fplist)

    # find minimum cost path
    min_cost = float("inf")
    #best_path = None
    #print(fplist)
    best_path = fplist[0]
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp

    return best_path


def generate_target_course(x, y):
    csp = cubic_spline_planner.CubicSpline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp
def points_in_circle_np(radius, x0=0.0, y0=0.0):   # get a list of obstacles - all the points in the circular intersection

    final = []
    x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
    y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
    x, y = np.where((x_[:, np.newaxis] - x0) ** 2 + (y_ - y0) ** 2 <= radius ** 2)

    for x, y in zip(x_[x], y_[y]):
        tt = []
        tt.append(x)
        tt.append(y)
        #print(tt)
        final.append(tt)
    return final

def calc_ttc(first_agent,second_agent):
    active_x, active_y, active_vx, active_vy, active_psi, active_int, active_b, active_e = extract_dataset_traj_active("Scenario4", False, [6], [3], data_lim=100, track_id_list=[first_agent])  # [37,47,77]
    active_2x, active_2y, active_2vx, active_2vy, active_2psi, active_2int, active_2b, active_2e = extract_dataset_traj_active("Scenario4", False, [4], [6], data_lim=100, track_id_list=[second_agent])  # [41,46,60]

    base_timestamp1 = 0
    base_timestamp2 = 0
    if active_b[0]<= active_2b[0]:
        ptime_start= active_b[0]
        active_start = active_2b[0]
        base_timestamp1 = ptime_start
        base_timestamp2 = active_start
        #ptime_third= active_b[0]+ int(((active_e[0]- active_b[0])%3)*200)
        ptime_third = active_b[0]+ int(((103/141)*(active_e[0]- active_b[0]))/100)*100
    else:
        ptime_start = active_2b[0]
        active_start = active_b[0]
        base_timestamp1 = active_start
        base_timestamp2 = ptime_start
        #ptime_third= active_b[0]+ int(((active_e[0]- active_b[0])%3)*100)
        ptime_third = active_b[0]+ int(((103/141)*(active_e[0]- active_b[0]))/100)*100

    if active_e[0] >= active_2e[0]:
        ptime_end = active_e[0]
        active_end = active_2e[0]
    else:
        ptime_end = active_2e[0]
        active_end = active_e[0]
    #print((103/141)*(active_e[0]- active_b[0]))
    print((160 / 215) * (active_e[0] - active_b[0]))
    #print((109 / 145) * (active_e[0] - active_b[0]))
    print("ptime third", ptime_third)
    print("ptime start", ptime_start)
    print("active_e", active_e[0])

    #first car enters ( primitive)
    #second car enters (primitive ) (calculating TTC)
    #when TTC (game)
    #primitive second car
    #primirive first car
    timesteps_car1 = list(np.arange(active_b[0], active_e[0], 100))
    timesteps_car2 = list(np.arange(active_2b[0], active_2e[0], 100))
   # print(timesteps_car1)

    game_frenet = False
    calc_frenet = False



    ptime_start = ptime_third
    car1_states_inferred_x = []
    car2_states_inferred_x = []
    car1_states_inferred_y = []
    car2_states_inferred_y = []
    first_time_index = -1
    while ptime_start< active_end: # primitive end time
        hascar1= False #ego car
        hascar2= False

        if ptime_start in timesteps_car1:
            time_index1 = timesteps_car1.index(ptime_start)
            position_1 = (active_x[0][time_index1] ** 2 + active_y[0][time_index1] ** 2) ** (1 / 2)
            velocity_1 = (active_vx[0][time_index1] ** 2 + active_vy[0][time_index1] ** 2) ** (1 / 2)
            hascar1= True

        if ptime_start in timesteps_car2:
            time_index2 = timesteps_car2.index(ptime_start)
            position_2 = (active_2x[0][time_index2] ** 2 + active_2y[0][time_index2] ** 2) ** (1 / 2)
            velocity_2 = (active_2vx[0][time_index2] ** 2 + active_2vy[0][time_index2] ** 2) ** (1 / 2)
            hascar2= True

        #print(hascar1, " has car ",  hascar2)
        ptime_start = ptime_start + 100
        nodes_removed = 0
        # when both car are present #TODO: but have to start primitive when one car enters the intersections

        if hascar1 and hascar2:
            car_length = 5.0 # we chose maximum
            TTC = np.abs((position_2- position_1-car_length)/ (velocity_1- velocity_2))
            dist = abs(position_2- position_1)


            print("Time to collision is:",TTC)
            #total_prediction= 10

            car_states_true= []

            if dist < 5.0:
                # The agents are interacting
                print("The agents are interacting. Distance between them is", dist)
            else:
                print("The agents are NOT interacting.")


def main():
    print(__file__ + " start!!")

    # way points
    # wx = [0.0, 10.0, 20.5, 35.0, 70.5]
    # wy = [0.0, -6.0, 5.0, 6.5, 0.0]
    wx = []
    wy = []
    wx_2 =[]
    wy_2 =[]
    name = 'vehicle_tracks_000_1.csv'
    vehicle_no = 19
    second_agent = 18
    with open('C:/Users/skart/PycharmProjects/interaction-dataset/data/' + name, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        t = 0

        for row in reader:
            if t > 0:
                #row = [float(val) for val in row]
                row_no = int(row[0])

                # print("This is Row")
                # print(row)
                #print(i)
                #i = i + 1
                #print("OK")

                # col.append(row[1])
                #if row[0] == vehicle_no: # vehicle 2 data for testing\
                if row_no == vehicle_no and t % 20 == 0:
                    #T.append(row[2])
                    row_x = float(row[3])
                    row_y = float(row[4])

                    #wx.append(row[3])
                    #wy.append(row[4])
                    wx.append(row_x)
                    wy.append(row_y)

                    #vx.append(row[5])
                    #vy.append(row[6])
                    #psi.append(row[7])
                if row_no == second_agent and t % 15 == 0:
                    row_x = float(row[3])
                    row_y = float(row[4])

                    #wx.append(row[3])
                    #wy.append(row[4])
                    wx_2.append(row_x)
                    wy_2.append(row_y)

            t = t+1

    # print("this is the wx vals:")
    # print(len(wx))
    # print(len(wy))





    # get one trajectory eg trajectory 19
    # get random 10 points from it            # I have taken all the points for now. Not just 10 points.

    #spherical obstcale , four point obstacle
    # obstacle lists

    ob = points_in_circle_np(10.3, 1018.5, 999) # call function to calculate list of obstcale - here it is the list of all point inside the intersection

    ob = np.array(ob)

    # #ob = np.array([[20.0, 10.0],
    #                [30.0, 6.0],
    #                [30.0, 8.0],
    #                [35.0, 8.0],
    #                [50.0, 3.0]
    #                ])

    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)
    tx_2, ty_2, tyaw_2, tc_2, csp_2 = generate_target_course(wx_2, wy_2)

    #change based on interaction dataset
    # initial state
    #c_speed = 10.0 / 3.6  # current speed [m/s]
    c_speed = 6.9444       # current speed [m/s] for interaction dataset
    c_accel = 0.0  # current acceleration [m/ss]
    c_d = 2.0  # current lateral position [m]
    c_d_d = 0.0  # current lateral speed [m/s]
    c_d_dd = 0.0  # current lateral acceleration [m/s]
    s0 = 0.0  # current course position

    c_speed_2 = 6.9444       # current speed [m/s] for interaction dataset
    c_accel_2 = 0.0  # current acceleration [m/ss]
    c_d_2 = 2.0  # current lateral position [m]
    c_d_d_2 = 0.0  # current lateral speed [m/s]
    c_d_dd_2 = 0.0  # current lateral acceleration [m/s]
    s0_2 = 0.0  # current course position

    area = 100.0  # animation area length [m]
    yy = 0
    for i in range(SIM_LOOP):
        path = frenet_optimal_planning(
            csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, ob)
        path_2 = frenet_optimal_planning(
            csp_2, s0_2, c_speed_2, c_accel_2, c_d_2, c_d_d_2, c_d_dd_2, ob)

        # print(yy)
        # yy = yy + 1
        #print(path)

        s0 = path.s[1]
        c_d = path.d[1]
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]
        c_speed = path.s_d[1]
        c_accel = path.s_dd[1]

        s0_2 = path_2.s[1]
        c_d_2 = path_2.d[1]
        c_d_d_2 = path_2.d_d[1]
        c_d_dd_2 = path_2.d_dd[1]
        c_speed_2 = path_2.s_d[1]
        c_accel_2 = path_2.s_dd[1]



        #
        # if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 1.0:
        #     print("Goal")
        #     break
        if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 1.0 and np.hypot(path_2.x[1] - tx_2[-1], path_2.y[1] - ty_2[-1]) <= 1.0:
            print("Goal")
            break
        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(tx, ty)
            plt.plot(tx_2, ty_2)
            plt.plot(ob[:, 0], ob[:, 1], "xk")

            ### for first agent ###
            plt.plot(path.x[1:], path.y[1:], "-or")
            plt.plot(path.x[1], path.y[1], "vc")

            ### for second agent ###
            plt.plot(path_2.x[1:], path_2.y[1:], "-oc")
            plt.plot(path_2.x[1], path_2.y[1], "vr")

            plt.xlim(path.x[1] - area, path.x[1] + area)
            plt.ylim(path.y[1] - area, path.y[1] + area)
            plt.title("v[km/h]:" + str(c_speed * 3.6)[0:4])
            plt.grid(True)
            plt.pause(0.0001)

            calc_ttc(vehicle_no, second_agent)

    print("Finish")
    if show_animation:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.0001)
        plt.show()


if __name__ == '__main__':
    main()

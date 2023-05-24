
import sys
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
fullpath = parentdir + '\\python'
sys.path.append(fullpath)

import numpy as np
import pickle
from utils import dataset_types 
import matplotlib.pyplot as plt


# containter for the data



def get_traj_filepath(scenario, file_number):
    return "C:/Users/skart/PycharmProjects/interaction-dataset/trajectory_files/" + str(scenario) + "/vehicle_tracks_" + str(file_number).zfill(3) + "_trajs.pickle"

def get_traj_filepath_interaction(scenario, file_number):
    return "C:/Users/skart/PycharmProjects/interaction-dataset/trajectory_files/" + str(scenario) + "/vehicle_tracks_" + str(file_number).zfill(3) + "_trajs_int.pickle"

def get_traj_filepath_active(scenario, file_number):
    return "C:/Users/skart/PycharmProjects/interaction-dataset/trajectory_files/" + str(scenario) + "/vehicle_tracks_" + str(file_number).zfill(3) + "_trajs_active.pickle"

    #trajectory_files/Scenario4/vehicle_tracks_000_trajs_int.pickle"
def extract_dataset_traj(scenario, is_interaction, entrances=[],exits=[],data_lim=sys.maxsize):
    '''
        This function takes in a scenario name and a trajectory type (interaction
        vs no interaction) and returns x, y, xv, yv, psi values from cars from the
        specified scenario that fit the trajectory type

        Specifying data_lim will take only the first <data_lim> cars from
        <scenario> that have the specified trajectory type
    '''
    car_id = []
    big_x_traj = []
    big_y_traj = []
    big_vx_traj = []
    big_vy_traj = []
    big_psi_traj = []
    traj_dict = {}
    #plt.figure()
    # load first available traj file
    file_number = 0
    file_path = get_traj_filepath(scenario, file_number)
    print("reading from,", file_path)

    while os.path.isfile(file_path):
        # open trajectory file
        with open(file_path, 'rb') as handle:
            traj_dict = pickle.load(handle)

        # for all cars in current traj file
        for car in traj_dict.values():
            # check if data exceeds data limits
            data_count = len(big_x_traj)
            if data_count >= data_lim:
                return(big_x_traj, big_y_traj, big_vx_traj, big_vy_traj, big_psi_traj)
            # extract car data
            # debug figure

            if is_interaction == car.interaction:
                if (not exits and not entrances) or (car.exit_id in exits and car.entrance_id in entrances):
                    #car_id.append([file_number, car.track_id])    # -- debug
                    if car.track_id != 131:
                        # plt.plot(np.array(car.x_vals),np.array(car.y_vals))
                        # plt.title("Car track id {}".format(car.track_id))
                        # plt.show()
                        big_x_traj.append(np.array(car.x_vals))
                        big_y_traj.append(np.array(car.y_vals))
                        big_vx_traj.append(np.array(car.vx_vals))
                        big_vy_traj.append(np.array(car.vy_vals))
                        big_psi_traj.append(np.array(car.psi_vals))

                    #print(np.sum (car.psi_vals[0:10]-car.vy_vals[0:10]))

        # iterate to next file
        file_number += 1
        file_path = get_traj_filepath(scenario, file_number)

    # return all cars found
    return(big_x_traj, big_y_traj, big_vx_traj, big_vy_traj, big_psi_traj)


#x, y, vx, vy, psi  = extract_dataset_traj("Scenario4", False, [3], [5], data_lim=10)
def extract_dataset_traj_interaction(scenario, is_interaction, entrances=[], exits=[], data_lim=sys.maxsize):
    '''
        This function takes in a scenario name and a trajectory type (interaction
        vs no interaction) and returns x, y, xv, yv, psi values from cars from the
        specified scenario that fit the trajectory type

        Specifying data_lim will take only the first <data_lim> cars from
        <scenario> that have the specified trajectory type
    '''
    big_x_traj = []
    big_y_traj = []
    big_vx_traj = []
    big_vy_traj = []
    big_psi_traj = []
    valid_id = []
    big_interact = []

    traj_dict = {}
    # plt.figure()
    # load first available traj file
    file_number = 0
    file_path = get_traj_filepath_interaction(scenario, file_number)
    print("reading from,", file_path)

    while os.path.isfile(file_path):
        # open trajectory file
        with open(file_path, 'rb') as handle:
            traj_dict = pickle.load(handle)
        #print(len(traj_dict))

        # for all cars in current traj file
        for car in traj_dict.values():
            # check if data exceeds data limits
            # if not hasattr(car, "interact_with"):
            #     print(car.track_id, " is missing")
            # else:
            #     print (car)
            data_count = len(big_x_traj)
            if data_count >= data_lim:
                return (big_x_traj, big_y_traj, big_vx_traj, big_vy_traj, big_psi_traj)
            # extract car data
            # debug figure

            if is_interaction == car.interaction:

                if (not exits and not entrances):
                    if car.track_id != 131:
                        # plt.plot(np.array(car.x_vals), np.array(car.y_vals))
                        # plt.title("Car track id {}".format(car.track_id))
                        # plt.show()
                        big_x_traj.append(np.array(car.x_vals))
                        big_y_traj.append(np.array(car.y_vals))
                        big_vx_traj.append(np.array(car.vx_vals))
                        big_vy_traj.append(np.array(car.vy_vals))
                        big_psi_traj.append(np.array(car.psi_vals))

                elif (car.exit_id in exits and car.entrance_id in entrances):
                    # same entrance and exit id
                    if entrances[exits.index(car.exit_id)] == car.entrance_id:
                        # car_id.append([file_number, car.track_id])    # -- debug

                        if car.track_id != 131:
                            # plt.plot(np.array(car.x_vals), np.array(car.y_vals))
                            # plt.title("Car track id {}".format(car.track_id))
                            # plt.show()
                            valid_id.append(car.track_id)
                            big_x_traj.append(np.array(car.x_vals))
                            big_y_traj.append(np.array(car.y_vals))
                            big_vx_traj.append(np.array(car.vx_vals))
                            big_vy_traj.append(np.array(car.vy_vals))
                            big_psi_traj.append(np.array(car.psi_vals))

                            print(car.track_id, " interacts with ", car.interact_with.keys())
                            # if not hasattr(car,"interact_with"):
                            #     exit()
                            # print(car)
                            # print(car.interact_with)
                            big_interact.append(car.interact_with)

        # iterate to next file
        file_number += 1
        file_path = get_traj_filepath_interaction(scenario, file_number)

    int_x_traj = []
    int_y_traj = []
    int_vx_traj = []
    int_vy_traj = []
    int_psi_traj = []
    int_interact = []

    for i in range(len(big_x_traj)):
        interact_id = big_interact[i].keys()
        if len(set(interact_id)& set(valid_id)) > 0:
            print("valid ", valid_id[i], " interacts with ", big_interact[i])
            int_x_traj.append(big_x_traj[i])
            int_y_traj.append(big_y_traj[i])
            int_vx_traj.append(big_vx_traj[i])
            int_vy_traj.append(big_vy_traj[i])
            int_psi_traj.append(big_psi_traj[i])
            int_interact.append(big_interact[i])



    # return all cars found
    return (int_x_traj, int_y_traj, int_vx_traj, int_vy_traj, int_psi_traj, int_interact)


def extract_dataset_traj_active(scenario, is_interaction, entrances=[], exits=[], data_lim=sys.maxsize, track_id_list = []):
    '''
        This function takes in a scenario name and a trajectory type (interaction
        vs no interaction) and returns x, y, xv, yv, psi values from cars from the
        specified scenario that fit the trajectory type

        Specifying data_lim will take only the first <data_lim> cars from
        <scenario> that have the specified trajectory type
    '''
    big_x_traj = []
    big_y_traj = []
    big_vx_traj = []
    big_vy_traj = []
    big_psi_traj = []
    valid_id = []
    big_interact = []
    big_time_start = []
    big_time_end = []

    traj_dict = {}
    # plt.figure()
    # load first available traj file
    file_number = 1
    file_path = get_traj_filepath_interaction(scenario, file_number)
    print("reading from,", file_path)

    # while os.path.isfile(file_path):
    #     # open trajectory file
    with open(file_path, 'rb') as handle:
        traj_dict = pickle.load(handle)
        print(len(traj_dict))


    print("reading from,", file_path)
    # for all cars in current traj file
    for car in traj_dict.values():
        # check if data exceeds data limits
        # if not hasattr(car, "interact_with"):
        #     print(car.track_id, " is missing")
        # else:
        #     print (car)
        data_count = len(big_x_traj)
        if data_count >= data_lim:
            return (big_x_traj, big_y_traj, big_vx_traj, big_vy_traj, big_psi_traj)
        # extract car data
        # debug figure

        if track_id_list:
            for i in track_id_list:
                if (car.track_id == i):
                    valid_id.append(car.track_id)
                    big_x_traj.append(np.array(car.x_vals))
                    big_y_traj.append(np.array(car.y_vals))
                    big_vx_traj.append(np.array(car.vx_vals))
                    big_vy_traj.append(np.array(car.vy_vals))
                    big_psi_traj.append(np.array(car.psi_vals))
                    big_time_start.append(car.time_stamp_ms_first)
                    big_time_end.append(car.time_stamp_ms_last)

                    print(car.track_id, " interacts with first", car.interact_with.keys())
                    # if not hasattr(car,"interact_with"):
                    #     exit()
                    # print(car)
                    # print(car.interact_with)
                    big_interact.append(car.interact_with)


        else:
            if is_interaction == car.interaction:

                if (not exits and not entrances):
                    if car.track_id != 131:
                        # plt.plot(np.array(car.x_vals), np.array(car.y_vals))
                        # plt.title("Car track id {}".format(car.track_id))
                        # plt.show()
                        #big_timesteps.append()
                        big_x_traj.append(np.array(car.x_vals))
                        big_y_traj.append(np.array(car.y_vals))
                        big_vx_traj.append(np.array(car.vx_vals))
                        big_vy_traj.append(np.array(car.vy_vals))
                        big_psi_traj.append(np.array(car.psi_vals))

                elif (car.exit_id in exits and car.entrance_id in entrances):
                    # same entrance and exit id
                    if entrances[exits.index(car.exit_id)] == car.entrance_id:
                        # car_id.append([file_number, car.track_id])    # -- debug

                        if car.track_id != 131:
                            # plt.plot(np.array(car.x_vals), np.array(car.y_vals))
                            # plt.title("Car track id {}".format(car.track_id))
                            # plt.show()
                            valid_id.append(car.track_id)
                            big_x_traj.append(np.array(car.x_vals))
                            big_y_traj.append(np.array(car.y_vals))
                            big_vx_traj.append(np.array(car.vx_vals))
                            big_vy_traj.append(np.array(car.vy_vals))
                            big_psi_traj.append(np.array(car.psi_vals))
                            big_time_start.append(car.time_stamp_ms_first)
                            big_time_end.append(car.time_stamp_ms_last)

                            print(car.track_id, " interacts with second", car.interact_with.keys())
                            # if not hasattr(car,"interact_with"):
                            #     exit()
                            # print(car)
                            # print(car.interact_with)
                            big_interact.append(car.interact_with)


    # iterate to next file
    # file_number += 1
    # file_path = get_traj_filepath_active(scenario, file_number)


    int_trackid= []
    int_x_traj = []
    int_y_traj = []
    int_vx_traj = []
    int_vy_traj = []
    int_psi_traj = []
    int_interact = []
    int_start= []
    int_end  = []


    for i in range(len(big_x_traj)):
        interact_id = big_interact[i].keys()
        if track_id_list:
            #print("valid ", valid_id[i], " interacts with ", big_interact[i])
            int_trackid.append(valid_id[i])
            int_x_traj.append(big_x_traj[i])
            int_y_traj.append(big_y_traj[i])
            int_vx_traj.append(big_vx_traj[i])
            int_vy_traj.append(big_vy_traj[i])
            int_psi_traj.append(big_psi_traj[i])
            int_interact.append(big_interact[i])
            int_start.append(big_time_start[i])
            int_end.append(big_time_end[i])


        elif len(set(interact_id)& set(valid_id)) > 0:
            print("valid ", valid_id[i], " interacts with ", big_interact[i])
            int_trackid.append(valid_id[i])
            int_x_traj.append(big_x_traj[i])
            int_y_traj.append(big_y_traj[i])
            int_vx_traj.append(big_vx_traj[i])
            int_vy_traj.append(big_vy_traj[i])
            int_psi_traj.append(big_psi_traj[i])
            int_interact.append(big_interact[i])
            int_start.append(big_time_start[i])
            int_end.append(big_time_end[i])




    # return all cars found
    return (int_x_traj, int_y_traj, int_vx_traj, int_vy_traj, int_psi_traj, int_interact, int_start, int_end)# int_trackid)

def generate_demo_traj():
    big_x_traj = []
    big_y_traj = []
    big_vx_traj = []
    big_vy_traj = []
    big_psi_traj = []
    psi = np.linspace(-np.pi / 4, np.pi / 4, 60)
    ts = 0.1
    v = 1.0
    L = 4
    # x = 0
    # y = 0
    # theta = 0
    # demonstaration of the data
    for demo_psi in range(len(psi)):
        x = 0
        y = 0
        theta = 0
        x_traj = []
        y_traj = []
        vx_traj = []
        vy_traj = []
        psi_traj = []

        for t in range(100):
            #print(psi[demo_psi])
            theta_dot = (v/L) * np.tan(psi[demo_psi])
            theta = theta + ts* theta_dot
            x_dot = v* np.cos(theta)
            y_dot = v* np.sin(theta)
            x = x + ts * x_dot
            y = y + ts* y_dot

            x_traj.append(x)
            y_traj.append(y)
            vx_traj.append(x_dot)
            vy_traj.append(y_dot)
            psi_traj.append(theta)

        big_x_traj.append(np.array(x_traj))
        big_y_traj.append(np.array(y_traj))
        big_vx_traj.append(np.array(vx_traj))
        big_vy_traj.append(np.array(vy_traj))
        big_psi_traj.append(np.array(psi_traj))


        plt.figure(1)
        plt.plot(x_traj, y_traj)
        # x = 0
        # y = 0
        # theta = 0
        # x_traj = []
        # y_traj = []
        # vx_traj = []
        # vy_traj = []
        # psi_traj = []
        #
        # for t in range(100):
        #     #print(psi[demo_psi])
        #     theta_dot = -(v/L) * np.tan(psi[demo_psi])
        #     theta = theta + ts* theta_dot
        #     x_dot = -v* np.cos(theta)
        #     y_dot = - v* np.sin(theta)
        #     x = x + ts * x_dot
        #     y = y + ts* y_dot
        #
        #     x_traj.append(x)
        #     y_traj.append(y)
        #     vx_traj.append(x_dot)
        #     vy_traj.append(y_dot)
        #     psi_traj.append(theta)
        #
        # plt.plot(x_traj, y_traj)
        # big_x_traj.append(np.array(x_traj))
        # big_y_traj.append(np.array(y_traj))
        # big_vx_traj.append(np.array(vx_traj))
        # big_vy_traj.append(np.array(vy_traj))
        # big_psi_traj.append(np.array(psi_traj))

    # #
    plt.title('tester_plot')
    plt.show()


    return(big_x_traj, big_y_traj, big_vx_traj, big_vy_traj, big_psi_traj)

if __name__ == "__main__":
    #all_x, all_y, all_vx, all_vy, all_psi= generate_demo_traj()
    all_x, all_y, all_vx, all_vy, all_psi = extract_dataset_traj("Scenario4", False, [3], [5], data_lim=1000)
    all_x2, all_y2, all_vx2, all_vy2, all_psi2, all_int, all_2b, all_2e = extract_dataset_traj_active("Scenario4", True, [4], [2], data_lim= 100, track_id_list = [19, 107])
    #int_x, int_y, int_vx, int_vy, int_psi, int_interact = extract_dataset_traj_interaction("Scenario4", True, [3], [5], data_lim=1000)
    active_x, active_y, active_vx, active_vy, active_psi, active_int, active_b, active_e =  extract_dataset_traj_active("Scenario4", True, [3], [5], data_lim=100, track_id_list = [37,47,77])
    active_2x, active_2y, active_2vx, active_2vy, active_2psi, active_2int, active_2b, active_2e =  extract_dataset_traj_active("Scenario4", True, [3], [5], data_lim=100, track_id_list = [41,46,60])
    # print("total traj ", len(int_x))
    # import matplotlib.image as mpimg
    # img = mpimg.imread('C:/Users/samatya.ASURITE/PycharmProjects/interaction-dataset/images/DR_USA_Roundabout_FT.png')
    plt.figure()

    # fig, ax1 = plt.subplots()
    #plt.imshow(img)
    for i in range( len (all_x2)):
        #print (int_interact[i])
        #plt.plot(all_x[i], all_y[i])
        #plt.plot(all_vx[i], label= 'non interacting')
        #plt.plot(int_vx[i], label= 'interacting')
        #print(all_interact[i])
        #plt.plot(all_x[i],all_y[i], label= 'non interacting')
        #plt.plot(int_x[i], int_y[i],  label= 'interacting')
        #plt.plot(all_vx[i], 'r', label= 'non interacting velocity')
        # intracting
        plt.plot(active_x[i], active_y[i], 'g', label= 'interacting velocity')
        plt.plot(active_2x[i], active_2y[i], 'y', label= 'other interacting velocity')
        #non interaacting
        plt.plot(all_x[i+1], all_y[i+1], 'b', label='non interacting trajectory')
        plt.plot(all_x2[i], all_y2[i], 'r', label='other non interacting trajectory')
        # ax1.plot(all_vx[i], color="r", label="non interacting velocity ", linewidth=2.0)
        # ax1.plot(int_vx[i], color="m", label="interacting velocity ", linewidth=2.0)
        # plt.legend()
        # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        # ax2.plot(all_psi[i], color="b", label="non interacting angle ", linewidth=2.0)
        # ax2.plot(int_psi[i], color="c", label="non interacting angle ", linewidth=2.0)
        plt.legend()
    plt.show()







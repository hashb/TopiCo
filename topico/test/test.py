import numpy as np

import sys

sys.path.insert(0, "/home/kchenna/qoowa/TopiCo/topico/build")
np.set_printoptions(suppress=True, precision=4)

import pytopico


VELOCITY_LIMITS = np.array([[-3.3, 3.3], [-3.3, 3.3], [-3.3, 3.3], [-3.3, 3.3], [-3.3, 3.3], [-3.3, 3.3]])

ACCELERATION_LIMITS = np.array(
        [
            [-64.9626, 64.9626],
            [-64.9626, 64.9626],
            [-64.9626, 64.9626],
            [-64.9626, 64.9626],
            [-64.9626, 64.9626],
            [-64.9626, 64.9626],
        ]
    )

TRAJ = np.array(
    [
        [
            1.9705736637115479,
            -1.7495037317276,
            1.8706997632980347,
            -1.7114025354385376,
            -1.5922554731369019,
            0.4408855438232422,
        ],
        [
            1.9746211767196655,
            -1.7288497686386108,
            1.8279708623886108,
            -1.6833264827728271,
            -1.5853582620620728,
            0.4442117512226105,
        ],
        [
            1.9786685705184937,
            -1.708195686340332,
            1.7852420806884766,
            -1.6552503108978271,
            -1.578460931777954,
            0.44753798842430115,
        ],
        [
            1.9827160835266113,
            -1.6875417232513428,
            1.7425131797790527,
            -1.6271742582321167,
            -1.571563720703125,
            0.45086419582366943,
        ],
        [
            1.986763596534729,
            -1.6668877601623535,
            1.6997843980789185,
            -1.5990982055664062,
            -1.564666509628296,
            0.4541904032230377,
        ],
        [
            1.9908111095428467,
            -1.6462337970733643,
            1.6570554971694946,
            -1.5710220336914062,
            -1.5577691793441772,
            0.457516610622406,
        ],
        [
            1.9948585033416748,
            -1.6255797147750854,
            1.6143267154693604,
            -1.5429459810256958,
            -1.5508719682693481,
            0.4608428478240967,
        ],
        [
            1.9989060163497925,
            -1.6049257516860962,
            1.5715978145599365,
            -1.5148699283599854,
            -1.543974757194519,
            0.46416905522346497,
        ],
        [
            2.00295352935791,
            -1.584271788597107,
            1.5288689136505127,
            -1.4867937564849854,
            -1.5370774269104004,
            0.46749526262283325,
        ],
        [
            2.0070009231567383,
            -1.5636178255081177,
            1.4861401319503784,
            -1.458717703819275,
            -1.5301802158355713,
            0.4708214998245239,
        ],
        [
            2.019178628921509,
            -1.5832899808883667,
            1.5266751050949097,
            -1.4800697565078735,
            -1.5291085243225098,
            0.48327702283859253,
        ],
        [
            2.0313563346862793,
            -1.6029621362686157,
            1.5672101974487305,
            -1.5014216899871826,
            -1.5280369520187378,
            0.49573254585266113,
        ],
        [
            2.04353404045105,
            -1.6226342916488647,
            1.6077451705932617,
            -1.5227737426757812,
            -1.5269652605056763,
            0.5081880688667297,
        ],
        [
            2.0557117462158203,
            -1.6423065662384033,
            1.6482802629470825,
            -1.5441256761550903,
            -1.5258935689926147,
            0.5206435918807983,
        ],
        [
            2.067889451980591,
            -1.6619787216186523,
            1.6888152360916138,
            -1.565477728843689,
            -1.5248219966888428,
            0.5330991148948669,
        ],
        [
            2.0800669193267822,
            -1.6816508769989014,
            1.7293503284454346,
            -1.586829662322998,
            -1.5237503051757812,
            0.5455546975135803,
        ],
        [
            2.0922446250915527,
            -1.7013230323791504,
            1.7698853015899658,
            -1.6081817150115967,
            -1.5226786136627197,
            0.5580102205276489,
        ],
        [
            2.1044223308563232,
            -1.720995306968689,
            1.8104203939437866,
            -1.6295336484909058,
            -1.5216070413589478,
            0.5704657435417175,
        ],
        [
            2.1166000366210938,
            -1.740667462348938,
            1.8509553670883179,
            -1.6508857011795044,
            -1.5205353498458862,
            0.5829212665557861,
        ],
    ]
)
SEG_IDX = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        2.0,
    ]
)


SYNC = SEG_IDX == 0.0

def filter_waypoints(traj):
    traj_shape  = traj.shape
    n_waypoints = traj_shape[0]

    is_active = [True] * traj_shape[0]
    threshold_distance = [0.1] * traj_shape[1]

    start = 0
    end = start + 2

    while end < n_waypoints+2:
        pos_start = traj[max(0, start-1)]
        pos_end = traj[min(end - 1, n_waypoints-1)]

        are_all_below = True
        for current in range(start + 1, end):
            pos_current = traj[current-1]

            t_start_max = 0.0
            t_end_min = 1.0

            for dof in range(traj_shape[1]):
                h0 = (pos_current[dof] - pos_start[dof])/(pos_end[dof] - pos_start[dof])
                t_start = h0 - threshold_distance[dof]/abs(pos_end[dof] - pos_start[dof])
                t_end = h0 + threshold_distance[dof] / abs(pos_end[dof] - pos_start[dof])

                t_start_max = max(t_start, t_start_max)
                t_end_min = min(t_end, t_end_min)

                if t_start_max > t_end_min:
                    are_all_below = False
                    break
            if not are_all_below:
                break
        is_active[end-2] = not are_all_below
        if not are_all_below:
            start = end - 1
        end += 1

    is_active[0] = True
    is_active[-1] = True

    filtered_points = np.zeros((sum(is_active), traj_shape[1]))
    seg_idx = []
    count = 0
    for i, v in enumerate(is_active):
        if v:
            seg_idx.append(i)
            filtered_points[count,:] = traj[i, :]
            count+=1
    return filtered_points, seg_idx

import time

new_traj, seg_idxs = filter_waypoints(TRAJ)
scaling_factors = SEG_IDX[seg_idxs]
sync = SYNC[seg_idxs]

print("new traj: ", new_traj.shape, TRAJ.shape)
print("scaling_factors: ", scaling_factors.shape, SEG_IDX.shape)
print("sync: ", sync.shape, SYNC.shape)

t1 = time.monotonic()
result = pytopico.bb_retime(new_traj.copy(), scaling_factors.copy(), sync.copy(), VELOCITY_LIMITS, ACCELERATION_LIMITS)
t2 = time.monotonic()
time.sleep(0.1)
print(result[:,1:])
print(result.shape)
print(TRAJ.shape)
print(f"Time to compute {t2 - t1}")

# print(TRAJ.shape)
# print(filer_waypoints(TRAJ).shape)

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "rt_nonfinite.h"
#include "topico_wrapper.h"
#include "topico_wrapper_terminate.h"
#include "topico_wrapper_types.h"
#include "coder_array.h"
#include <sstream>
#include <stdexcept>
#include <string>

namespace py = pybind11;

py::array_t<double> bb_retime(py::array_t<double> &traj, py::array_t<double> &scaling_factors, py::array_t<bool> &sync, py::array_t<double> vel_lims, py::array_t<double> acc_lims)
{
    auto traj_arr = traj.unchecked<2>();
    auto scaling_factors_arr = scaling_factors.unchecked<1>();
    auto sync_arr = sync.unchecked<1>();
    auto vel_lims_arr = vel_lims.unchecked<2>();
    auto acc_lims_arr = acc_lims.unchecked<2>();

    py::buffer_info traj_buf_info = traj.request();
    py::array_t<double> result = py::array_t<double>(traj_buf_info.size);

    uint num_dim = traj_buf_info.shape[1];
    uint num_wayp = traj_buf_info.shape[0];

    // Declare Inputs
    coder::array<double, 2U> State_start;
    coder::array<double, 3U> Waypoints;
    coder::array<double, 2U> V_max;
    coder::array<double, 2U> V_min;
    coder::array<double, 2U> A_max;
    coder::array<double, 2U> A_min;
    coder::array<double, 2U> J_max;
    coder::array<double, 2U> J_min;
    coder::array<double, 1U> A_global;
    coder::array<bool, 2U> b_sync_V;
    coder::array<bool, 2U> b_sync_A;
    coder::array<bool, 2U> b_sync_J;
    coder::array<bool, 2U> b_sync_W;
    coder::array<bool, 2U> b_rotate;
    coder::array<bool, 2U> b_hard_V_lim;
    coder::array<bool, 2U> b_catch_up;
    coder::array<signed char, 2U> direction;
    double ts_rollout = 0.008;

    // Declare Outputs
    coder::array<struct0_T, 2U> J_setp_struct;
    coder::array<int, 2U> solution_out;
    coder::array<double, 2U> T_waypoints;
    coder::array<double, 2U> P;
    coder::array<double, 2U> V;
    coder::array<double, 2U> A;
    coder::array<double, 2U> J;
    coder::array<double, 2U> t;

    // Resize Inputs
    State_start.set_size(num_dim, 3);
    Waypoints.set_size(num_dim, 5, num_wayp);
    V_max.set_size(num_dim, num_wayp);
    V_min.set_size(num_dim, num_wayp);
    A_max.set_size(num_dim, num_wayp);
    A_min.set_size(num_dim, num_wayp);
    J_max.set_size(num_dim, num_wayp);
    J_min.set_size(num_dim, num_wayp);
    A_global.set_size(num_dim);
    b_sync_V.set_size(num_dim, num_wayp);
    b_sync_A.set_size(num_dim, num_wayp);
    b_sync_J.set_size(num_dim, num_wayp);
    b_sync_W.set_size(num_dim, num_wayp);
    b_rotate.set_size(num_dim - 1, num_wayp);
    b_hard_V_lim.set_size(num_dim, num_wayp);
    b_catch_up.set_size(num_dim, num_wayp);
    direction.set_size(num_dim, num_wayp);

    // update start state
    State_start[(0 + num_dim * 0)] = traj_arr(0, 0); // Initial X Position
    State_start[(1 + num_dim * 0)] = traj_arr(0, 1); // Initial Y Position
    State_start[(2 + num_dim * 0)] = traj_arr(0, 2); // Initial Z Position
    State_start[(3 + num_dim * 0)] = traj_arr(0, 3); // Initial Z Position
    State_start[(4 + num_dim * 0)] = traj_arr(0, 4); // Initial Z Position
    State_start[(5 + num_dim * 0)] = traj_arr(0, 5); // Initial Z Position

    State_start[(0 + num_dim * 1)] = 0.0; // Initial X Velocity
    State_start[(1 + num_dim * 1)] = 0.0; // Initial Y Velocity
    State_start[(2 + num_dim * 1)] = 0.0; // Initial Z Velocity
    State_start[(3 + num_dim * 1)] = 0.0; // Initial Z Velocity
    State_start[(4 + num_dim * 1)] = 0.0; // Initial Z Velocity
    State_start[(5 + num_dim * 1)] = 0.0; // Initial Z Velocity

    State_start[(0 + num_dim * 2)] = 0.0; // Initial X Acceleration
    State_start[(1 + num_dim * 2)] = 0.0; // Initial Y Acceleration
    State_start[(2 + num_dim * 2)] = 0.0; // Initial Z Acceleration
    State_start[(3 + num_dim * 2)] = 0.0; // Initial Z Acceleration
    State_start[(4 + num_dim * 2)] = 0.0; // Initial Z Acceleration
    State_start[(5 + num_dim * 2)] = 0.0; // Initial Z Acceleration

    for (size_t idx_wayp = 0; idx_wayp < num_wayp; idx_wayp++)
    {
        Waypoints[(0 + num_dim * 0) + num_dim * 5 * idx_wayp] = traj_arr(idx_wayp, 0); // Waypoint X Position
        Waypoints[(1 + num_dim * 0) + num_dim * 5 * idx_wayp] = traj_arr(idx_wayp, 1); // Waypoint Y Position
        Waypoints[(2 + num_dim * 0) + num_dim * 5 * idx_wayp] = traj_arr(idx_wayp, 2); // Waypoint Z Position
        Waypoints[(3 + num_dim * 0) + num_dim * 5 * idx_wayp] = traj_arr(idx_wayp, 3); // Waypoint Z Position
        Waypoints[(4 + num_dim * 0) + num_dim * 5 * idx_wayp] = traj_arr(idx_wayp, 4); // Waypoint Z Position
        Waypoints[(5 + num_dim * 0) + num_dim * 5 * idx_wayp] = traj_arr(idx_wayp, 5); // Waypoint Z Position

        Waypoints[(0 + num_dim * 1) + num_dim * 5 * idx_wayp] = rtNaNF; // Waypoint X Velocity
        Waypoints[(1 + num_dim * 1) + num_dim * 5 * idx_wayp] = rtNaNF; // Waypoint Y Velocity
        Waypoints[(2 + num_dim * 1) + num_dim * 5 * idx_wayp] = rtNaNF; // Waypoint Z Velocity
        Waypoints[(3 + num_dim * 1) + num_dim * 5 * idx_wayp] = rtNaNF; // Waypoint Z Velocity
        Waypoints[(4 + num_dim * 1) + num_dim * 5 * idx_wayp] = rtNaNF; // Waypoint Z Velocity
        Waypoints[(5 + num_dim * 1) + num_dim * 5 * idx_wayp] = rtNaNF; // Waypoint Z Velocity

        Waypoints[(0 + num_dim * 2) + num_dim * 5 * idx_wayp] = rtNaNF; // Waypoint X Acceleration
        Waypoints[(1 + num_dim * 2) + num_dim * 5 * idx_wayp] = rtNaNF; // Waypoint Y Acceleration
        Waypoints[(2 + num_dim * 2) + num_dim * 5 * idx_wayp] = rtNaNF; // Waypoint Z Acceleration
        Waypoints[(3 + num_dim * 2) + num_dim * 5 * idx_wayp] = rtNaNF; // Waypoint Z Acceleration
        Waypoints[(4 + num_dim * 2) + num_dim * 5 * idx_wayp] = rtNaNF; // Waypoint Z Acceleration
        Waypoints[(5 + num_dim * 2) + num_dim * 5 * idx_wayp] = rtNaNF; // Waypoint Z Acceleration

        if (idx_wayp == num_wayp - 1)
        {
            Waypoints[(0 + num_dim * 1) + num_dim * 5 * idx_wayp] = 0.0; // Waypoint X Velocity
            Waypoints[(1 + num_dim * 1) + num_dim * 5 * idx_wayp] = 0.0; // Waypoint Y Velocity
            Waypoints[(2 + num_dim * 1) + num_dim * 5 * idx_wayp] = 0.0; // Waypoint Z Velocity
            Waypoints[(3 + num_dim * 1) + num_dim * 5 * idx_wayp] = 0.0; // Waypoint Z Velocity
            Waypoints[(4 + num_dim * 1) + num_dim * 5 * idx_wayp] = 0.0; // Waypoint Z Velocity
            Waypoints[(5 + num_dim * 1) + num_dim * 5 * idx_wayp] = 0.0; // Waypoint Z Velocity

            Waypoints[(0 + num_dim * 2) + num_dim * 5 * idx_wayp] = 5.0; // Waypoint X Acceleration
            Waypoints[(1 + num_dim * 2) + num_dim * 5 * idx_wayp] = 5.0; // Waypoint Y Acceleration
            Waypoints[(2 + num_dim * 2) + num_dim * 5 * idx_wayp] = 5.0; // Waypoint Z Acceleration
            Waypoints[(3 + num_dim * 2) + num_dim * 5 * idx_wayp] = 5.0; // Waypoint Z Acceleration
            Waypoints[(4 + num_dim * 2) + num_dim * 5 * idx_wayp] = 5.0; // Waypoint Z Acceleration
            Waypoints[(5 + num_dim * 2) + num_dim * 5 * idx_wayp] = 5.0; // Waypoint Z Acceleration
        }

        Waypoints[(0 + num_dim * 3) + num_dim * 5 * idx_wayp] = 0.0; // Waypoint X Movement Velocity
        Waypoints[(1 + num_dim * 3) + num_dim * 5 * idx_wayp] = 0.0; // Waypoint Y Movement Velocity
        Waypoints[(2 + num_dim * 3) + num_dim * 5 * idx_wayp] = 0.0; // Waypoint Z Movement Velocity
        Waypoints[(3 + num_dim * 3) + num_dim * 5 * idx_wayp] = 0.0; // Waypoint Z Movement Velocity
        Waypoints[(4 + num_dim * 3) + num_dim * 5 * idx_wayp] = 0.0; // Waypoint Z Movement Velocity
        Waypoints[(5 + num_dim * 3) + num_dim * 5 * idx_wayp] = 0.0; // Waypoint Z Movement Velocity

        Waypoints[(0 + num_dim * 4) + num_dim * 5 * idx_wayp] = 0.0; // reserved
        Waypoints[(1 + num_dim * 4) + num_dim * 5 * idx_wayp] = 0.0; // reserved
        Waypoints[(2 + num_dim * 4) + num_dim * 5 * idx_wayp] = 0.0; // reserved
        Waypoints[(3 + num_dim * 4) + num_dim * 5 * idx_wayp] = 0.0; // reserved
        Waypoints[(4 + num_dim * 4) + num_dim * 5 * idx_wayp] = 0.0; // reserved
        Waypoints[(5 + num_dim * 4) + num_dim * 5 * idx_wayp] = 0.0; // reserved

        for (int idx_dim = 0; idx_dim < num_dim; idx_dim++)
        {
            std::cout << traj_arr(idx_wayp, idx_dim) << ", ";

            V_max[idx_dim + num_dim * idx_wayp] = vel_lims_arr(idx_dim, 1);
            V_min[idx_dim + num_dim * idx_wayp] = vel_lims_arr(idx_dim, 0);
            A_max[idx_dim + num_dim * idx_wayp] = acc_lims_arr(idx_dim, 1);
            A_min[idx_dim + num_dim * idx_wayp] = acc_lims_arr(idx_dim, 0);
            J_max[idx_dim + num_dim * idx_wayp] = rtInf;
            J_min[idx_dim + num_dim * idx_wayp] = -rtInf;
            b_sync_V[idx_dim + num_dim * idx_wayp] = true;
            b_sync_A[idx_dim + num_dim * idx_wayp] = true;
            if (idx_dim == 6)
            {
                b_sync_V[idx_dim + num_dim * idx_wayp] = sync_arr(idx_wayp);
                b_sync_A[idx_dim + num_dim * idx_wayp] = sync_arr(idx_wayp);
            }
            else
            {
                b_rotate[idx_dim + num_dim * idx_wayp] = false;
            }
            b_sync_J[idx_dim + num_dim * idx_wayp] = false;
            b_sync_W[idx_dim + num_dim * idx_wayp] = true;
            b_hard_V_lim[idx_dim + num_dim * idx_wayp] = false;
            b_catch_up[idx_dim + num_dim * idx_wayp] = false;
            direction[idx_dim + num_dim * idx_wayp] = 0.0;
            A_global[idx_dim] = 0.0;
        }
        std::cout << " [" << acc_lims_arr(0, 0) << ", " << acc_lims_arr(0, 1) << "]"
                  << "\n ";
    }

    topico_wrapper(State_start, Waypoints, V_max, V_min, A_max, A_min, J_max, J_min, A_global, b_sync_V, b_sync_A, b_sync_J, b_sync_W, b_rotate, b_hard_V_lim, b_catch_up, direction, ts_rollout, J_setp_struct, solution_out, T_waypoints, P, V, A, J, t);

    // py::buffer_info buf1 = input1.request();
    // py::buffer_info buf2 = input2.request();

    // if (buf1.size != buf2.size)
    // {
    //     throw std::runtime_error("Input shapes must match");
    // }

    // /*  allocate the buffer */
    // py::array_t<double> result = py::array_t<double>(buf1.size);

    // py::buffer_info buf3 = result.request();

    // double *ptr1 = (double *)buf1.ptr,
    //        *ptr2 = (double *)buf2.ptr,
    //        *ptr3 = (double *)buf3.ptr;
    // int X = buf1.shape[0];
    // int Y = buf1.shape[1];

    // for (size_t idx = 0; idx < X; idx++)
    // {
    //     for (size_t idy = 0; idy < Y; idy++)
    //     {
    //         ptr3[idx * Y + idy] = ptr1[idx * Y + idy] + ptr2[idx * Y + idy];
    //     }
    // }

    int size_rollout = P.size(1);
    result.resize({size_rollout, int(num_dim + 1)});

    py::buffer_info result_buf = result.request();
    double *ptr_result = (double *)result_buf.ptr;

    std::cout << "Output: \n";
    for (int i = 0; i < size_rollout; i++)
    {
        ptr_result[i * result_buf.shape[1]] = 0.008;
        for (int j = 0; j < num_dim; j++)
        {
            ptr_result[i * result_buf.shape[1] + j + 1] = P[num_dim * i + j];
            std::cout << P[num_dim * i + j] << ", ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    topico_wrapper_terminate();

    return result;
}

namespace pytopico
{
    PYBIND11_MODULE(pytopico, m)
    {
        m.doc() = "Python TopiCo Interface";

        m.def("bb_retime", &bb_retime, "A function computes time parameterization");
    }

}
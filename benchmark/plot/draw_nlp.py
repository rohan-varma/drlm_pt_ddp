import matplotlib.pyplot as plt
import numpy as np


colors = [
    [0.3, 0.3, 0.3],
    [0.6, 0.6, 0.6],
    [239/256.0, 74/256.0, 40/256.0],
]

WIDTH = 0.3
SHOW = False
FONT = {'fontname':'Times New Roman', 'size':22}

# figure 1
# BS = 32

# 1 GPU
# a. local pipeline fwd-bwd
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd

# 2 GPU
# a. local pipeline fwd-bwd
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd


# 4 GPU
# a. local pipeline fwd-bwd
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd


# 8 GPU
# a. local pipeline fwd-bwd
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd


data_fwd_mean = [
    [
        950.77, 
        941.81,
        933.35,
        931.17,
    ], # local pipeline
    [
        957.66,
        958.80,
        954.39,
        959.46,
    ], # CPU RPC
    [
        1160.06,
        1105.11,
        1103.12,
        1151.98,
    ], # CUDA RPC
]

data_fwd_stdv = [
    [
        1.26, 
        1.76, 
        2.08,
        2.25,
    ], # local pipeline
    [
        2.32,
        2.00,
        2.13,
        2.52,
    ], # CPU RPC
    [
        18.82,
        128.73,
        185.94,
        644.52,
    ], # CUDA RPC
]

data_comm_mean = [
    [
        0.02,
        1.50, 
        3.08,
        9.70,
    ],
    [
        654.42,
        706.89,
        809.52,
        1371.57,
    ],
    [
        36.34,
        24.44,
        16.17,
        138.47,
    ],
]

data_comm_stdv = [
    [
        0,
        0, 
        0,
        0,
    ],
    [
        4.62,
        5.17,
        24.47,
        26.13,
    ],
    [
        1.40,
        0.85,
        0.52,
        40.4,
    ],
]

data_bwd_mean = [
    [
        1748.39,
        1735.81,
        1729.72,
        1716.74,
    ],
    [
        2401.89,
        2472.37,
        2575.96,
        3163.30,
    ],
    [
        1775.24,
        1744.60,
        1741.34,
        1836.21,
    ],
]

data_bwd_stdv = [
    [
        3.67,
        4.63,
        5.79,
        4.84,
    ],
    [
        5.25,
        12.48,
        13.18,
        4.08,
    ],
    [
        5.02,
        4.21,
        5.65,
        45.48,
    ],
]

def plot_nlp(x_name, f_name, y_lim):
    plt.figure(figsize=(8, 4))
    xs = np.asarray(range(4))


    for i in range(3):
        fwd = np.asarray(data_fwd_mean[i]) / 1e3
        com = np.asarray(data_comm_mean[i]) / 1e3
        bwd = np.asarray(data_bwd_mean[i]) / 1e3 
        fwd_stdv = np.asarray(data_fwd_stdv[i]) / 1e3
        com_stdv = np.asarray(data_comm_stdv[i]) / 1e3
        bwd_stdv = np.asarray(data_bwd_stdv[i]) / 1e3

        bwd -= com
        com *= 2

        configs = {
            "width" : WIDTH,
            "color" : colors[i],
            "edgecolor" : "black",
            "capsize" : 6,
        }

        plt.bar(xs + (i - 1) * WIDTH, fwd, yerr=fwd_stdv, hatch="///", **configs)
        plt.bar(xs + (i - 1) * WIDTH, com, yerr=com_stdv, hatch="\\\\\\", bottom=fwd, **configs)
        plt.bar(xs + (i - 1) * WIDTH, bwd, yerr=bwd_stdv, hatch="...", bottom=fwd+com, **configs)


    color_handles = []
    color_handles.append(plt.bar([4], [0], color=colors[0]))
    color_handles.append(plt.bar([4], [0], color=colors[1]))
    color_handles.append(plt.bar([4], [0], color=colors[2]))
    color_names = ["SPMD", "CPU", "CUDA"]

    hatch_handles = []
    hatch_handles.append(plt.bar([4], [0], hatch="///", color="white"))
    hatch_handles.append(plt.bar([4], [0], hatch="\\\\\\", color="white"))
    hatch_handles.append(plt.bar([4], [0], hatch="...", color="white"))
    hatch_names = ["FWD", "COMM", "BWD"]

    def interleave(l1, l2):
        return [val for pair in zip(l1, l2) for val in pair]

    plt.legend(
        handles=interleave(color_handles, hatch_handles),
        loc="upper left",
        labels=interleave(color_names, hatch_names),
        prop={'family':FONT['fontname'], 'size':FONT['size'] - 2},
        ncol=3,
        #bbox_to_anchor=(-0.015, 0.3, 0.5, 0.5)
    )


    plt.xticks(xs, ["1", "2", "4", "8"], **FONT)
    plt.yticks(**FONT)

    plt.xlabel(x_name, **FONT)
    plt.ylabel("Delay (Second)", **FONT)

    plt.ylim(y_lim)
    plt.xlim([-0.5, 3.5])

    if SHOW:
        plt.show()
    else:
        plt.savefig(f"../images/{f_name}.pdf", bbox_inches='tight')


plot_nlp(x_name = "Number of GPUs", f_name="nlp_single", y_lim = [0, 10])

# figure 2
# BS = 32 nGPU = 8

# 1 machine
# a. local pipeline
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd

# 2 machine
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd


# 4 machine
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd


# 8 machine
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd

data_fwd_mean = [
    [
        946.43,
        0,
        0,
        0,
    ], # Local
    [
        948.76,
        942.96,
        950.27,
        943.09,
    ], # CPU RPC
    [
        1151.98,
        1255.05,
        1284.88,
        1154.35,
    ], # CUDA RPC
]

data_fwd_stdv = [
    [
        2.25,
        0,          
        0,
        0,
    ], # Local
    [
        2.52,
        5.19,
        5.91,
        5.14,
    ], # CPU RPC
    [
        630.23,
        838.80,
        712.13,
        654.59,
    ], # CUDA RPC
]

data_comm_mean = [
    [
        0,
        0, 
        0,
        0,
    ],
    [
        1279.16,
        2785.49,
        3291.64,
        3853.69,
    ],
    [
        123.31,
        842.68,
        997.94,
        1139.79
    ],
]

data_comm_stdv = [
    [
        0,
        0, 
        0,
        0,
    ],
    [
        17.80,
        22.71,
        25.95,
        23.94,
    ],
    [
        19.38,
        3.93,
        3.57,
        7.19,
    ],
]

data_bwd_mean = [
    [
        1716.74,
        0, 
        0,
        0,
    ],
    [
        5577.68,
        33219.79,
        32327.55,
        23170.96,
    ],
    [
        1836.21,
        2559.46,
        2726.81,
        2823.44,
    ],
]

data_bwd_stdv = [
    [
        4.84,
        0, 
        0,
        0,
    ],
    [
        150.76,
        215.27,
        224.92,
        579.19,
    ],
    [
        45.48,
        15.90,
        18.59,
        13.55,
    ],
]

plot_nlp(x_name="Number of Machines", f_name="nlp_multi_32", y_lim=[0, 40])

# figure 3
# BS = 128 nGPU = 8

# 1 machine
# a. local pipeline
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd

# 2 machine
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd


# 4 machine
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd


# 8 machine
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd




# figure 4
# BS = 256 nGPU = 8

# 1 machine
# a. local pipeline
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd

# 2 machine
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd


# 4 machine
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd


# 8 machine
# b. CPU RPC fwd-comm-bwd
# c. CUDA RPC fwd-comm-bwd



data_fwd_mean = [
    [
        6692.10,
        0,
        0,
        0,
    ], # Local
    [
        0,
        6673.24,
        6682.26,
        6661.95,
    ], # CPU RPC
    [
        7683.85,
        7933.53,
        7796.52,
        7731.03,
    ], # CUDA RPC
]

data_fwd_stdv = [
    [
        2.25,
        0,          
        0,
        0,
    ], # Local
    [
        0,
        7.48,
        14.63,
        11.27,
    ], # CPU RPC
    [
        105.22,
        127.51,
        77.30,
        125.28,
    ], # CUDA RPC
]

data_comm_mean = [
    [
        0,
        0,          
        0,
        0,
    ],
    [
        0,
        20498.61,
        19089.44,
        16194.94,
    ],
    [
        948.97,
        6651.87,
        7884.19,
        8738.51,
    ],
]

data_comm_stdv = [
    [
        0,
        0,          
        0,
        0,
    ],
    [
        0,
        64.29,
        33.30,
        51.62,
    ],
    [
        66.96,
        22.65,
        19.61,
        8.68,
    ],
]

data_bwd_mean = [
    [
        14424.63,
        0,          
        0,
        0,
    ],
    [
        0, 
        270899.38,
        259924.34,
        180946.89,
    ],
    [
        14578.84,
        20100.85,
        21365.95,
        22098.43,
    ],
]

data_bwd_stdv = [
    [
        53.63,
        0,          
        0,
        0,
    ],
    [
        0,
        818.91,
        903.98,
        415.90,
    ],
    [
        182.55,
        35.38,
        41.04,
        24.66,
    ],
]

plot_nlp(x_name="Number of Machines", f_name="nlp_multi_128", y_lim=[0, 300])
















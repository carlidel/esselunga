import numpy as np
from numba import njit, prange, int32
from numba.experimental import jitclass

spec = [
    ('normal', int32[:]),
    ('special', int32[:]),
    ('n_packets', int32),
    ('n_normal', int32),
    ('n_special', int32),
    ('n_p_normal', int32),
    ('n_p_special', int32),
]

@jitclass(spec)
class album(object):
    def __init__(self, N_FIGURES_NORMAL, N_FIGURES_SPECIAL, N_NORMAL_IN_PACKET, N_SPECIAL_IN_PACKET):
        self.normal = np.zeros(N_FIGURES_NORMAL, dtype=int32)
        self.special = np.zeros(N_FIGURES_SPECIAL, dtype=int32)
        self.n_packets = 0
        self.n_normal = N_FIGURES_NORMAL
        self.n_special = N_FIGURES_SPECIAL
        self.n_p_normal = N_NORMAL_IN_PACKET
        self.n_p_special = N_SPECIAL_IN_PACKET

    def doubles(self):
        return self.normal > 1, self.special > 1

    def missings(self):
        return self.normal == 0, self.special == 0

    def trade(self, trade_normal, trade_special):
        self.normal += trade_normal
        self.special += trade_special

    def open_packet(self, n_packets=1):
        for i in range(n_packets):
            self.n_packets += 1

            normal_sample = np.random.choice(self.n_normal, self.n_p_normal)
            for i in normal_sample:
                self.normal[i] += 1
            
            special_sample = np.random.choice(self.n_special, self.n_p_special)
            for i in special_sample:
                self.special[i] += 1
        

def trade_albums(album1, album2):
    dn1, ds1 = album1.doubles()
    dn2, ds2 = album2.doubles()
    
    mn1, ms1 = album1.missings()
    mn2, ms2 = album2.missings()

    trade_n_1 = np.zeros_like(dn1, dtype=int) 
    trade_n_2 = np.zeros_like(dn1, dtype=int)
    trade_s_1 = np.zeros_like(ds1, dtype=int)
    trade_s_2 = np.zeros_like(ds1, dtype=int)

    trade_n_1, trade_n_2, trade_s_1, trade_s_2 = jit_compare(
        dn1, ds1, dn2, ds2,
        mn1, ms1, mn2, ms2,
        trade_n_1, trade_n_2, trade_s_1, trade_s_2
    )

    album1.trade(trade_n_1, trade_s_1)
    album2.trade(trade_n_2, trade_s_2)


@njit()
def jit_compare(dn1, ds1, dn2, ds2, mn1, ms1, mn2, ms2, trade_n_1, trade_n_2, trade_s_1, trade_s_2):
    # TRADING STRATEGY: treat normal and special separately.
    need_1 = 0
    need_2 = 0
    for i in range(len(dn1)):
        if dn1[i] and mn2[i]:
            need_2 += 1
            trade_n_1[i] = -1
            trade_n_2[i] = need_2
        if dn2[i] and mn1[i]:
            need_1 += 1
            trade_n_2[i] = -1
            trade_n_1[i] = need_1
    min_need = min(need_1, need_2)
    for i in range(len(dn1)):
        if trade_n_1[i] > min_need:
            trade_n_1[i] = 0
        elif trade_n_1[i] > 0:
            trade_n_1[i] = 1
        if trade_n_2[i] > min_need:
            trade_n_2[i] = 0
        elif trade_n_2[i] > 0:
            trade_n_2[i] = 1

    for i in range(len(dn1)):
        if trade_n_1[i] == -1 and trade_n_2[i] == 0:
            trade_n_1[i] = 0
        if trade_n_2[i] == -1 and trade_n_1[i] == 0:
            trade_n_2[i] = 0

    need_1 = 0
    need_2 = 0
    for i in range(len(ds1)):
        if ds1[i] and ms2[i]:
            need_2 += 1
            trade_s_1[i] = -1
            trade_s_2[i] = need_2
        if ds2[i] and ms1[i]:
            need_1 += 1
            trade_s_2[i] = -1
            trade_s_1[i] = need_1
    min_need = min(need_1, need_2)
    for i in range(len(ds1)):
        if trade_s_1[i] > min_need:
            trade_s_1[i] = 0
        elif trade_s_1[i] > 0:
            trade_s_1[i] = 1
        if trade_s_2[i] > min_need:
            trade_s_2[i] = 0
        elif trade_s_2[i] > 0:
            trade_s_2[i] = 1
    
    for i in range(len(ds1)):
        if trade_s_1[i] == -1 and trade_s_2[i] == 0:
            trade_s_1[i] = 0
        if trade_s_2[i] == -1 and trade_s_1[i] == 0:
            trade_s_2[i] = 0

    return trade_n_1, trade_n_2, trade_s_1, trade_s_2

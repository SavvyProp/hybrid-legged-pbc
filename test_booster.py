from lowctrl.eefpbc import ctrl2components
from models.booster_t1.booster_ids import ids
from pipelines.booster_eefpbc import default_act

ctrl = default_act()

(des_pos, gnd_acc, 
     qp_weights, tau_mix, 
     w, oriens, 
     base_acc, select) = ctrl2components(ctrl, ids)


print(des_pos)
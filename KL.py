import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
cmap=[plt.cm.Paired(1),plt.cm.Paired(6),plt.cm.Paired(2),plt.cm.Paired(3),plt.cm.Paired(4),plt.cm.Paired(5),plt.cm.Paired(0),plt.cm.Paired(7),plt.cm.Paired(8),plt.cm.Paired(9),plt.cm.Paired(10)]
cm = plt.get_cmap('tab20')
cm1 = plt.get_cmap('tab20b')
fig, (ax1,ax2) = plt.subplots(1,2,figsize=[14,5])
# fig, ax1 = plt.subplots(figsize=[15,6])
k1 = np.load('kl_list_1_proj_PC_allsins.npy')
k2 = np.load('kl_list_2_concat_PC_allsingle.npy')
k3 = np.load('kl_list_1_layer_concat_PC_allsins.npy')
k4 = np.load('kl_list_2_encoder_h_concat_PC_allsins.npy')
k_c1 = np.load('hier_kl_self_seg_com_latent_seg_PC_sinturns.npy')
# hier_kl_self_seg_com_latent_seg_PC_sinturns
#
# gaussian_filter1d(k1[:10000], sigma=5)
ax1.plot(gaussian_filter1d(k1[:9000], sigma=2),label = 'PC_proj_sint',c=cm(0))
ax1.plot(gaussian_filter1d(k2[:9000], sigma=2),label = 'PC_concat_sint',c=cm(2))
ax1.plot(gaussian_filter1d(k3[:9000], sigma=2),label = 'PC_past_sint',c=cm(4))
ax1.plot(gaussian_filter1d(k4[:9000], sigma=2),label = 'PC_encoder_sint',c=cm(6))
ax1.plot(gaussian_filter1d(k_c1[:9000], sigma=2),label = 'PC_com_sint',c=cm(8))

# ax1.plot(k1[:8000],label = 'PC_proj_sint')
# ax1.plot(k2[:8000],label = 'PC_concat_sit')
# ax1.plot(k3[:8000],label = 'PC_past_sint')
# ax1.plot(k4[:8000],label = 'PC_encoder_sint')


k5 = np.load('kl_list_2_proj_PC_allturns.npy')
k6 = np.load('kl_list_1_concat_PC_allturns.npy')
k7 = np.load('kl_list_1_layer_concat_PC_allturns.npy')
k8 = np.load('kl_list_1_encoder_h_concat_PC_allturns.npy')
k_c2 = np.load('hier_kl_self_seg_com_latent_seg_PC_allturns.npy')

ax1.plot(gaussian_filter1d(k5[:9000], sigma=2),label = 'PC_proj_allt',c=cm(1))
ax1.plot(gaussian_filter1d(k6[:9000], sigma=2),label = 'PC_concat_allt',c=cm(3))
ax1.plot(gaussian_filter1d(k7[:9000], sigma=2),label = 'PC_past_allt',c=cm(5))
ax1.plot(gaussian_filter1d(k8[:9000], sigma=2),label = 'PC_encoder_allt',c=cm1(15))
ax1.plot(gaussian_filter1d(k_c2[:9000], sigma=2),label = 'PC_com_allt',c=cm(9))

# ax1.plot(k5[:8000],label = 'PC_proj_allt')
# ax1.plot(k6[:8000],label = 'PC_concat_allt')
# ax1.plot(k7[:8000],label = 'PC_past_allt')
# ax1.plot(k8[:8000],label = 'PC_encoder_allt')


k9 = np.load('kl_list_2_proj_DD_allsingle.npy')
k10 = np.load('kl_list_2_concat_DD_allsins.npy')
k11 = np.load('kl_list_2_layer_concat_DD_allsin.npy')
k12 = np.load('kl_list_2_encoder_h_concat_DD_allsin.npy')
k_c3 =np.load('hier_kl_self_seg_com_latent_seg_DD_sinturns.npy')


# ax2.plot(k9[:8000],label = 'DD_proj_sint')
# ax2.plot(k10[:8000],label = 'DD_concat_sit')
# ax2.plot(k11[:8000],label = 'DD_past_sint')
# ax2.plot(k12[:8000],label = 'DD_encoder_sint')
ax2.plot(gaussian_filter1d(k9[:9000], sigma=2),label = 'DD_proj_sint',c=cm(0))
ax2.plot(gaussian_filter1d(k10[:9000], sigma=2),label = 'DD_concat_sint',c=cm(2))
ax2.plot(gaussian_filter1d(k11[:9000], sigma=2),label = 'DD_past_sint',c=cm(4))
ax2.plot(gaussian_filter1d(k12[:9000], sigma=2),label = 'DD_encoder_sint',c=cm(6))
ax2.plot(gaussian_filter1d(k_c3[:9000], sigma=2),label = 'DD_com_sint',c=cm(8))

k13 = np.load('kl_list_2_proj_DD_allturns.npy')
k14 = np.load('kl_list_2_concat_DD_allturns.npy')
k15 = np.load('kl_list_2_layer_concat_p_r_DD_allturns.npy')
k16 = np.load('kl_list_2_encoder_h_concat_DD_allturn.npy')
k_c4 =np.load('hier_kl_self_seg_com_latent_seg_DD_allturns.npy')

# ax2.plot(k13[:8000],label = 'DD_proj_allt')
# ax2.plot(k14[:8000],label = 'DD_concat_allt')
# ax2.plot(k15[:8000],label = 'DD_past_allt')
# ax2.plot(k16[:8000],label = 'DD_encoder_allt')
ax2.plot(gaussian_filter1d(k13[:9000], sigma=2),label = 'DD_proj_allt',c=cm(1))
ax2.plot(gaussian_filter1d(k14[:9000], sigma=2),label = 'DD_concat_allt',c=cm(3))
ax2.plot(gaussian_filter1d(k15[:9000], sigma=2),label = 'DD_past_allt',c=cm(5))
ax2.plot(gaussian_filter1d(k16[:9000], sigma=2),label = 'DD_encoder_allt',c=cm1(15))
ax2.plot(gaussian_filter1d(k_c4[:9000], sigma=2),label = 'DD_com_allt',c=cm(9))

# plt.plot(tt)
ax1.set(xlabel='Training Times', ylabel='KL',
       title='KL in Persona Chat')
ax2.set(xlabel='Training Times', ylabel='KL',
       title='KL in Daily Dialog ')
ax2.set_ylim(0,1450)
ax1.set_ylim(0,1450)
ax1.legend()
ax2.legend()
plt.savefig('KL.jpg',dpi =600)
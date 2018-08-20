import netCDF4 as nc
import numpy as np
import pylab as plt


my_colors=['blue','red','green','magenta','cyan','yellow','brown','Orange','DarkViolet','SkyBlue','Chartreuse',
           'DarkKhaki','DarkSeaGreen','DeepPink','IndianRed','Lavender', 'MediumOrchid']

file_plus_path = '/Volumes/data5/Kyle/spectra/CondStats.DYCOMS_RF01_order_25.nc'


dx = 35.0
dz = 5.0
nx = 96
dk = 2.0*np.pi/(dx*nx)

data = nc.Dataset(file_plus_path, 'r' )

spectra = data.groups['spectra']

z = spectra.variables['z'][:]

kappa = spectra.variables['wavenumber'][:]/dk

E_TKE = spectra.variables['energy_spectrum'][23,:,:]

E_QT = spectra.variables['qt_spectrum'][23,:,:]

E_S = spectra.variables['s_spectrum'][23,:,:]
E_SQT = spectra.variables['s_qt_cospectrum'][23,:,:]


k53 = kappa**(-5.0/3.0)

plt.figure(1)
plt.title('Energy spectra')
count = 0
for k in np.arange(80,90,20):
    plt.semilogy(kappa, E_TKE[k,:]/(dk*np.sum(E_TKE[k,:])),   basey=10,label = '5m5s, ILES', color = my_colors[count] )
    count +=1
plt.semilogy(kappa, k53*1000.0,linewidth=2,   basey=10, color ='black', label='__nolabel__')
plt.legend(loc=0)
plt.xlabel(r'$\kappa\left(\frac{L}{2\pi} \right)$',fontsize=20)
plt.ylabel(r'$E\left(\kappa\right)$',fontsize=20)
plt.ylim(1e-8,1e3)

plt.figure(2)
plt.title('Entropy variance spectra')
count = 0
for k in np.arange(80,90,20):
    plt.semilogy(kappa, E_S[k,:]/(dk*np.sum(E_S[k,:])),   basey=10,label = '5m5s, ILES', color = my_colors[count] )
    count +=1
plt.semilogy(kappa, k53*1000.0,linewidth=2,   basey=10,  color ='black', label='__nolabel__')
plt.legend(loc=0)
plt.xlabel(r'$\kappa\left(\frac{L}{2\pi} \right)$',fontsize=20)
plt.ylabel(r'$E_s\left(\kappa\right)$',fontsize=20)
plt.ylim(1e-8,1e3)

plt.figure(3)
plt.title('Total water variance spectra')
count = 0
for k in np.arange(80,90,20):
    plt.semilogy(kappa, E_QT[k,:]/(dk*np.sum(E_QT[k,:])),   basey=10,label = '5m5s, ILES', color = my_colors[count] )
    count +=1
plt.semilogy(kappa, k53*1000.0,linewidth=2,   basey=10,  color ='black', label='__nolabel__')
plt.legend(loc=0)
plt.xlabel(r'$\kappa\left(\frac{L}{2\pi} \right)$',fontsize=20)
plt.ylabel(r'$E_{q_t}\left(\kappa\right)$',fontsize=20)

plt.ylim(1e-8,1e3)




plt.figure(4)
plt.title('Total water--entropy covariance spectra')
count = 0
for k in np.arange(80,90,20):
    plt.semilogy(kappa, E_SQT[k,:]/(dk*np.sum(E_SQT[k,:])),   basey=10,label = '5m5s, ILES', color = my_colors[count] )
    count +=1
plt.semilogy(kappa, k53*1000.0,linewidth=2,   basey=10,  color ='black', label='__nolabel__')
plt.legend(loc=0)
plt.xlabel(r'$\kappa\left(\frac{L}{2\pi} \right)$',fontsize=20)
plt.ylabel(r'$E_{s,q_t}\left(\kappa\right)$',fontsize=20)

plt.ylim(1e-8,1e3)






file_plus_path = '/Volumes/data5/Kyle/spectra/CondStats.DYCOMS_RF01_order_27.nc'


dx = 35.0
dz = 5.0
nx = 96
dk = 2.0*np.pi/(dx*nx)

data = nc.Dataset(file_plus_path, 'r' )

spectra = data.groups['spectra']

z = spectra.variables['z'][:]

kappa = spectra.variables['wavenumber'][:]/dk

E_TKE = spectra.variables['energy_spectrum'][23,:,:]

E_QT = spectra.variables['qt_spectrum'][23,:,:]

E_S = spectra.variables['s_spectrum'][23,:,:]
E_SQT = spectra.variables['s_qt_cospectrum'][23,:,:]


k53 = kappa**(-5.0/3.0)

plt.figure(1)
plt.title('Energy spectra')
count = 1
for k in np.arange(80,90,20):
    plt.semilogy(kappa, E_TKE[k,:]/(dk*np.sum(E_TKE[k,:])),   basey=10,label = '7m7s, ILES', color = my_colors[count] )
    count +=1
plt.semilogy(kappa, k53*1000.0,linewidth=2,   basey=10, color ='black', label='__nolabel__')
plt.legend(loc=0)
plt.xlabel(r'$\kappa\left(\frac{L}{2\pi} \right)$',fontsize=20)
plt.ylabel(r'$E\left(\kappa\right)$',fontsize=20)
plt.ylim(1e-8,1e3)

plt.figure(2)
plt.title('Entropy variance spectra')
count = 1
for k in np.arange(80,90,20):
    plt.semilogy(kappa, E_S[k,:]/(dk*np.sum(E_S[k,:])),   basey=10,label = '7m7s, ILES', color = my_colors[count] )
    count +=1
plt.semilogy(kappa, k53*1000.0,linewidth=2,   basey=10,  color ='black', label='__nolabel__')
plt.legend(loc=0)
plt.xlabel(r'$\kappa\left(\frac{L}{2\pi} \right)$',fontsize=20)
plt.ylabel(r'$E_s\left(\kappa\right)$',fontsize=20)
plt.ylim(1e-8,1e3)

plt.figure(3)
plt.title('Total water variance spectra')
count = 1
for k in np.arange(80,90,20):
    plt.semilogy(kappa, E_QT[k,:]/(dk*np.sum(E_QT[k,:])),   basey=10,label = '7m7s, ILES', color = my_colors[count] )
    count +=1
plt.semilogy(kappa, k53*1000.0,linewidth=2,   basey=10,  color ='black', label='__nolabel__')
plt.legend(loc=0)
plt.xlabel(r'$\kappa\left(\frac{L}{2\pi} \right)$',fontsize=20)
plt.ylabel(r'$E_{q_t}\left(\kappa\right)$',fontsize=20)

plt.ylim(1e-8,1e3)




plt.figure(4)
plt.title('Total water--entropy covariance spectra')
count = 1
for k in np.arange(80,90,20):
    plt.semilogy(kappa, E_SQT[k,:]/(dk*np.sum(E_SQT[k,:])),   basey=10,label = '7m7s, ILES', color = my_colors[count] )
    count +=1
plt.semilogy(kappa, k53*1000.0,linewidth=2,   basey=10,  color ='black', label='__nolabel__')
plt.legend(loc=0)
plt.xlabel(r'$\kappa\left(\frac{L}{2\pi} \right)$',fontsize=20)
plt.ylabel(r'$E_{s,q_t}\left(\kappa\right)$',fontsize=20)

plt.ylim(1e-8,1e3)





file_plus_path = '/Volumes/data5/Kyle/spectra/CondStats.DYCOMS_RF01_order_29.nc'


dx = 35.0
dz = 5.0
nx = 96
dk = 2.0*np.pi/(dx*nx)


data = nc.Dataset(file_plus_path, 'r' )

spectra = data.groups['spectra']

z = spectra.variables['z'][:]

kappa = spectra.variables['wavenumber'][:]/dk

E_TKE = spectra.variables['energy_spectrum'][23,:,:]

E_QT = spectra.variables['qt_spectrum'][23,:,:]

E_S = spectra.variables['s_spectrum'][23,:,:]
E_SQT = spectra.variables['s_qt_cospectrum'][23,:,:]


k53 = kappa**(-5.0/3.0)

plt.figure(1)
plt.title('Energy spectra')
count = 2
for k in np.arange(80,90,20):
    plt.semilogy(kappa, E_TKE[k,:]/(dk*np.sum(E_TKE[k,:])),   basey=10,label ='9m9s, ILES', color = my_colors[count] )
    count +=1
plt.semilogy(kappa, k53*1000.0,linewidth=2,   basey=10, color ='black', label='__nolabel__')
plt.legend(loc=0)
plt.xlabel(r'$\kappa\left(\frac{L}{2\pi} \right)$',fontsize=20)
plt.ylabel(r'$E\left(\kappa\right)$',fontsize=20)
plt.ylim(1e-8,1e3)

plt.figure(2)
plt.title('Entropy variance spectra')
count = 2
for k in np.arange(80,90,20):
    plt.semilogy(kappa, E_S[k,:]/(dk*np.sum(E_S[k,:])),   basey=10,label = '9m9s, ILES', color = my_colors[count] )
    count +=1
plt.semilogy(kappa, k53*1000.0,linewidth=2,   basey=10,  color ='black', label='__nolabel__')
plt.legend(loc=0)
plt.xlabel(r'$\kappa\left(\frac{L}{2\pi} \right)$',fontsize=20)
plt.ylabel(r'$E_s\left(\kappa\right)$',fontsize=20)
plt.ylim(1e-8,1e3)

plt.figure(3)
plt.title('Total water variance spectra')
count = 2
for k in np.arange(80,90,20):
    plt.semilogy(kappa, E_QT[k,:]/(dk*np.sum(E_QT[k,:])),   basey=10,label = '9m9s, ILES', color = my_colors[count] )
    count +=1
plt.semilogy(kappa, k53*1000.0,linewidth=2,   basey=10,  color ='black', label='__nolabel__')
plt.legend(loc=0)
plt.xlabel(r'$\kappa\left(\frac{L}{2\pi} \right)$',fontsize=20)
plt.ylabel(r'$E_{q_t}\left(\kappa\right)$',fontsize=20)

plt.ylim(1e-8,1e3)




plt.figure(4)
plt.title('Total water--entropy covariance spectra')
count = 2
for k in np.arange(80,90,20):
    plt.semilogy(kappa, E_SQT[k,:]/(dk*np.sum(E_SQT[k,:])),   basey=10,label = '9m9s, ILES', color = my_colors[count] )
    count +=1
plt.semilogy(kappa, k53*1000.0,linewidth=2,   basey=10,  color ='black', label='__nolabel__')
plt.legend(loc=0)
plt.xlabel(r'$\kappa\left(\frac{L}{2\pi} \right)$',fontsize=20)
plt.ylabel(r'$E_{s,q_t}\left(\kappa\right)$',fontsize=20)

plt.ylim(1e-8,1e3)


file_plus_path = '/Users/ckaul/Desktop/LES_results/DYCOMS_RF01/IC_res/CondStats.DYCOMS_RF01_order_9_IC.nc'


dx = 35.0
dz = 5.0
nx = 96
dk = 2.0*np.pi/(dx*nx)

data = nc.Dataset(file_plus_path, 'r' )

spectra = data.groups['spectra']

z = spectra.variables['z'][:]

kappa = spectra.variables['wavenumber'][:]/dk

E_TKE = spectra.variables['energy_spectrum'][23,:,:]

E_QT = spectra.variables['qt_spectrum'][23,:,:]

E_S = spectra.variables['s_spectrum'][23,:,:]
E_SQT = spectra.variables['s_qt_cospectrum'][23,:,:]


k53 = kappa**(-5.0/3.0)

plt.figure(1)
plt.title('Energy spectra')
count = 3
for k in np.arange(80,90,20):
    plt.semilogy(kappa, E_TKE[k,:]/(dk*np.sum(E_TKE[k,:])),   basey=10,label ='9m9s, ELES', color = my_colors[count] )
    count +=1
plt.semilogy(kappa, k53*1000.0,linewidth=2,   basey=10, color ='black', label='__nolabel__')
plt.legend(loc=0)
plt.xlabel(r'$\kappa\left(\frac{L}{2\pi} \right)$',fontsize=20)
plt.ylabel(r'$E\left(\kappa\right)$',fontsize=20)
plt.ylim(1e-8,1e3)

plt.figure(2)
plt.title('Entropy variance spectra')
count = 3
for k in np.arange(80,90,20):
    plt.semilogy(kappa, E_S[k,:]/(dk*np.sum(E_S[k,:])),   basey=10,label = '9m9s, ELES', color = my_colors[count] )
    count +=1
plt.semilogy(kappa, k53*1000.0,linewidth=2,   basey=10,  color ='black', label='__nolabel__')
plt.legend(loc=0)
plt.xlabel(r'$\kappa\left(\frac{L}{2\pi} \right)$',fontsize=20)
plt.ylabel(r'$E_s\left(\kappa\right)$',fontsize=20)
plt.ylim(1e-8,1e3)

plt.figure(3)
plt.title('Total water variance spectra')
count = 3
for k in np.arange(80,90,20):
    plt.semilogy(kappa, E_QT[k,:]/(dk*np.sum(E_QT[k,:])),   basey=10,label = '9m9s, ELES', color = my_colors[count] )
    count +=1
plt.semilogy(kappa, k53*1000.0,linewidth=2,   basey=10,  color ='black', label='__nolabel__')
plt.legend(loc=0)
plt.xlabel(r'$\kappa\left(\frac{L}{2\pi} \right)$',fontsize=20)
plt.ylabel(r'$E_{q_t}\left(\kappa\right)$',fontsize=20)

plt.ylim(1e-8,1e3)




plt.figure(4)
plt.title('Total water--entropy covariance spectra')
count = 3
for k in np.arange(80,90,20):
    plt.semilogy(kappa, E_SQT[k,:]/(dk*np.sum(E_SQT[k,:])),   basey=10,label = '9m9s, ELES', color = my_colors[count] )
    count +=1
plt.semilogy(kappa, k53*1000.0,linewidth=2,   basey=10,  color ='black', label='__nolabel__')
plt.legend(loc=0)
plt.xlabel(r'$\kappa\left(\frac{L}{2\pi} \right)$',fontsize=20)
plt.ylabel(r'$E_{s,q_t}\left(\kappa\right)$',fontsize=20)

plt.ylim(1e-8,1e3)




plt.show()


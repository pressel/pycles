#!/bin/csh -f

set echo
# sudo port select gcc mp-gcc48

set MPATH_LW = $cwd/lw/modules
set CPATH_LW = $cwd/lw/src
set MPATH_SW = $cwd/sw/modules
set CPATH_SW = $cwd/sw/src
set BPATH = $cwd/rrtmg_build

# The following paths on Euler are current as of Feb 01 2016.
#set LIB_NETCDF = /cluster/apps/netcdf/4.3.1/x86_64/gcc_4.8.2/openmpi_1.6.5/lib
#set INC_NETCDF = /cluster/apps/netcdf/4.3.1/x86_64/gcc_4.8.2/openmpi_1.6.5/include
#set LIB_NETCDF = /opt/local/lib
#set INC_NETCDF = /opt/local/include

# Compile rrtmg_lw
cd ${BPATH}
###########rm -R ./rrobj_rrtmg
mkdir -p obj_rrtmg
cd obj_rrtmg

# build RRTMG_LW
$FC -ffixed-line-length-none -freal-4-real-8 -fno-range-check -fPIC \
  -O3 -I${INC_NETCDF} -L${LIB_NETCDF} -lnetcdf -c \
  ${MPATH_LW}/parkind.f90 ${MPATH_LW}/parrrtm.f90 ${MPATH_LW}/rrlw_cld.f90 ${MPATH_LW}/rrlw_con.f90 \
  ${MPATH_LW}/rrlw_kg01.f90 ${MPATH_LW}/rrlw_kg02.f90 ${MPATH_LW}/rrlw_kg03.f90 ${MPATH_LW}/rrlw_kg04.f90 \
  ${MPATH_LW}/rrlw_kg05.f90 ${MPATH_LW}/rrlw_kg06.f90 ${MPATH_LW}/rrlw_kg07.f90 ${MPATH_LW}/rrlw_kg08.f90 \
  ${MPATH_LW}/rrlw_kg09.f90 ${MPATH_LW}/rrlw_kg10.f90 ${MPATH_LW}/rrlw_kg11.f90 ${MPATH_LW}/rrlw_kg12.f90 \
  ${MPATH_LW}/rrlw_kg13.f90 ${MPATH_LW}/rrlw_kg14.f90 ${MPATH_LW}/rrlw_kg15.f90 ${MPATH_LW}/rrlw_kg16.f90 \
  ${MPATH_LW}/rrlw_ncpar.f90 ${MPATH_LW}/rrlw_ref.f90 ${MPATH_LW}/rrlw_tbl.f90 ${MPATH_LW}/rrlw_vsn.f90 \
  ${MPATH_LW}/rrlw_wvn.f90 \
  ${CPATH_LW}/rrtmg_lw_cldprmc.f90 ${CPATH_LW}/rrtmg_lw_cldprop.f90 ${CPATH_LW}/rrtmg_lw_rtrn.f90 \
  ${CPATH_LW}/rrtmg_lw_rtrnmc.f90 ${CPATH_LW}/rrtmg_lw_rtrnmr.f90 ${CPATH_LW}/rrtmg_lw_setcoef.f90 \
  ${CPATH_LW}/rrtmg_lw_taumol.f90 ${CPATH_LW}/rrtmg_lw_k_g.f90 ${CPATH_LW}/rrtmg_lw_init.f90 \
  ${CPATH_LW}/rrtmg_lw_rad.nomcica.f90 ${BPATH}/rrtmg_lw_wrapper.f90

# build RRTMG_SW - Note that parkind.f90 is the same as RRTMG_LW
$FC -ffixed-line-length-none -freal-4-real-8 -fno-range-check -fPIC \
  -O3 -I${INC_NETCDF} -L${LIB_NETCDF} -lnetcdf -c \
  ${MPATH_SW}/parkind.f90 ${MPATH_SW}/parrrsw.f90 ${MPATH_SW}/rrsw_aer.f90 ${MPATH_SW}/rrsw_cld.f90 ${MPATH_SW}/rrsw_con.f90 \
  ${MPATH_SW}/rrsw_kg16.f90 ${MPATH_SW}/rrsw_kg17.f90 ${MPATH_SW}/rrsw_kg18.f90 ${MPATH_SW}/rrsw_kg19.f90 \
  ${MPATH_SW}/rrsw_kg20.f90 ${MPATH_SW}/rrsw_kg21.f90 ${MPATH_SW}/rrsw_kg22.f90 ${MPATH_SW}/rrsw_kg23.f90 \
  ${MPATH_SW}/rrsw_kg24.f90 ${MPATH_SW}/rrsw_kg25.f90 ${MPATH_SW}/rrsw_kg26.f90 ${MPATH_SW}/rrsw_kg27.f90 \
  ${MPATH_SW}/rrsw_kg28.f90 ${MPATH_SW}/rrsw_kg29.f90 \
  ${MPATH_SW}/rrsw_ncpar.f90 ${MPATH_SW}/rrsw_ref.f90 ${MPATH_SW}/rrsw_tbl.f90 ${MPATH_SW}/rrsw_vsn.f90 \
  ${MPATH_SW}/rrsw_wvn.f90 \
  ${CPATH_SW}/rrtmg_sw_cldprmc.f90 ${CPATH_SW}/rrtmg_sw_cldprop.f90 ${CPATH_SW}/rrtmg_sw_reftra.f90\
  ${CPATH_SW}/rrtmg_sw_vrtqdr.f90 ${CPATH_SW}/rrtmg_sw_taumol.f90 \
  ${CPATH_SW}/rrtmg_sw_spcvmc.f90 ${CPATH_SW}/rrtmg_sw_spcvrt.f90 ${CPATH_SW}/rrtmg_sw_setcoef.f90 \
  ${CPATH_SW}/rrtmg_sw_k_g.f90 ${CPATH_SW}/rrtmg_sw_init.f90 \
  ${CPATH_SW}/rrtmg_sw_rad.nomcica.f90 ${BPATH}/rrtmg_sw_wrapper.f90
  
ld -r parkind.o \
  parrrtm.o rrlw_cld.o rrlw_con.o \
  rrlw_kg01.o rrlw_kg02.o rrlw_kg03.o rrlw_kg04.o \
  rrlw_kg05.o rrlw_kg06.o rrlw_kg07.o rrlw_kg08.o \
  rrlw_kg09.o rrlw_kg10.o rrlw_kg11.o rrlw_kg12.o \
  rrlw_kg13.o rrlw_kg14.o rrlw_kg15.o rrlw_kg16.o \
  rrlw_ncpar.o rrlw_ref.o rrlw_tbl.o rrlw_vsn.o \
  rrlw_wvn.o \
  parrrsw.o rrsw_aer.o rrsw_cld.o rrsw_con.o \
  rrsw_kg16.o rrsw_kg17.o rrsw_kg18.o rrsw_kg19.o \
  rrsw_kg20.o rrsw_kg21.o rrsw_kg22.o rrsw_kg23.o \
  rrsw_kg24.o rrsw_kg25.o rrsw_kg26.o rrsw_kg27.o \
  rrsw_kg28.o rrsw_kg29.o \
  rrsw_ncpar.o rrsw_ref.o rrsw_tbl.o rrsw_vsn.o \
  rrsw_wvn.o \
  rrtmg_lw_cldprmc.o rrtmg_lw_cldprop.o rrtmg_lw_rtrn.o \
  rrtmg_lw_rtrnmc.o rrtmg_lw_rtrnmr.o rrtmg_lw_setcoef.o \
  rrtmg_lw_taumol.o rrtmg_lw_k_g.o rrtmg_lw_init.o \
  rrtmg_sw_cldprmc.o rrtmg_sw_cldprop.o rrtmg_sw_reftra.o \
  rrtmg_sw_vrtqdr.o rrtmg_sw_taumol.o \
  rrtmg_sw_spcvmc.o rrtmg_sw_spcvrt.o rrtmg_sw_setcoef.o \
  rrtmg_sw_k_g.o rrtmg_sw_init.o \
  rrtmg_lw_rad.nomcica.o rrtmg_lw_wrapper.o\
  rrtmg_sw_rad.nomcica.o rrtmg_sw_wrapper.o \
  -o ../rrtmg_combined.o

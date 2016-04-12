module rrtmg_sw_wrapper

use iso_c_binding, only: c_double, c_int
use parrrsw, only : nbndsw, naerec
use rrtmg_sw_init, only: rrtmg_sw_ini
use rrtmg_sw_rad,  only: rrtmg_sw

implicit none

contains

subroutine c_rrtmg_sw_init(cpdair) bind(c)
    real(c_double), intent(in) :: cpdair
    call rrtmg_sw_ini(cpdair)
end subroutine c_rrtmg_sw_init


subroutine c_rrtmg_sw &
            (ncol    ,nlay    ,icld    ,iaer    , &
             play    ,plev    ,tlay    ,tlev    ,tsfc    , &
             h2ovmr  ,o3vmr   ,co2vmr  ,ch4vmr  ,n2ovmr  ,o2vmr, &
             asdir   ,asdif   ,aldir   ,aldif   , &
             coszen  ,adjes   ,dyofyr  ,scon    , &
             inflgsw ,iceflgsw,liqflgsw,cldfr   , &
             taucld  ,ssacld  ,asmcld  ,fsfcld  , &
             cicewp  ,cliqwp  ,reice   ,reliq   , &
             tauaer  ,ssaaer  ,asmaer  ,ecaer   , &
             swuflx  ,swdflx  ,swhr    ,swuflxc ,swdflxc ,swhrc) bind(c)
             
      integer(c_int), intent(in) :: ncol            ! Number of horizontal columns     
      integer(c_int), intent(in) :: nlay            ! Number of model layers
      integer(c_int), intent(inout) :: icld         ! Cloud overlap method
                                                      !    0: Clear only
                                                      !    1: Random
                                                      !    2: Maximum/random
                                                      !    3: Maximum
      integer(c_int), intent(inout) :: iaer         ! Aerosol option flag
                                                      !    0: No aerosol
                                                      !    6: ECMWF method
                                                      !    10:Input aerosol optical 
                                                      !       properties

      real(c_double), intent(in) :: play(ncol,nlay)          ! Layer pressures (hPa, mb)
                                                      !    Dimensions: (ncol,nlay)
      real(c_double), intent(in) :: plev(ncol,nlay+1)          ! Interface pressures (hPa, mb)
                                                      !    Dimensions: (ncol,nlay+1)
      real(c_double), intent(in) :: tlay(ncol,nlay)          ! Layer temperatures (K)
                                                      !    Dimensions: (ncol,nlay)
      real(c_double), intent(in) :: tlev(ncol,nlay+1)          ! Interface temperatures (K)
                                                      !    Dimensions: (ncol,nlay+1)
      real(c_double), intent(in) :: tsfc(ncol)            ! Surface temperature (K)
                                                      !    Dimensions: (ncol)
      real(c_double), intent(in) :: h2ovmr(ncol,nlay)        ! H2O volume mixing ratio
                                                      !    Dimensions: (ncol,nlay)
      real(c_double), intent(in) :: o3vmr(ncol,nlay)         ! O3 volume mixing ratio
                                                      !    Dimensions: (ncol,nlay)
      real(c_double), intent(in) :: co2vmr(ncol,nlay)        ! CO2 volume mixing ratio
                                                      !    Dimensions: (ncol,nlay)
      real(c_double), intent(in) :: ch4vmr(ncol,nlay)        ! Methane volume mixing ratio
                                                      !    Dimensions: (ncol,nlay)
      real(c_double), intent(in) :: n2ovmr(ncol,nlay)        ! Nitrous oxide volume mixing ratio
                                                      !    Dimensions: (ncol,nlay)
      real(c_double), intent(in) :: o2vmr(ncol,nlay)         ! Oxygen volume mixing ratio
                                                      !    Dimensions: (ncol,nlay)
      real(c_double), intent(in) :: asdir(ncol)           ! UV/vis surface albedo direct rad
                                                      !    Dimensions: (ncol)
      real(c_double), intent(in) :: aldir(ncol)           ! Near-IR surface albedo direct rad
                                                      !    Dimensions: (ncol)
      real(c_double), intent(in) :: asdif(ncol)           ! UV/vis surface albedo: diffuse rad
                                                      !    Dimensions: (ncol)
      real(c_double), intent(in) :: aldif(ncol)           ! Near-IR surface albedo: diffuse rad
                                                      !    Dimensions: (ncol)

      integer(c_int), intent(in) :: dyofyr          ! Day of the year (used to get Earth/Sun
                                                      !  distance if adjflx not provided)
      real(c_double), intent(in) :: adjes              ! Flux adjustment for Earth/Sun distance
      real(c_double), intent(in) :: coszen(ncol)          ! Cosine of solar zenith angle
                                                      !    Dimensions: (ncol)
      real(c_double), intent(in) :: scon               ! Solar constant (W/m2)

      integer(c_int), intent(in) :: inflgsw         ! Flag for cloud optical properties
      integer(c_int), intent(in) :: iceflgsw        ! Flag for ice particle specification
      integer(c_int), intent(in) :: liqflgsw        ! Flag for liquid droplet specification

      real(c_double), intent(in) :: cldfr(ncol,nlay)         ! Cloud fraction
                                                      !    Dimensions: (ncol,nlay)
      real(c_double), intent(in) :: taucld(nbndsw,ncol,nlay)      ! In-cloud optical depth
                                                      !    Dimensions: (nbndsw,ncol,nlay)
      real(c_double), intent(in) :: ssacld(nbndsw,ncol,nlay)      ! In-cloud single scattering albedo
                                                      !    Dimensions: (nbndsw,ncol,nlay)
      real(c_double), intent(in) :: asmcld(nbndsw,ncol,nlay)      ! In-cloud asymmetry parameter
                                                      !    Dimensions: (nbndsw,ncol,nlay)
      real(c_double), intent(in) :: fsfcld(nbndsw,ncol,nlay)      ! In-cloud forward scattering fraction
                                                      !    Dimensions: (nbndsw,ncol,nlay)
      real(c_double), intent(in) :: cicewp(ncol,nlay)        ! In-cloud ice water path (g/m2)
                                                      !    Dimensions: (ncol,nlay)
      real(c_double), intent(in) :: cliqwp(ncol,nlay)        ! In-cloud liquid water path (g/m2)
                                                      !    Dimensions: (ncol,nlay)
      real(c_double), intent(in) :: reice(ncol,nlay)         ! Cloud ice effective radius (microns)
                                                      !    Dimensions: (ncol,nlay)
                                                      ! specific definition of reice depends on setting of iceflgsw:
                                                      ! iceflgsw = 0: (inactive)
                                                      !              
                                                      ! iceflgsw = 1: ice effective radius, r_ec, (Ebert and Curry, 1992),
                                                      !               r_ec range is limited to 13.0 to 130.0 microns
                                                      ! iceflgsw = 2: ice effective radius, r_k, (Key, Streamer Ref. Manual, 1996)
                                                      !               r_k range is limited to 5.0 to 131.0 microns
                                                      ! iceflgsw = 3: generalized effective size, dge, (Fu, 1996),
                                                      !               dge range is limited to 5.0 to 140.0 microns
                                                      !               [dge = 1.0315 * r_ec]
      real(c_double), intent(in) :: reliq(ncol,nlay)         ! Cloud water drop effective radius (microns)
                                                      !    Dimensions: (ncol,nlay)
      real(c_double), intent(in) :: tauaer(ncol,nlay,nbndsw)      ! Aerosol optical depth (iaer=10 only)
                                                      !    Dimensions: (ncol,nlay,nbndsw)
                                                      ! (non-delta scaled)      
      real(c_double), intent(in) :: ssaaer(ncol,nlay,nbndsw)      ! Aerosol single scattering albedo (iaer=10 only)
                                                      !    Dimensions: (ncol,nlay,nbndsw)
                                                      ! (non-delta scaled)      
      real(c_double), intent(in) :: asmaer(ncol,nlay,nbndsw)      ! Aerosol asymmetry parameter (iaer=10 only)
                                                      !    Dimensions: (ncol,nlay,nbndsw)
                                                      ! (non-delta scaled)      
      real(c_double), intent(in) :: ecaer(ncol,nlay,naerec)       ! Aerosol optical depth at 0.55 micron (iaer=6 only)
                                                      !    Dimensions: (ncol,nlay,naerec)
                                                      ! (non-delta scaled)      

! ----- Output -----

      real(c_double), intent(out) :: swuflx(ncol,nlay+1)       ! Total sky shortwave upward flux (W/m2)
                                                      !    Dimensions: (ncol,nlay+1)
      real(c_double), intent(out) :: swdflx(ncol,nlay+1)       ! Total sky shortwave downward flux (W/m2)
                                                      !    Dimensions: (ncol,nlay+1)
      real(c_double), intent(out) :: swhr(ncol,nlay)         ! Total sky shortwave radiative heating rate (K/d)
                                                      !    Dimensions: (ncol,nlay)
      real(c_double), intent(out) :: swuflxc(ncol,nlay+1)      ! Clear sky shortwave upward flux (W/m2)
                                                      !    Dimensions: (ncol,nlay+1)
      real(c_double), intent(out) :: swdflxc(ncol,nlay+1)      ! Clear sky shortwave downward flux (W/m2)
                                                      !    Dimensions: (ncol,nlay+1)
      real(c_double), intent(out) :: swhrc(ncol,nlay)        ! Clear sky shortwave radiative heating rate (K/d)
                                                      !    Dimensions: (ncol,nlay)
                                                      
    
    
    call rrtmg_sw &
            (ncol    ,nlay    ,icld    ,iaer    , &    ! idelm added by ZTAN
             play    ,plev    ,tlay    ,tlev    ,tsfc    , &
             h2ovmr  ,o3vmr   ,co2vmr  ,ch4vmr  ,n2ovmr  ,o2vmr, &
             asdir   ,asdif   ,aldir   ,aldif   , &
             coszen  ,adjes   ,dyofyr  ,scon    , &
             inflgsw ,iceflgsw,liqflgsw,cldfr   , &
             taucld  ,ssacld  ,asmcld  ,fsfcld  , &
             cicewp  ,cliqwp  ,reice   ,reliq   , &
             tauaer  ,ssaaer  ,asmaer  ,ecaer   , &
             swuflx  ,swdflx  ,swhr    ,swuflxc ,swdflxc ,swhrc)
end subroutine c_rrtmg_sw


end module
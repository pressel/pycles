cdef extern from 'stats_objects.h':
    struct StatsProfileStruct:
        long dim
        long sum_n
        double* data
    void init_stats_profile_struct(StatsProfileStruct* PS, long dim) nogil
    void stats_profile_reset(StatsProfileStruct* PS) nogil
    void stats_profile_struct_free(StatsProfileStruct* PS) nogil

cdef class ProfileObject:
    cdef:
        StatsProfileStruct profile_struct
        void initialize(self,long dim)
        void reset(self)
        void free(self)
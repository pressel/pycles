
cdef class ProfileObject:
    def __init__(self,dim):
        return

    cdef void initialize(self, long dim):
        init_stats_profile_struct(&self.profile_struct,dim)
        return

    cdef void reset(self):
        stats_profile_reset(&self.profile_struct)
        return

    cdef void free(self):
        stats_profile_struct_free(&self.profile_struct)
        return
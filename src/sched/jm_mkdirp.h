/*
 BSD 3-Clause License
 Copyright (c) 2025, The Regents of the University of California
 See Repository LICENSE file
 */
// Utility for constructing directory path.
// Like unix command  mkdir -p
// Returns 0 on success, -1 with errno set on failure
//
#ifdef __cplusplus
extern "C" {
#endif

// might want to call it from jm_worker.c
// so make it C api
int jm_mkdirp(const char *path);

#ifdef __cplusplus
}
#endif

/*
 * Information worker process needs about being launched from the
 * job manager.
 */
#ifdef __cplusplus
extern "C" {
#endif
extern int jm_active;              /* true if under job manager */
extern const char *jm_parentname;  /* "<self>" if not under job manager */
extern const char *jm_jobname;     /* name of job passed from scheduler */
extern void jm_parent_handshake(int *argcp, char ***argv);  /* call immed after MPI_Init */
extern void jm_finish(int rc, const char *msg);             /* call after MPI_Finalize */
extern void jm_mark_started();     /* used to flag init complete.  */
                                   /* call after initial startup complete.   Sometimes */
								   /* GPU init hangs, home dir nfs glitches, ... and we want to kill it. */

extern int jm_getcpu(void);                   /* get current logical cpu */
extern char *jm_getcwd(void); /* current working directory allocated with new */
extern void jm_freecwd(char *cwd); /* free return from jm_getcwd */

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
#ifdef __cplusplus
}
#endif

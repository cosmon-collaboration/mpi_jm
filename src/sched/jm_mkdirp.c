
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

//
// Back up one step in the path and
// test for existance.
// If not, recurse
// at top, return error and set errno = EACCES
static int PMkDirPSub(char *path) {
	int rc = 0;
	struct stat s;
	char *cp = strrchr(path, '/');
	if(cp) {
		*cp = 0; // remove last segment
		rc = stat(path, &s);
		if(rc < 0) {
			if(errno == ENOENT) {
				// make sure parent path is constructed
				rc = PMkDirPSub(path);
				if(rc < 0) return -1;
			} else {
				rc = -1;
			}
		} else {
			if(S_ISDIR(s.st_mode)) {
				rc = 0;
			} else {
				errno = ENOTDIR;
				rc = -1;
			}
		}
		*cp = '/'; // add last segment back
		if(rc == 0) {
			// We have a parent path, add last segment
			rc = mkdir(path, 0777);
		}
	} else {
		if(path[0] == 0) {
			errno = EACCES;
			rc = -1;
		} else {
			// looks like request for local directory
			// Give it a try.
			rc = mkdir(path, 0777);
		}
	}
	return rc;
}

//
// Make sure the directory path is constructed
// all the way from the root.
//
// Optimize for the case where the directory exists and
// is accessable.
//
// Return values
// -1 on error with errno set
//  0 on success
int jm_mkdirp(const char *path) {
	int rc;
	struct stat s;
	// Test the full path first, it may already exist.
	rc = stat(path, &s);
	if(rc < 0) {
		if (errno == ENOENT) {
			// Have to try to create path
			char *buf = malloc(strlen(path) + 1);
			strcpy(buf, path);
			rc = PMkDirPSub(buf);
			free(buf);
			if(rc == 0) {
				if(access(path, R_OK|W_OK|X_OK) == 0) {
					rc = 0;
				} else {
					errno = EACCES;
					rc = -1;
				}
			}
		} else {
			// errno is set from stat call
			rc = -1;
		}
	} else {
		// make sure it is an accessable (rwx) directory
		// We are using this for constructing directories to
		// dump log files into.
		if(S_ISDIR(s.st_mode)) {
			if(access(path, R_OK|W_OK|X_OK) == 0) {
				rc = 0;
			} else {
				errno = EACCES;
				rc = -1;
			}
		} else {
			errno = ENOTDIR;
			rc = -1;
		}
	}
	return rc;
}

#ifdef TESTMAIN
void main(int argc, char **argv) {
	if(argc != 2) {
		printf("usage: mkdirh <path>\n");
		exit(1);
	}
	int rc = PMkDirP(argv[1]);
	if(rc < 0) {
		const char *estr = strerror(errno);
		fprintf(stderr, "Failed to create path '%s', errno=%d : %s\n", argv[1], errno, estr ? estr : "<nomsg>");
		exit(-1);
	}
	exit(0);
}
#endif

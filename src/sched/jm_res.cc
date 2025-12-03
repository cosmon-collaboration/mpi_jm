/*
 BSD 3-Clause License
 Copyright (c) 2025, The Regents of the University of California
 See Repository LICENSE file
 */
#include "jm_sched.h"

jmres_block_type jmres_block;

//! \brief constructor for block description
jmres_block_type::jmres_block_type() {
	numnodes = 1;
	bnumslots = 0;
	noderescnt = new int[1];
	blockrescnt = new int[1];
	nodememKb = 16 * 1024; // default to 16Gb
	envlist = nullptr;
	// use (1 << resid) to test res_threaded_mask
	res_threaded_mask = 0; // default is no resources are threaded
}

//! \brief destructor for block description
jmres_block_type::~jmres_block_type() {
	resnames.clear();
	delete[] noderescnt;
	delete[] blockrescnt;
}

//! \brief add a resource type like gpu or cpu to a node definition
void jmres_block_type::addres(const char *resname) {
	int i;
	for(i = resnames.size(); --i >= 0;) {
		if(streq(resnames[i], resname)) break;
	}
	if(i < 0) { // not found
		// add space
		i = resnames.size();
		resnames.push_back(jm_mstr(resname));
		int *tmpn = new int[i+1];
		int *tmpb = new int[i+1];
		for(int j = 0; j < i; j++) {
			tmpn[j] = noderescnt[j];
			tmpb[j] = blockrescnt[j];
		}
		delete[] noderescnt;
		delete[] blockrescnt;
		noderescnt = tmpn;
		blockrescnt = tmpb;
		noderescnt[i] = 0;
		blockrescnt[i] = 0;
		// TODO: pass in threading of resource
		if(strcaseeq(resname, "cpu")) {
			res_threaded_mask |= (1u << (unsigned)i);
		}
	}
}

//! \brief get the number of resource types
int jmres_block_type::numres() {
	return resnames.size();
}

//! \brief convert a resource name to the resource id.
int jmres_block_type::getresid(const char *resname) {
	int i;
	for(i = resnames.size(); --i >= 0;) {
		if(streq(resnames[i], resname)) break;
	}
	return i;
}

//! \brief get the number of slots in a node supplying the resource named by resname.
int jmres_block_type::getnoderes(const char *resname) {
	int i = getresid(resname);
	if(i < 0) return 0;
	return noderescnt[i];
}

//! \brief get the number of slots in a block supplying the resource named by resname.
int jmres_block_type::getblockres(const char *resname) {
	int i = getresid(resname);
	if(i < 0) return 0;
	return blockrescnt[i];
}

//! \brief set the default number of nodes in a block.
// A direct argument (-bs #) to jm_sched wins
void jmres_block_type::setnumnodes(int num) {
	numnodes = jm_block_size_arg >= 0 ? jm_block_size_arg : num;
	for(int i = (int)resnames.size(); --i >= 0;) {
		blockrescnt[i] = noderescnt[i] * numnodes;
	}
}
//! \brief get the number of nodes in a block.
int jmres_block_type::getnumnodes() {
	return numnodes;
}
//! \brief convert a resource id to a resource name.
const char *jmres_block_type::getresname(int resid) {
	if(resid < 0 || resid >= (int)resnames.size())
		return NULL;
	return resnames[resid];
}

//! \brief Get the number of slots in a block.
int jmres_block_type::getnodeslots() {
	return slots.size();
}

//! \brief Define a slot that we can assign computation to.  Resources can be  cpu|gpu|numa0|numa1.
int jmres_block_type::addslot(const char *res) {
	int rmask = 0;
	int i = 0;
	const char *cp, *ncp;

	cp = res;
	while(*cp) {
		ncp = cp;
		while(*ncp && *ncp != '|') ncp++;
		int len = ncp - cp;
		char *s = new char[len+1];
		strncpy(s, cp, len);
		s[len] = 0;
		// find resource
		for(i = resnames.size(); --i >= 0;) {
			if(streq(resnames[i], s)) break;
		}
		if(i < 0) {
			printf("machinedef addslot: No resource type '%s'\n", s);
			delete[] s;
			jm_sched_abort();
		}
		rmask |= 1 << i;
		delete[] s;
		cp = ncp;
		if(*cp) cp++;
	}
	if(!rmask) {
		printf("slot with no resources, res='%s'\n", res);
		return -1;
	}
	slots.push_back(rmask);
	return 0;
}

//
// We are done defining resources.  Compute totals
// for nodes and blocks.
//
void jmres_block_type::finish() {
	int m, x, ssize;
	ssize = (int)slots.size();
	// sum each resource by scanning slots
	for(int i = 0; i < ssize; i++) {
		m = slots[i];
		for(x = resnames.size(); --x >= 0;) {
			if(m & (1 << x)) {
				noderescnt[x]++;
			}
		}
	}
	setnumnodes(numnodes); // update blockrescnt
	bnumslots = numnodes * ssize;
	bool err = false;
	for(jm_res_slot_env *ep = envlist; ep; ep=ep->next) {
		if(ep->slot < 0 || ep->slot >= ssize) {
			printf("addslotenv:  slot number %d for %s=%s is out of range [0,%d]\n", ep->slot, ep->name, ep->value, ssize);
			err = true;
		}
		if(strchr(ep->name, '=')) {
			printf("addslotenv:  env name \"%s\" for slot %d may not contain '='\n", ep->name, ep->slot);
			err = true;
		}
	}
	if(err)
		jm_sched_abort();
}

//
// Convert a string like "cpu|gpu" into a bitmask
//
int jmres_block_type::getresmask(const char *reslist) {
	int x;
	int rmask = 0;
	const char *cp, *ncp;
	char buf[40];
	printf("getresmask: reslist=%s\n", reslist);
	cp = reslist;
	while(*cp) {
		ncp = cp;
		while(*ncp && *ncp != '|') ncp++;
		if(cp == ncp) return -1;
		int len = ncp - cp;
		if(len > 32) return -1;
		strncpy(buf, cp, len);
		buf[len] = 0;
		printf("Checking resource %s\n", buf);
		for(x = resnames.size(); --x >= 0;) {
			char *rs = resnames[x];
			printf("comparing to %s\n", rs);
			if(streq(rs, buf)) break;
		}
		if(x < 0) return -1;
		printf("found res=%d\n", x);
		rmask |= (1 << x);
		if(!*ncp) break;
		cp = ncp + 1;
	}
	return rmask;
}

//! \brief package up all slot environment variables as a single buffer.
char *jmres_block_type::getslotenvbuf() {
	int bsize = 1;
	jm_res_slot_env *ep;
	for(ep = envlist; ep; ep=ep->next) {
		bsize += strlen(ep->name) + strlen(ep->value) + 10; // overhead for @%d:%s=%s
	}
	auto *buf = new char[bsize];
	char *cp = buf;
	for(ep = envlist; ep; ep=ep->next) {
		*cp++ = '@';
		sprintf(cp, "%d", ep->slot);
		cp += strlen(cp);
		*cp++ = ':';
		strcpy(cp, ep->name); // we make sure names don't have embedded '='.
		cp += strlen(cp);
		*cp++ = '=';
		strcpy(cp, ep->value);
		cp += strlen(cp);
	}
	*cp = 0;
	return buf;
}

//! \brief constructor for storage of slot based environment variables
jm_res_slot_env::jm_res_slot_env(int aslot, const char *aname, const char *avalue) {
	slot = aslot;
	name = jm_mstr(aname);
	value = jm_mstr(avalue);
	next = nullptr;
}

//! \brief destructor for storage of slot based environment variables
jm_res_slot_env::~jm_res_slot_env() {
	delete[] name;
	delete[] value;
}

//! \brief save environment description for slots.
void jmres_block_type::addslotenv(int slot, const char *name, const char *value) {
	auto *ep = new jm_res_slot_env(slot, name, value);
	printf("Sched: adding slot env %d:%s=%s\n", ep->slot, ep->name, ep->value);
	// remember for jm_res_block_type::finish()
	ep->next = envlist;
	envlist = ep;
}

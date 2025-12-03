/*
 BSD 3-Clause License
 Copyright (c) 2025, The Regents of the University of California
 See Repository LICENSE file
 */
//
// Describes resources in a block.
// nodes are assumed to be uniform.
//
class jm_res_slot_env {
public:
	int slot;
	char *name, *value;
	jm_res_slot_env *next;
	jm_res_slot_env(int slot, const char *name, const char *value);
	~jm_res_slot_env();
};

class jmres_block_type {
private:
	int numnodes;
	int bnumslots; // numnodes * slots.size()
	size_t nodememKb; // node memory in Kb.
	vector<char *> resnames; // indexed by [resid]
	int *noderescnt;
	int *blockrescnt;
	// entries in slots have a bit mask for the resources at that slot.
	vector<int> slots;
	jm_res_slot_env *envlist;
public:
	unsigned res_threaded_mask; // use (1u << resid) to test res_threaded_mask

	jmres_block_type();
	~jmres_block_type();
	void addres(const char *resname);
	int numres();
	int getresid(const char *resname);
	const char *getresname(int resid);
	int getnoderes(const char *resname);
	int getblockres(const char *resname);
	void setnumnodes(int num);
	int getnumnodes();
	int getnodeslots();
	inline int getblockrescnt(int i) { return blockrescnt[i];}
	int addslot(const char *res);
	void finish();
	inline int blockslots() { return bnumslots; }
	int getresmask(const char *resnames);
	void setnodemem(size_t gb) { nodememKb = gb; }
	size_t getnodemem() { return nodememKb; }
	inline int bslot2resmask(int bslot) {
		return slots[bslot % slots.size()];
	}
	void addslotenv(int slot, const char *name, const char *value); // add one slot env
	char *getslotenvbuf(); // get encoded buffer with all slot environment vars
};

//
//  Eventually we may want an array of blocks
//  with different numbers of nodes / resources
//  For now we will assume that they are all the same.
//
extern jmres_block_type jmres_block;
extern int jm_block_size_arg;  // overrides default from machine file

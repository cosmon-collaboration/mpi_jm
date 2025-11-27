//
// jm_lump.cc
// Groups of nodes running jm_master can connect and offer their services.
// We call such groups "Lumps"
//
#include "jm_sched.h"

Lump *jm_lumplist = nullptr; // chain of all registered lumps

//! \brief Create Lump for block of nodes connecting via accept/connect
//! \param alumpname Name of lump, used for private connection
//! \param tmpcomm Global comm for introduction.
Lump::Lump(char *alumpname, MPI_Comm tmpcomm) {
	lockval = 0;

	// keep track of all lumps as they register
	Lump **lpp;
	lpp = &jm_lumplist;
	while(*lpp) lpp = &(*lpp)->next;
	*lpp = this;
	next = nullptr;

	block_link_comm = nullptr;
	block_link_str = nullptr;
	bic_rank = nullptr;
	parentlump = false;

	MPI_Status status; 
	lumpname = jm_mstr(alumpname);
	printf("Server: reading lumpsize\n");
	// read lump size on original connection.
	MPI_Recv(&lumpsize, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, tmpcomm, &status);
	printf("Scheduler: received lumpsize=%d from lump %s\n", lumpsize, lumpname);
	int nnodes = jmres_block.getnumnodes();
	block_count = lumpsize / nnodes;
	if(block_count <= 0) {
		printf("number of nodes %d is less than the specified block size %d\n", lumpsize, nnodes);
		jm_sched_abort();
	}

	// each block will get it's own private comm
	MakeBlockLinks(tmpcomm);
	printf("disconnecting introduction port for Lump %s\n", lumpname);
	MPI_Comm_disconnect(&tmpcomm); // not needed anymore

	CreateMsg();
}

//! \brief Create Lump corresonding to the initial jm_master launch that creates jm_sched.  
Lump::Lump(MPI_Comm tmpcomm) { // Used to track Lump created in initial launch of jm_master
	lockval = 0;
	// keep track of all lumps as they register
	Lump **lpp;
	lpp = &jm_lumplist;
	while(*lpp) lpp = &(*lpp)->next;
	*lpp = this;
	next = nullptr;
	parentlump = true;

	MPI_Status status; 
	lumpname = jm_mstr("parent"); // This lump is from the parent
	// read lump size on original connection.
	printf("Server: reading lumpsize for %s\n", lumpname);
	MPI_Recv(&lumpsize, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, tmpcomm, &status);
	printf("Scheduler: received lumpsize=%d from lump %s\n", lumpsize, lumpname);
	int nnodes = jmres_block.getnumnodes();
	block_count = lumpsize / nnodes;
	if(block_count <= 0) {
		printf("number of nodes %d is less than the specified block size %d\n", lumpsize, nnodes);
		jm_sched_abort();
	}

	// each block will get it's own private comm
	MakeBlockLinks(tmpcomm);

	CreateMsg();
}

//! \brief Let Lump know what the machine parameters are.
void Lump::SendMachineParameters() {
	printf("Sched: sending machine parameters to %s\n", lumpname);
	MPI_Comm b0comm = block_link_comm[0]; // use first block
	block_count = lumpsize / jmres_block.getnumnodes(); // number of blocks in this Lump
	// We can set environment variables on a slot basis to help with GPU/other binding
	char *slotenvbuf = jmres_block.getslotenvbuf(); // returns alloced buffer 

	int mparms[JM_MACH_PARMS_SIZE]; // currently size 8
	mparms[0] = JM_MACH_PARMS_VERSION;
	mparms[1] = jmres_block.getnumnodes();    // number of nodes in block
	mparms[2] = jmres_block.getnodeslots();   // # of cpu slots
	mparms[3] = strlen(slotenvbuf) + 1;       // include NUL at end
	MPI_Send(mparms, JM_MACH_PARMS_SIZE, MPI_INT, 0, 0, b0comm);
	printf("Sched: sending slotenvbuf: '%s'\n", slotenvbuf);
	MPI_Send(slotenvbuf, mparms[3], MPI_CHAR, 0, 0, b0comm);
	delete[] slotenvbuf;
	printf("Sched: sent machine parameters to lump %s\n", lumpname);
}

//! \brief Disconnect intercomms to lump of jm_masters
void Lump::Disconnect() {
	for(int bid = 0; bid < block_count; bid++) {
		// make sure we disconnect only once.  First Lump may share intercomms
		MPI_Comm c = block_link_comm[bid];
		if(c == MPI_COMM_NULL) continue;
		for(int xid = bid; xid < block_count; xid++) {
			if(c == block_link_comm[xid])
				block_link_comm[xid] = MPI_COMM_NULL;
		}
		MPI_Comm_disconnect(&c);
	}
}

//! \brief Send machine parmeters to all lumps of nodes
// Also sets up jm_block_use array to point back to Lumps with block offsets
void JmSendMachineParameters() {
	jm_block_count = 0;
	for(Lump *lp = jm_lumplist; lp; lp=lp->next) {
		jm_block_count += lp->BlockCount();
	}
	jm_block_use = new jm_block_use_t[jm_block_count];
	// now point each block at a Lump.
	int pos = 0;
	for(Lump *lp = jm_lumplist; lp; lp=lp->next) {
		int lbc = lp->BlockCount();
		for(int i = 0; i < lbc; i++) {
			jm_block_use[pos+i].lp = lp;  // Lump containing block
			jm_block_use[pos+i].lbid = i; // block id in Lump
		}
		pos += lbc;
	}
	for(int i = 0; i < jm_block_count; i++) {
		printf("Sched: block %d in lump %s, lbid=%d\n", i, jm_block_use[i].lp->Name(), jm_block_use[i].lbid);
	}
	for(Lump *lp = jm_lumplist; lp; lp=lp->next) {
		lp->SendMachineParameters();
	}
}

//
// We sit in JmAcceptLumps until all lumps are ready.
// Alternatively, AcceptBlocks could run in rank 1 and
// add blocks into the sytem.   This would work by
// sending a message to the rank 0 process to publish
// a lump specific name.   Once that is done, then
// JmAcceptLumps in rank 1 can reply and an accept/connect
// pair for the block rank 0 processes and jm_sched rank 0
// can happen.
//
void JmAcceptLumps() {
    MPI_Comm client;
    MPI_Status status;
    char lumpname[LUMPNAMESIZE];
    int collecting = true;
    const char *collectname = "mpijm";
	char collect_port[MPI_MAX_PORT_NAME];

	// create central port for accepting lumps
    MPI_Info portinfo;
    MPI_Info_create(&portinfo);
    MPI_Open_port(portinfo, collect_port);
    MPI_Info_free(&portinfo);
    printf("Server: lump collection port=%s\n", collect_port);

	// Now publish it so that lumps know how to connect
	MPI_Info publishinfo;
    MPI_Info_create(&publishinfo);
    MPI_Info_set(publishinfo, "ompi_global_scope", "true");
    MPI_Publish_name(collectname, publishinfo, collect_port);
    printf("Server: published name %s=%s\n", collectname, collect_port);

    while(collecting) {
        // need to treat as critical region.
        MPI_Comm_accept(collect_port, MPI_INFO_NULL, 0, MPI_COMM_SELF, &client);
        printf("Lump has asked for introduction, receiving name for permanant accept/connect\n");
        MPI_Recv(lumpname, LUMPNAMESIZE, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, client, &status);

        printf("receive completed, tag=%d\n", (int)status.MPI_TAG);
        switch(status.MPI_TAG) {
        case 0:
			// this is used to signal end of lumps
            MPI_Comm_disconnect(&client);
            collecting = false;
            break;
        case 1:
            printf("Server: accepted connection from %s\n", lumpname);
			new Lump(lumpname, client );
            break;
        default:
            printf("Unknown message tag=%d\n", (int)status.MPI_TAG);
            break;
        }
    }
	printf("Server: Done accepting lumps\n");
}

//
//! \brief Create ports and perform accepts of connections from jm_master blocks
//  Also records comm and rank info for later communications
//
void Lump::MakeBlockLinks(MPI_Comm tmpcomm) {
	char link_port[MPI_MAX_PORT_NAME];

	printf("Server: Making links to block root nodes, block_count=%d\n", block_count);
	// now connect up blocks one at a time.
	block_link_str = new char*[block_count];
	block_link_comm = new MPI_Comm[block_count];
	bic_rank = new int[block_count];

	int nodesperblock = jmres_block.getnumnodes();
	printf("Server: sending nodesperblock\n");
	MPI_Send(&nodesperblock, 1, MPI_INT, 0, 1, tmpcomm);

	printf("Server: opening ports for each block\n");
	for(int bid = 0; bid < block_count; bid++) {
		//MPI_Info portinfo;
		//MPI_Info_create(&portinfo);
		// int rc = MPI_Open_port(portinfo, link_port); // create port and get info to connect with in link_port
		int rc = MPI_Open_port(MPI_INFO_NULL, link_port); // create port and get info to connect with in link_port
		//MPI_Info_free(&portinfo);
		printf("Server: Open port returned\n");
		printf("Server: link_port=%s\n", link_port);
		if(rc != MPI_SUCCESS)
			printf("Server: MPI_Open_port failed!\n");
		block_link_str[bid] = jm_mstr(link_port);
	}

	printf("Server: connect port for each block");
	for(int bid = 0; bid < block_count; bid++){
		strcpy(link_port, block_link_str[bid]);
		MPI_Send(link_port, MPI_MAX_PORT_NAME, MPI_CHAR, 0, 1, tmpcomm);
		MPI_Comm_accept(link_port, MPI_INFO_NULL, 0, MPI_COMM_SELF, &block_link_comm[bid]);
		bic_rank[bid] = 0; // other end uses MPI_COMM_SELF for now
		printf("Accepted %s:B%d\n", lumpname, bid);
	}
	printf("Server: done making block links\n");
}


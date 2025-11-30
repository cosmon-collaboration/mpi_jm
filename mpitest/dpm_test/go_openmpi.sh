#
#

h=`hostname`
echo $h slots=20 > hosts.txt

namefile=nameserver.id

# Nameserver
# in case we start one
newnspid=0
# see if one is hanging around
nspid=`pgrep -u $USER ompi-server`
if [[ ${nspid}x == x ]]
then
	# No nameserver running
	echo "Starting nameserver with file" $namefile
	# --no-daemonize tells it not to fork automatically.  We keep valid pid
	ompi-server --no-daemonize -r $namefile & # forks and runs in background
	newnspid=`pgrep ompi-server`
	echo "Nameserver pid = " ${newnspid}
else
	echo "Using existing nameserver, pid = ", ${nspid}
	ps -ww ${nspid}
fi

/bin/rm -f server.log client1.log client2.log client3.log
mpirun -n 1 -hostfile hosts.txt --oversubscribe -ompi-server file:${namefile} ./server >& server.log &
sleep 1  # so server starts before clients start trying to connect

mpirun -n 1 -hostfile hosts.txt --oversubscribe -ompi-server file:${namefile} ./client -msg "client 1 I got here first" >& client1.log
mpirun -n 1 -hostfile hosts.txt --oversubscribe -ompi-server file:${namefile} ./client -msg "client 2 follows" >& client2.log
# note -done switch, which tells server to shut down
mpirun -n 1 -hostfile hosts.txt --oversubscribe -ompi-server file:${namefile} ./client -done -msg "client 3 quits" >& client3.log

if ((${newnspid} > 0)) 
then
	echo "Killing name server ompi-server, pid=", ${newnspid}
	kill ${newnspid}
fi


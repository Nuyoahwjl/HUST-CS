#include "Server.h"

void main(){
	
	Server srv;
	if(srv.WinsockStartup() == -1) return;
	if(srv.ServerStartup() == -1) return;
	if(srv.ListenStartup() == -1) return;
	if(srv.Loop() == -1) return;
}
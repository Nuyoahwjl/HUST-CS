#pragma once

class WinsockEnv
{
private:
	WinsockEnv(void);
	~WinsockEnv(void);
public:
	static int Startup();
};

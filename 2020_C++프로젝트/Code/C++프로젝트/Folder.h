#ifndef Folder
#define Folder
extern "C"
#include <iostream>
#include <direct.h>		//mkdir
#include <errno.h>	    //errno
#include <string>
using namespace std;

class make_folder {
public:
	static void make_f(char path[]);
	//static void SF_make(int nResult);
	//static char set_newperson(string yn);
};
#endif
#include <iostream>
#include <direct.h>		//mkdir
#include <errno.h>	    //errno
#include <string>
#include "Folder.h"
#include "Console.h"
using namespace std;

//char make_folder::set_newperson(string yn) {
//	if (yn == "yes") {
//		cout << "등록할 이름을 입력하시오>> ";
//		cin >> path;
//		return path[20];
//	}
//	else if (yn == "no") {
//		cout << "등록된 이름을 입력하시오>> ";
//		cin >> path;
//		return path[20];
//	}
//}

extern "C" void make_folder::make_f(char path[]) {
	char a[100] = "../";
	strcat_s(a, 100, path);
	int nResult = _mkdir(a);
	//SF_make(nResult);
	cout << endl;
	cout << "사용자 폴던 : " << a << endl;//저장되는 경로 출력
	//0과 -1저장 0은 만들기 성공,-1은 만들기 실패 
		
	char temp[100] = "";
	strcat_s(temp, 100, a);
	strcat_s(temp, 100, "/문제");//경로에 문제폴더 추가
	_mkdir(temp);
	strcat_s(a, 100, "/오답노트");//경로에 오답노트폴더 추가
	_mkdir(a);
}
//extern "C" void make_folder::SF_make(int nResult) {//생성됐는지 실패했는지 출력
//	//string result="";
//	if (nResult == 0)
//		cout << "폴더 생성 성공\n";
//	else
//		cout << "폴더 생성 실패 - 폴더가 이미 있거나 부정확함\n";//nResult = -1일때
//}

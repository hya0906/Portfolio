#include <iostream>
#include <direct.h>		//mkdir
#include <errno.h>	    //errno
#include <string>
#include "Folder.h"
#include "Console.h"
using namespace std;

//char make_folder::set_newperson(string yn) {
//	if (yn == "yes") {
//		cout << "����� �̸��� �Է��Ͻÿ�>> ";
//		cin >> path;
//		return path[20];
//	}
//	else if (yn == "no") {
//		cout << "��ϵ� �̸��� �Է��Ͻÿ�>> ";
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
	cout << "����� ���� : " << a << endl;//����Ǵ� ��� ���
	//0�� -1���� 0�� ����� ����,-1�� ����� ���� 
		
	char temp[100] = "";
	strcat_s(temp, 100, a);
	strcat_s(temp, 100, "/����");//��ο� �������� �߰�
	_mkdir(temp);
	strcat_s(a, 100, "/�����Ʈ");//��ο� �����Ʈ���� �߰�
	_mkdir(a);
}
//extern "C" void make_folder::SF_make(int nResult) {//�����ƴ��� �����ߴ��� ���
//	//string result="";
//	if (nResult == 0)
//		cout << "���� ���� ����\n";
//	else
//		cout << "���� ���� ���� - ������ �̹� �ְų� ����Ȯ��\n";//nResult = -1�϶�
//}

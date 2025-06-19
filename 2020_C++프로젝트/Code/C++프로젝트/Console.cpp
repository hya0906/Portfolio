#include <iostream>
#include <windows.h>
#include <string>
using namespace std;
#include "Console.h"
#include "text_control.h"
#include "Folder.h"
//string Console::newperson_main() {
//	int check;
//	while (1) {
//		cout << "���ο� ����ڸ� ����ϰڽ��ϱ�?(yes/no)>>";
//		cin >> yn;
//		if (yn == "yes") {
//			cout << "����� �̸��� �Է��Ͻÿ�>> ";
//			cin >> Console::name;
//			make_folder::make_f(name, check);
//			if (check == 0)
//				return name;
//		}
//		else if (yn == "no") {
//			cout << "��ϵ� �̸��� �Է��Ͻÿ�>> ";
//			cin >> name;
//			return name;
//		}
//		else {
//			cout << "�ٽ� �Է��ϼ���" << endl;
//			continue;
//		}
//	}
//}

string Console::newperson_main()
{
	while (1)
	{
		cout << "������� �̸��� �Է��Ͻÿ�" << endl;
		cout << "(������ ������ �Է����ֽ� �̸����� ������ �����˴ϴ�.) >> ";
		cin >> Console::name;
		make_folder::make_f(name);
		return name;
	}
}


void Console::do_review_q_main() {
	bool check = true;
	do {
		string in;
		cout << "1. �޸��� �ҷ�����" << endl;
		cout << "2. �ڷ� ����" << endl;
		cout << "�Է�: ";
		cin >> in;
		if (Console::getint(in) != 0)
		{
			Console::c = getint(in);
			switch (Console::c) {
			case 1:
				system("cls");
				cout << "�޸����� �ҷ��ɴϴ�." << endl;
				return;
			case 2:
				system("cls");
				return;
			default:
				system("cls");
				cout << "�߸��� �Է��Դϴ�. �ٽ� �Է����ּ���" << endl << endl;
				break;
			}
		}
		system("cls");
		cout << "�߸��� �Է��Դϴ�. �ٽ� �Է����ּ���" << endl;
	} while (1);
}
/*
�޸��� �ҷ����� �����ϴ� �޴�
*/

void Console::getMain() {
	do
	{
		string in;
		cout << "1. ���������(������ ���� �Է�)" << endl;
		cout << "2. ���� ���� ����(���� ���� �Է��ϰ� �������� ���� ���)" << endl;
		cout << "3. �����ϱ�(���� ���� �� �����̽� ������ ��)" << endl;
		cout << "4. �����ϱ�" << endl;
		cout << "�Է��Ͻÿ�: ";
		cin >> in;
		if (Console::getint(in) != 0)
		{
			Console::c = getint(in);
			cout << c << endl;
			switch (Console::c) {
			case 1:
				system("cls");
				cout << "<**********������ �� �Է��ϱ�*********>. " << endl;
				return;
			case 2:
				system("cls");
				cout << "<*************���� ���� ����*************>." << endl;
				return;
			case 3:
				system("cls");
				cout << "<*************�����ϱ�*************>." << endl;
				return;
			case 4:
				system("cls");
				cout << "���α׷��� �����մϴ�." << endl;
				return;
			default:
				system("cls");
				cout << "�߸��� �Է��Դϴ�. �ٽ� �Է����ּ���" << endl << endl;
				break;
			}
		}
		system("cls");
		cout << "�߸��� �Է��Դϴ�. �ٽ� �Է����ּ���" << endl << endl;
	} while (1);

}
/*
���� �޴�
Console�� static ������ c�� ������� �Է°�(�޴� ��ȣ)�� ����
*/
void Console::get_text_name(string person)//string���� �ٲ�void����
{
	char temp[100] = "dir ..\\";
	strcat_s(temp, 100, name);
	strcat_s(temp, 100, "\\���� /b");
	cout << "���� ���� ���� �ȿ� �ִ� ����" << endl;
	system(temp);
	cout << endl;
	cout << "�޸����� �̸��� �Է�(.txt���� �Է�) : ";
	cin.ignore(5, '\n');
	while (1) {
		Console::text_name.clear();
		getline(cin, Console::text_name, '\n');
		if (text_control::text_check(Console::text_name, person)) {
			cout << Console::text_name << "(��)��� �޸����� �ҷ��ɴϴ�." << endl;
			c = 1;
			break;
		}

		else {
			system("cls");
			cout << Console::text_name << "(��)��� �޸����� �������� �ʽ��ϴ�. �ٽ� �Է����ּ���." << endl;//���ο� �޸��� ����� ��� �߰�
			cout << "���� ���� ���� �ȿ� �ִ� ����" << endl;
			system(temp);
			cout << "�޸����� �̸�(.txt���� �Է�) : ";
			}
	}
}

string Console::yn = " ";
/*
�޸��� �̸� �Է� �޴�
Console�� static ������ text_name�� ������� �Է°�(�޸����� �̸�)�� ����
*/

void Console::after_test(int num)
{
	system("cls");
	cout << "������ ����Ǿ����ϴ�" << endl << endl << endl;
	cout << "Ʋ�������� " << num << "�� �ֽ��ϴ�" << endl;
	do
	{
		string in;
		cout << "1.�����Ʈ �ۼ��ϱ� " << endl;
		cout << "2.�׳� ���θ޴��� ������" << endl;
		cout << "�Է� : ";
		cin >> in;
		if (Console::getint(in) != 0)
		{
			Console::c = getint(in);
			switch (Console::c) {
			case 1:
				system("cls");
				cout << "****************������ ���� �ԷµǾ����ϴ�**************";
			case 2:
				system("cls");
				cout << "<*************�����Ʈ�� �ۼ��Ǿ����ϴ�*************>." << endl;
				return;
			case 3:
				system("cls");
				return;
			default:
				system("cls");
				cout << "�߸��� �Է��Դϴ�. �ٽ� �Է����ּ���" << endl << endl;
				break;
			}
		}
		system("cls");
		cout << "�߸��� �Է��Դϴ�. �ٽ� �Է����ּ���" << endl << endl;
	} while (1);
}
/*
�����Ʈ �޴�
�����Ʈ ���θ� �Է¹޾� Console�� C�� �ۼ��մϴ�.
*/

string Console::pop_text_name()
{
	return Console::text_name;
}
//Console::text_name�� ����

int Console::pop_user_put()
{
	return Console::c;
}
//Console::c�� ����

int Console::c = 0;
string Console::text_name = " ";
char Console::name[] = { ' ' };
void Console::make_q_main() {
	bool check = true;
	do {
		string in;
		cout << "1. ���� ������ �޸��� �ҷ�����(���ڵ�:ANSI �ʼ�!)" << endl;
		cout << "2. �޸��� ���� ���� �����ϱ�" << endl;
		cout << "3. �ڷ� ����" << endl;
		cout << "�Է�: ";
		cin >> in;
		if (Console::getint(in) != 0)
		{
			Console::c = getint(in);
			switch (Console::c) {
			case 1:
				system("cls");
				cout << "�޸����� �ҷ��ɴϴ�." << endl;
				return;
			case 2:
				system("cls");
				cout << "�޸����� ���� ����ϴ�." << endl;
				return;
			case 3:
				system("cls");
				return;
			default:
				system("cls");
				cout << "�߸��� �Է��Դϴ�. �ٽ� �Է����ּ���" << endl << endl;
				break;
			}
		}
		system("cls");
		cout << "�߸��� �Է��Դϴ�. �ٽ� �Է����ּ���" << endl << endl;
	} while (1);
}

int Console::getint(string in)
{
	return atoi(in.data());
}
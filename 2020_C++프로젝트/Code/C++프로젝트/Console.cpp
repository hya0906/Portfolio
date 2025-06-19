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
//		cout << "새로운 사용자를 등록하겠습니까?(yes/no)>>";
//		cin >> yn;
//		if (yn == "yes") {
//			cout << "등록할 이름을 입력하시오>> ";
//			cin >> Console::name;
//			make_folder::make_f(name, check);
//			if (check == 0)
//				return name;
//		}
//		else if (yn == "no") {
//			cout << "등록된 이름을 입력하시오>> ";
//			cin >> name;
//			return name;
//		}
//		else {
//			cout << "다시 입력하세요" << endl;
//			continue;
//		}
//	}
//}

string Console::newperson_main()
{
	while (1)
	{
		cout << "사용자의 이름을 입력하시오" << endl;
		cout << "(파일이 없으면 입력해주신 이름으로 파일이 생성됩니다.) >> ";
		cin >> Console::name;
		make_folder::make_f(name);
		return name;
	}
}


void Console::do_review_q_main() {
	bool check = true;
	do {
		string in;
		cout << "1. 메모장 불러오기" << endl;
		cout << "2. 뒤로 가기" << endl;
		cout << "입력: ";
		cin >> in;
		if (Console::getint(in) != 0)
		{
			Console::c = getint(in);
			switch (Console::c) {
			case 1:
				system("cls");
				cout << "메모장을 불러옵니다." << endl;
				return;
			case 2:
				system("cls");
				return;
			default:
				system("cls");
				cout << "잘못된 입력입니다. 다시 입력해주세요" << endl << endl;
				break;
			}
		}
		system("cls");
		cout << "잘못된 입력입니다. 다시 입력해주세요" << endl;
	} while (1);
}
/*
메모장 불러올지 결정하는 메뉴
*/

void Console::getMain() {
	do
	{
		string in;
		cout << "1. 문제만들기(문제와 답을 입력)" << endl;
		cout << "2. 모의 시험 보기(문제 수를 입력하고 무작위로 문제 출력)" << endl;
		cout << "3. 복습하기(문제 나온 후 스페이스 누르면 답)" << endl;
		cout << "4. 종료하기" << endl;
		cout << "입력하시오: ";
		cin >> in;
		if (Console::getint(in) != 0)
		{
			Console::c = getint(in);
			cout << c << endl;
			switch (Console::c) {
			case 1:
				system("cls");
				cout << "<**********문제와 답 입력하기*********>. " << endl;
				return;
			case 2:
				system("cls");
				cout << "<*************모의 시험 보기*************>." << endl;
				return;
			case 3:
				system("cls");
				cout << "<*************복습하기*************>." << endl;
				return;
			case 4:
				system("cls");
				cout << "프로그램을 종료합니다." << endl;
				return;
			default:
				system("cls");
				cout << "잘못된 입력입니다. 다시 입력해주세요" << endl << endl;
				break;
			}
		}
		system("cls");
		cout << "잘못된 입력입니다. 다시 입력해주세요" << endl << endl;
	} while (1);

}
/*
메인 메뉴
Console의 static 변수인 c에 사용자의 입력값(메뉴 번호)을 삽입
*/
void Console::get_text_name(string person)//string으로 바꿈void에서
{
	char temp[100] = "dir ..\\";
	strcat_s(temp, 100, name);
	strcat_s(temp, 100, "\\문제 /b");
	cout << "현재 문제 파일 안에 있는 문제" << endl;
	system(temp);
	cout << endl;
	cout << "메모장의 이름을 입력(.txt없이 입력) : ";
	cin.ignore(5, '\n');
	while (1) {
		Console::text_name.clear();
		getline(cin, Console::text_name, '\n');
		if (text_control::text_check(Console::text_name, person)) {
			cout << Console::text_name << "(이)라는 메모장을 불러옵니다." << endl;
			c = 1;
			break;
		}

		else {
			system("cls");
			cout << Console::text_name << "(이)라는 메모장은 존재하지 않습니다. 다시 입력해주세요." << endl;//새로운 메모장 만드는 기능 추가
			cout << "현재 문제 파일 안에 있는 문제" << endl;
			system(temp);
			cout << "메모장의 이름(.txt없이 입력) : ";
			}
	}
}

string Console::yn = " ";
/*
메모장 이름 입력 메뉴
Console의 static 변수인 text_name에 사용자의 입력값(메모장의 이름)을 삽입
*/

void Console::after_test(int num)
{
	system("cls");
	cout << "시험이 종료되었습니다" << endl << endl << endl;
	cout << "틀린문제가 " << num << "개 있습니다" << endl;
	do
	{
		string in;
		cout << "1.오답노트 작성하기 " << endl;
		cout << "2.그냥 메인메뉴로 나가기" << endl;
		cout << "입력 : ";
		cin >> in;
		if (Console::getint(in) != 0)
		{
			Console::c = getint(in);
			switch (Console::c) {
			case 1:
				system("cls");
				cout << "****************문제와 답이 입력되었습니다**************";
			case 2:
				system("cls");
				cout << "<*************오답노트가 작성되었습니다*************>." << endl;
				return;
			case 3:
				system("cls");
				return;
			default:
				system("cls");
				cout << "잘못된 입력입니다. 다시 입력해주세요" << endl << endl;
				break;
			}
		}
		system("cls");
		cout << "잘못된 입력입니다. 다시 입력해주세요" << endl << endl;
	} while (1);
}
/*
오답노트 메뉴
오답노트 여부를 입력받아 Console의 C에 작성합니다.
*/

string Console::pop_text_name()
{
	return Console::text_name;
}
//Console::text_name을 리턴

int Console::pop_user_put()
{
	return Console::c;
}
//Console::c를 리턴

int Console::c = 0;
string Console::text_name = " ";
char Console::name[] = { ' ' };
void Console::make_q_main() {
	bool check = true;
	do {
		string in;
		cout << "1. 내용 저장할 메모장 불러오기(인코딩:ANSI 필수!)" << endl;
		cout << "2. 메모장 새로 만들어서 저장하기" << endl;
		cout << "3. 뒤로 가기" << endl;
		cout << "입력: ";
		cin >> in;
		if (Console::getint(in) != 0)
		{
			Console::c = getint(in);
			switch (Console::c) {
			case 1:
				system("cls");
				cout << "메모장을 불러옵니다." << endl;
				return;
			case 2:
				system("cls");
				cout << "메모장을 새로 만듭니다." << endl;
				return;
			case 3:
				system("cls");
				return;
			default:
				system("cls");
				cout << "잘못된 입력입니다. 다시 입력해주세요" << endl << endl;
				break;
			}
		}
		system("cls");
		cout << "잘못된 입력입니다. 다시 입력해주세요" << endl << endl;
	} while (1);
}

int Console::getint(string in)
{
	return atoi(in.data());
}
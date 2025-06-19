#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <string>
#include <vector>
using namespace std;
#include "Console.h"
#include "text_control.h"
#include "quizzes.h"
#include "Folder.h"

int main(void)
{
	quizzes a;
	vector<string> temp_problem, temp_answer;
	string person = Console::newperson_main();
	do
	{
		Console::getMain();
		switch (Console::pop_user_put())
		{
		case 1://문제만들기
			Console::make_q_main();//1번메뉴불러오기
			if (Console::pop_user_put() == 3)//2는 뒤로가기
				break;
			
			if (Console::pop_user_put() == 1)
			{
				Console::get_text_name(person);
				text_control::make_test(Console::text_name, person);
			}
			else if (Console::pop_user_put() == 2)
			{
				cin.ignore(10, '\n');
				cout << "문제의 이름 입력 : ";
				getline(cin, Console::text_name, '\n');
				text_control::make_test(Console::pop_text_name(), person);
			}
			break;
		case 2://모의시험풀기
			Console::do_review_q_main();//2번메뉴불러오기
			if (Console::pop_user_put() == 2)//2는 뒤로가기
				break;
			Console::get_text_name(person);
			text_control::read_text(Console::pop_text_name(), a.problem, a.answer, person);
			a.test_quizzes(temp_problem, temp_answer);
			Console::after_test(int(temp_problem.size()));
			if (Console::pop_user_put() == 2)
				break;

			text_control::make_text(Console::pop_text_name(), temp_problem, temp_answer, person);
			break;
		case 3://문제리뷰하기
			Console::do_review_q_main();
			if (Console::pop_user_put() == 2)
				break;
			Console::get_text_name(person);
			text_control::read_text(Console::pop_text_name(), a.problem, a.answer, person);
			a.show_quizzes();
			break;
		case 4://끝내기
			return 0;
		}
		a.problem.clear();
		a.answer.clear();

	} while (1);

	return 0;
}
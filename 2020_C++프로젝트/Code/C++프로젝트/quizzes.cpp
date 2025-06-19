#include <iostream>	
#include <vector>
#include <string>
#include <conio.h>
#include <cstdlib>
#include <ctime>
using namespace std;

#include "quizzes.h"

void quizzes::show_quizzes()
{
	cout << "(저장된 문제 수 :" << problem.size() << ")" << endl;
	for (int i = 0; i < int(this->problem.size()); i++)
	{
		cout << "문제 : " << problem[i] << endl;
		_getch();
		cout << "정답 : " << answer[i] << endl << endl;
		_getch();
	}
	cout << endl;
}

void quizzes::test_quizzes(vector<string>& xproblem, vector<string>& xanswer)
{
	vector<string> temp_problem = this->problem;
	vector<string> temp_answer = this->answer;
	vector<string>::iterator it;
	string input;
	xproblem.clear();
	xanswer.clear();
	srand((unsigned)time(0));
	cout << endl;
	cout << "**********시험 시작**********" << endl;
	do
	{
		if (temp_problem.empty())
			break;
		cout << "(남은 문제 : " << temp_problem.size() << ")" << endl;
		int num = rand();
		num = num % int(temp_problem.size());
		cout << "문제 : " << temp_problem[num] << endl;
		cout << "정답 : ";
		getline(cin, input, '\n');
		if (temp_answer[num] != input)
		{
			xproblem.push_back(temp_problem[num]);
			xanswer.push_back(temp_answer[num]);
		}
		it = temp_problem.begin();
		it += num;
		temp_problem.erase(it);
		it = temp_answer.begin();
		it += num;
		temp_answer.erase(it);

	} while (1);
	cout << endl;
}

/*
참조로 벡터 2개를 받습니다 -> 틀린 문제를 저장하는 벡터
test_quizzes는 quizzes의 객체가 가지고 있는 problem과 answer을 랜덤으로 출력,입력
틀린 문제는 위의 참조 변수인 xproblem과 xanswer에 저장합니다.
*/
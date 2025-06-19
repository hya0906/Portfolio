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
	cout << "(����� ���� �� :" << problem.size() << ")" << endl;
	for (int i = 0; i < int(this->problem.size()); i++)
	{
		cout << "���� : " << problem[i] << endl;
		_getch();
		cout << "���� : " << answer[i] << endl << endl;
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
	cout << "**********���� ����**********" << endl;
	do
	{
		if (temp_problem.empty())
			break;
		cout << "(���� ���� : " << temp_problem.size() << ")" << endl;
		int num = rand();
		num = num % int(temp_problem.size());
		cout << "���� : " << temp_problem[num] << endl;
		cout << "���� : ";
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
������ ���� 2���� �޽��ϴ� -> Ʋ�� ������ �����ϴ� ����
test_quizzes�� quizzes�� ��ü�� ������ �ִ� problem�� answer�� �������� ���,�Է�
Ʋ�� ������ ���� ���� ������ xproblem�� xanswer�� �����մϴ�.
*/
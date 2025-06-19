#ifndef QUIZZES
#define QUIZZES
#include <iostream>
#include <vector>
#include <string>
using namespace std;

class quizzes
{
public:
	vector<string>  problem;
	vector<string> answer;
	void show_quizzes();
	void test_quizzes(vector<string>& xproblem, vector<string>& xanswer);
};

#endif 
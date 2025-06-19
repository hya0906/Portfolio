#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "text_control.h"
using namespace std;


void text_control::read_text(string name, vector<string> &problem, vector<string> &answer, string person)
{
	fstream fin;//
	string text_name = "../" + person + "/문제/" + name + ".txt";// ../문제/C++.txt
	cout << person << "님의 " << name<<"에 대한 복습하기입니다." << endl;//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	string line;
	fin.open(text_name, ios::in);
	fin.imbue(locale("kor"));
	if (fin.is_open())
	{
		while (getline(fin, line))
		{
			if (problem.size() > answer.size())
				answer.push_back(line);
			else
				problem.push_back(line);
		}
	}
	else
		cout << "존재하지 않는 문서입니다" << endl << endl;
	fin.close();
}
//name이라는 매개변수를 이용하여 상위 폴더인 '문제'에서
//텍스트를 읽어와서 매개변수 벡터(problem과 answer)에 문제와 정답을 입력
void text_control::make_text(string name, vector<string> problem, vector<string> answer, string person)
{
	fstream fout;
	string text_name = "../" + person + "/오답노트/" + name + ".txt";
	cout << text_name<<endl;//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	fout.open(text_name, ios::out);
	char temp[] = "\n";
	for (int i = 0; i < int(problem.size()); i++)
	{
		fout.write(problem.at(i).c_str(), problem.at(i).size());
		fout.write(temp, 1);//'\n' 입력으로 띄어쓰기
		fout.write(answer.at(i).c_str(), answer.at(i).size());
		fout.write(temp, 1);
	}
	fout.close();
}
//name이라는 매개변수를 텍스트파일의 이름으로하고, 벡터(problem과 answer)의 문제와 정답을 바탕으로
//상위 폴더인 '오답노트'에 오답노트를 작성
bool text_control::text_check(string name, string person)
{
	fstream fin;
	string text_name = "../" + person + "/문제/" + name + ".txt";
	fin.open(text_name, ios::in);
	if (fin.is_open())
	{
		fin.close();
		return true;
	}
	else
	{
		fin.close();
		return false;
	}

}
//텍스트 파일이 있는지를 확인 존재하면 true , 아니면 false를 리턴

void text_control::make_test(string name, string person) { //메모장이 없을경우에는 만들고 문제를 저장하는 기능
	cout << "문제저장을 그만할 경우 문제부분에서 exit를 입력해주세요" << endl;
	string text_name = "../" + person + "/문제/" + name + ".txt";
	ofstream fout(text_name, ios::out | ios::app);
	vector<string> problem, answer;
	while (1) {
		cout << "\n문제>> ";
		getline(cin, p);//p,a값계속바뀜
		if (p == "exit")
			break;
		cout << "답>>";
		getline(cin, a);
		problem.push_back(p);//값계속넣음
		answer.push_back(a);
		count++;
	}
	for (int i = 0; i < count; i++) {//메모장에 내용넣음
		fout << problem[i] << endl;
		fout << answer[i] << endl;
	}
	cout << "\n" << count << "개의 문제를 입력했습니다." << endl;
	cout << "========================================================" << endl;
	count = 0;
	fout.close();
	return;
}
int text_control::count = 0;
string text_control::p = "";
string text_control::a = "";

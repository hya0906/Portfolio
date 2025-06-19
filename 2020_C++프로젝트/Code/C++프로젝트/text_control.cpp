#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "text_control.h"
using namespace std;


void text_control::read_text(string name, vector<string> &problem, vector<string> &answer, string person)
{
	fstream fin;//
	string text_name = "../" + person + "/����/" + name + ".txt";// ../����/C++.txt
	cout << person << "���� " << name<<"�� ���� �����ϱ��Դϴ�." << endl;//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
		cout << "�������� �ʴ� �����Դϴ�" << endl << endl;
	fin.close();
}
//name�̶�� �Ű������� �̿��Ͽ� ���� ������ '����'����
//�ؽ�Ʈ�� �о�ͼ� �Ű����� ����(problem�� answer)�� ������ ������ �Է�
void text_control::make_text(string name, vector<string> problem, vector<string> answer, string person)
{
	fstream fout;
	string text_name = "../" + person + "/�����Ʈ/" + name + ".txt";
	cout << text_name<<endl;//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	fout.open(text_name, ios::out);
	char temp[] = "\n";
	for (int i = 0; i < int(problem.size()); i++)
	{
		fout.write(problem.at(i).c_str(), problem.at(i).size());
		fout.write(temp, 1);//'\n' �Է����� ����
		fout.write(answer.at(i).c_str(), answer.at(i).size());
		fout.write(temp, 1);
	}
	fout.close();
}
//name�̶�� �Ű������� �ؽ�Ʈ������ �̸������ϰ�, ����(problem�� answer)�� ������ ������ ��������
//���� ������ '�����Ʈ'�� �����Ʈ�� �ۼ�
bool text_control::text_check(string name, string person)
{
	fstream fin;
	string text_name = "../" + person + "/����/" + name + ".txt";
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
//�ؽ�Ʈ ������ �ִ����� Ȯ�� �����ϸ� true , �ƴϸ� false�� ����

void text_control::make_test(string name, string person) { //�޸����� ������쿡�� ����� ������ �����ϴ� ���
	cout << "���������� �׸��� ��� �����κп��� exit�� �Է����ּ���" << endl;
	string text_name = "../" + person + "/����/" + name + ".txt";
	ofstream fout(text_name, ios::out | ios::app);
	vector<string> problem, answer;
	while (1) {
		cout << "\n����>> ";
		getline(cin, p);//p,a����ӹٲ�
		if (p == "exit")
			break;
		cout << "��>>";
		getline(cin, a);
		problem.push_back(p);//����ӳ���
		answer.push_back(a);
		count++;
	}
	for (int i = 0; i < count; i++) {//�޸��忡 �������
		fout << problem[i] << endl;
		fout << answer[i] << endl;
	}
	cout << "\n" << count << "���� ������ �Է��߽��ϴ�." << endl;
	cout << "========================================================" << endl;
	count = 0;
	fout.close();
	return;
}
int text_control::count = 0;
string text_control::p = "";
string text_control::a = "";

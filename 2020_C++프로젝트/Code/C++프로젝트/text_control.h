#ifndef TEXTCONTROL
#define TEXTCONTROL
#include <vector>
#include <string>
using namespace std;

class text_control {
	static int count;
	static string p, a;
public:
	static void read_text(string name, vector<string> &problem, vector<string> &answer, string person); //txt파일에서 문제와 정답을 가져 옵니다.
	static void make_text(string name, vector<string> problem, vector<string> answer, string person);
	static bool text_check(string name, string person);
	static void make_test(string name, string person);

};

#endif 

/*
위의 text_control이라는 클래스는 텍스트파일 관리를 담당하며
텍스트의 내용을 읽어오거나, 새로 생성하며, 텍스트의 존재 여부를 확인한다.

*/
#ifndef TEXTCONTROL
#define TEXTCONTROL
#include <vector>
#include <string>
using namespace std;

class text_control {
	static int count;
	static string p, a;
public:
	static void read_text(string name, vector<string> &problem, vector<string> &answer, string person); //txt���Ͽ��� ������ ������ ���� �ɴϴ�.
	static void make_text(string name, vector<string> problem, vector<string> answer, string person);
	static bool text_check(string name, string person);
	static void make_test(string name, string person);

};

#endif 

/*
���� text_control�̶�� Ŭ������ �ؽ�Ʈ���� ������ ����ϸ�
�ؽ�Ʈ�� ������ �о���ų�, ���� �����ϸ�, �ؽ�Ʈ�� ���� ���θ� Ȯ���Ѵ�.

*/
#ifndef CONSOLE
#define CONSOLE
#include <string>
#include <vector>
using namespace std;
class Console {
	static char name[];
	static int c;
	static string yn;
public:
	static string text_name;
	static string newperson_main();
	static void getMain();
	static void get_text_name(string person);
	static void do_review_q_main();
	static void after_test(int num);
	static string pop_text_name();
	static int pop_user_put();
	static void make_q_main();
	static int getint(string in);

};

#endif

/*
Console�̶� Ŭ������ ������� ���÷��̸� ����ϸ�,
���ÿ� ������� �Է°��� �޾� �����ϴ� Ŭ�����Դϴ�.


*/
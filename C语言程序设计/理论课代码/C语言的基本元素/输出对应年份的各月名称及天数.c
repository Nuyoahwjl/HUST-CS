#include <stdio.h>

int main() {
	/**********Begin**********/


	int n;
	scanf("%d", &n);
	if (n%4==0&&n%100!=0||n%400==0)
		printf(
			"Jan: 31\n"
			"Feb: 29\n"
			"Mar: 31\n"
			"Apr: 30\n"
			"May: 31\n"
			"Jun: 30\n"
			"Jul: 31\n"
			"Aug: 31\n"
			"Sep: 30\n"
			"Oct: 31\n"
			"Nov: 30\n"
			"Dec: 31\n"
		);
	else
		printf(
			"Jan: 31\n"
			"Feb: 28\n"
			"Mar: 31\n"
			"Apr: 30\n"
			"May: 31\n"
			"Jun: 30\n"
			"Jul: 31\n"
			"Aug: 31\n"
			"Sep: 30\n"
			"Oct: 31\n"
			"Nov: 30\n"
			"Dec: 31\n"
		);



	/**********End**********/
	return 0;
}
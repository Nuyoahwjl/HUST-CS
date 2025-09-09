#include <iostream>
#include <cstring>
using namespace std;
#define N 5
#define N1 2
#define N2 3
struct student{
    char name[8];
    short age;
    float score;
    char remark[200]; // ��ע��Ϣ
};

void input(student *s){
    for(int i=0;i<N;++i){
        cout<<"�������"<<i+1<<"��ѧ������Ϣ��"<<endl;
        cout<<"������";
        cin>>s[i].name;
        cout<<"���䣺";
        cin>>s[i].age;
        cout<<"�ɼ���";
        cin>>s[i].score;
        cout<<"��ע��";
        cin.ignore(); // ���Ի��з�
        cin.getline(s[i].remark, 200);
    }   
}

int pack_student_bytebybyte(student *s,int sno,char *buf){
    int len=0;
    for(int i=0;i<sno;++i){
        // memcpy(buf+len,s[i].name,8);
        // len+=8;
        // memcpy(buf+len,&s[i].age,sizeof(short));
        // len+=sizeof(short);
        // memcpy(buf+len,&s[i].score,sizeof(float));
        // len+=sizeof(float);
        // memcpy(buf+len,s[i].remark,200);
        // len+=200;
        for (int j = 0; j < 8; ++j) buf[len++] = s[i].name[j];
        buf[len++] = (char)(s[i].age & 0xFF);
        buf[len++] = (char)((s[i].age >> 8) & 0xFF);
        char* p = (char*)&s[i].score;
        for (int j = 0; j < static_cast<int>(sizeof(float)); ++j) buf[len++] = p[j];
        for (int j = 0; j < 200; ++j) buf[len++] = s[i].remark[j];
    }
    return len;
}

int pack_student_whole(student *s,int sno,char *buf){
    int len=0;
    for(int i=0;i<sno;++i){
        strcpy(buf+len,s[i].name);
        len+=8;
        *(short*)(buf + len) = s[i].age;
        len += sizeof(short);
        *(float*)(buf + len) = s[i].score;
        len += sizeof(float);
        strcpy(buf + len, s[i].remark);
        len += 200;
    }
    return len;
}

int restore_student(char *buf, int len, student* s){
    int num=0;
    int pos=0;
    while(pos<len){
        memcpy(s[num].name,buf+pos,8);
        pos+=8;
        s[num].age=*(short*)(buf+pos);
        pos+=sizeof(short);
        s[num].score=*(float*)(buf+pos);
        pos+=sizeof(float);
        memcpy(s[num].remark,buf+pos,200);  
        pos+=200;   
        num++;
    }
    return num;
}

void output(student *s,int count){
    for(int i=0;i<count;++i){
        cout<<"��"<<i+1<<"��ѧ������Ϣ��"<<endl;
        cout<<"������"<<s[i].name<<endl;
        cout<<"���䣺"<<s[i].age<<endl;
        cout<<"�ɼ���"<<s[i].score<<endl;
        cout<<"��ע��"<<s[i].remark<<endl;
    }
}

void print_message(char *buf, int len){
    cout<<"message��ǰ20���ֽ�:"<<endl;
    for(int i=0;i<len&&i<20;++i){
        printf("%02X ", (unsigned char)buf[i]);
    }
    cout<<endl;
}

int main(){
    struct student old_s[N], new_s[N];
    char message[2000];
    int packed_len=0;

    input(old_s);
    system("cls");

    packed_len=pack_student_bytebybyte(old_s,N1,message);
    packed_len+=pack_student_whole(old_s,N2,message+packed_len);

    cout<<"---------------------------"<<endl;
    print_message(message,packed_len);

    int num=restore_student(message,packed_len,new_s);

    cout<<"---------------------------"<<endl;
    cout<<"ԭ����ѧ����Ϣ��"<<endl;
    output(old_s,N);
    cout<<"---------------------------"<<endl;
    cout<<"�ָ����ѧ����Ϣ��"<<endl;
    output(new_s,num);

    return 0;
}
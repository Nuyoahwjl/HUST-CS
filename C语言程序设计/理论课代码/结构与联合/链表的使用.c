#include<stdio.h>
#include<stdlib.h>
#include<string.h>
struct list{
	char data;
	struct list *next;
};
int main()
{
	char c;
	char target;
	/*建立先进先出链表*/ 
	struct list *head=NULL,*tail;
    c=getchar();
	head=(struct list *)malloc(sizeof(struct list));
    head->data=c;
    tail=head;
        while ((c=getchar())!='\n')
        {
            tail->next=(struct list *)malloc(sizeof(struct list));
            tail=tail->next;
            tail->data=c;
        }
        tail->next=NULL;
    /******************/    
		scanf("%c",&target);
    /*输出链表并统计字符个数*/    
        int len=0;
		struct list *p=head;
        while(p!=NULL){
        	printf("%c ",p->data);
        	len++;
        	p=p->next;
		}
		printf("\n");
    /***********************/    
    /*将字符串无冗余存入字符数组并输出*/    
		char *q=(char *)malloc(len+1);
    	p=head;
    	while(p!=NULL){
        	*q=p->data;
        	 p=p->next;
        	 q++;
		}
		*q='\0';
		printf("%s\n",q-len);
	/**********************************/
	/*若target在链表中，则删除这个节点*/ 
	    int flag=0;
	    struct list *last=NULL;
	    while (head&&head->data == target)
	{
		p = head->next;
		free(head);
		head=p;
		flag=1;
	}
	    for(p=head;p->next!=NULL;){
	    	last=p;
	    	p=p->next;
	    	if(p->data==target){
	    	    last->next=p->next;
	    	    flag=1;
	    	    free(p);
	    	    p=last;
//				break;
			}
		}

		
        if(flag==0){
        struct list *k=NULL,*m=NULL;
		int a=100,gap=0;
		p=head;
		for(;p!=NULL;p=p->next){
			gap=p->data-target;
			if(abs(gap)<a){
				a=abs(gap);
				k=p;
				m=p->next;
				}
		}
		struct list *temp=(struct list *)malloc(sizeof(struct list));
		temp->data=target;
		k->next=temp;
		temp->next=m;
	}
		
		p=head;
	    while(p!=NULL){
        	printf("%c",p->data);
        	p=p->next;
		}
	return 0;
}
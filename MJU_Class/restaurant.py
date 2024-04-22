from person import *

## 도착 시간들 정의함 ##

ARRIVAL_TIMES = [1, 3, 5, 10, 15, 18, 25, 30,
                 40, 52, 55, 60, 70, 75, 80, 85, 94, 98]

### START CODE HERE ###
# TODO : Define Restaurant Class

class Restaurant:
    # restaurant에 seat수,employee수, closing_time 입력
    def __init__(self, num_seats, num_employees, closing_time):
        self.num_seats = num_seats
        self.num_employees = num_employees
        self.closing_time = closing_time
        self.current_time = 0
        self.employees=[]
        self.visitors = []
        self.total_visitors_served = []
        for i in range(num_employees):
            # emplyee들의 person_id생성 후 list에 저장
            self.employees.append(Employee(i))
 
    def num_seats_available(self):
        # seat의 수에서 visitor의 수를 빼면 사용가능한 좌석
        return self.num_seats - len(self.visitors)

    def visitor_arrived(self, visitor_obj):
        #새로 온 손님이 앉을 자리가 있으면
        if self.num_seats_available() > 0:
            #visitors attribute와 total_visitors_served attribute에 visitor_obj 파라미터 객체를 추가함.
            self.visitors.append(visitor_obj)
            self.total_visitors_served.append(visitor_obj)
        #손님을 받을 수 없으면
        else:
            #visitor_obj 파라미터 객체의 leave() 메서드를 호출함.
            visitor_obj.leave()
   
#식당에 도착 후 아직 주문 하지 못한 손님들의 음식을 요리하기 시작함.
#- 손님 객체들 중에서 현재 상태가 ARRIVAL인 객체
#위의 두 객체가 존재할 경우,
#cook 메서드를 호출함으로써 두 객체의 현재 상태가 자동으로 변화됨.
#대기 중인 종업원이 없거나
#주문을 하지 못한 손님이 없을 때까지
#위의 과정을 재귀적으로 반복함.

    def order_process(self):
        for employee in self.employees:
            #- 종업원 객체들 중에서 현재 상태가 WAITING인 객체
            if employee.state == WAITING:
                for visitor in self.visitors:
                    if visitor.state == ARRIVAL:
                        #대기 중인 종업원 객체의 메서드 cook을 호출하며 음식의 주문자인 손님 객체를 argument로 대입함.
                        employee.cook(visitor)
                        return

    def after_one_minute(self):
        for visitor in self.visitors:
            # vist에게는 vist after_one_min을 적용시켜서 remaining_eating_time을 -1씩 감소시킴
            visitor.after_one_minute()

            if visitor.state == LEAVING:
                self.visitors.pop(0)
            
        for employee in self.employees:
            # emp에게는 emp after_one_min을 적용시켜서 remainng_cooking_time을 -1씩 감소시킴
            employee.after_one_minute()

    def operate(self):

        self.current_time+=1  # 현재 시각을 1 증가시키고, 출력함.
        print("Current Time:", self.current_time)
        
        # 증가된 현재 시각이 식당 운영 마감 시각과 같으면,
        # summarize_result 메서드를 호출함.
        if(self.current_time == self.closing_time):
            self.summarize_result()
            return 
        
        # 현재 시각이 손님이 방문하는 시각인 경우 (ARRIVAL_TIMES에 의거해서)
        if self.current_time in ARRIVAL_TIMES:
            
            # 현재 시각을 argument로 대입하여 손님 객체를 생성하고,
            visitor = Visitor(len(self.total_visitors_served) + 1, self.current_time)

            # 이를 visitor_arrived 메서드에 대입함.
            self.visitor_arrived(visitor)
            print("\t new visitor. now available seats: ",self.num_seats_available())
            
        # order_process 메서드와 after_one_minute 메서드를
        # 호출하여 단위 시간 동안의 변화를 반영함.
        self.order_process()
        self.after_one_minute()
        
        for visitor in self.visitors:
            print("\t",visitor)
            
        for employee in self.employees:
            print("\t",employee)
        
        # 함수를 재귀적으로 구성하여 위의 과정을 반복
        self.operate()
        
    def summarize_result(self):
        served_visitors = len(self.total_visitors_served)
        unserved_visitors = 0
        total_waiting_time = 0
        total_employee_waiting_time = 0

        for v in self.total_visitors_served:
            if v.state == LEAVING:
                unserved_visitors += 1
            total_waiting_time += v.waiting_time

        if served_visitors > 0:
            total_waiting_time /= served_visitors
        else:
            total_waiting_time = 0

        if self.num_employees > 0:
            total_employee_waiting_time = sum(
                [e.waiting_time for e in self.employees]) / self.num_employees
        else:
            total_employee_waiting_time = 0

        total_employee_working_time = self.closing_time - \
            sum([e.waiting_time for e in self.employees])

        print("served:", served_visitors, ",",
              "not served:", len(ARRIVAL_TIMES) - served_visitors)
        print("visitors average waiting times:", total_waiting_time)
        print("employees average waiting time:", total_employee_waiting_time)
        print("employees average working time:", total_employee_working_time)

        ### END CODE HERE ###


if __name__ == '__main__':
    rest = Restaurant(5, 2, 100)
    rest.operate()
    

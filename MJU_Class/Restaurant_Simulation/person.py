## state들을 정의함 ##
ARRIVAL = 'arrival'
WAITING = 'waiting'
EATING = 'eating'
WORKING = 'working'
LEAVING = 'leaving'

### START CODE HERE ###

# TODO 1 : Define Person Class


class Person:
    def __init__(self, person_id):            # person_id를 파라미터로 입력 받음  self.person_id = person_id
        self.person_id = person_id
        self.waiting_time = 0
        self.state = None

# TODO 2 : Define Visitor Class


class Visitor(Person):
    def __init__(self, person_id, arrival_time):
        super().__init__(person_id)          # person class를 상속받음
        self.state = ARRIVAL                 # 도착했다는 state로 됨
        self.remaining_eating_time = 0
        self.arrival_time = arrival_time

    #state attribute의 값을 상수 WAITING으로 변경
    def order(self):
        self.state = WAITING

    def eat(self):
        self.state = EATING
        if self.remaining_eating_time == 0:
            self.remaining_eating_time = 20

    def leave(self):
        self.state = LEAVING


    def after_one_minute(self):
        #객체의 상태가 EATING일 경우, 음식을 다 먹기 위해 소요되는 시간이 남았으면
        if self.state == EATING and self.remaining_eating_time > 0:     # eating 중이고 먹을 시간이 남아 있으면
            self.eat()
            #이를 1 감소시키고
            self.remaining_eating_time -= 1
            # 다 먹었으면
        elif self.state == EATING and self.remaining_eating_time <= 0:
            # leave() 메서드를 호출함. 
            self.leave()

        #객체의 상태가 ARRIVAL 또는 WAITING일 경우, 객체의 누적 대기 시간을 1증가시킴.

        elif self.state == WAITING or ARRIVAL :
            self.waiting_time += 1


    def __str__(self):
        return "visitor arrived at {}, {}".format(self.arrival_time, self.state)

# TODO 3 : Define Employee Class


class Employee(Person):
    def __init__(self, person_id):               # person을 상속
        super().__init__(person_id)
        self.state = WAITING
        self.remaining_cooking_time = 0
        self.visitor = None


#visitor 파라미터는 음식을 주문한 손님이며,
    def cook(self, visitor):
        #이 손님 객체의 메서드 order()를 호출함.
        visitor.order()
        #요리를 시작하게 되었으므로, 종업원 객체의 상태를 상수 WORKING으로 변경.
        self.state = WORKING
        #객체의 음식을 완성하는 데 소요되는 시간을 15로 변경.
        self.remaining_cooking_time = 15
        #visitor attribute에 음식의 주문자인 visitor 파라미터를 할당.
        self.visitor = visitor
    

    def serve(self):
        if self.visitor:
            self.visitor.eat()
            self.state = WAITING
            self.visitor = None



    def after_one_minute(self):
        # 객체의 상태가 WORKING일 경우,
        if self.state == WORKING:
            # 음식을 완성하기 위해 소요되는 시간이 남았으면
            if self.remaining_cooking_time > 0:
                # 이를 1 감소시키고, 완성하였으면 serve() 메서드를 호출.
                self.remaining_cooking_time -= 1
            else:
                self.serve()
        # 객체의 상태가 WORKING이 아닐 경우,
        else:
            # 객체의 누적 대기 시간을 1증가시킴.
            self.waiting_time += 1



    def __str__(self):
        return "employee id {}, {}".format(self.person_id, self.state)

        ### END CODE HERE ###
